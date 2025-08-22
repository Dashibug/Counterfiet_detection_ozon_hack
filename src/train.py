import yaml, json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from torch.utils.data import DataLoader
#from torchvision import transforms
from transformers import AutoTokenizer
import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.data.dataset import MultimodalDataset
from src.image_model.image_encoder import ImageEncoder
from src.text_model.text_encoder import TextEncoder
#from src.fusion.multimodal_classifier import MultimodalClassifier
from src.utils.metrics import compute_metrics
from tqdm import tqdm
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from sklearn.decomposition import PCA
import joblib
import optuna
from sklearn.model_selection import StratifiedKFold
import open_clip

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def encode_image_with_tta(encoder, images, do_flip: bool):
    emb = encoder(images)
    if isinstance(emb, (tuple, list)):
        emb = emb[0]
    if do_flip:
        emb2 = encoder(TF.hflip(images))
        if isinstance(emb2, (tuple, list)):
            emb2 = emb2[0]
        emb = (emb + emb2) / 2
    return emb


def main(config_path="configs/config.yaml"):
    # --- CONFIG ---
    cfg = yaml.safe_load(open(config_path))
    seed_everything(cfg.get("seed", 42))
    device = cfg["train"]["device"]

    # Куда сохранять CLIP эмбеддинги (npz)
    save_clip_npz = cfg.get("features", {}).get("save_clip_npz", "clip_train_embeds.npz")

    # --- DATA ---
    df = pd.read_csv(cfg["data"]["csv_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_model"])

    numeric_cols = [c for c in df.columns if c.startswith("num_")]
    #meta_dim = len(numeric_cols) + 1  # + has_image

    # --- ENCODERS ---
    # CLIP image encoder: берём правильный transform прямо из энкодера
    clip_model_name = cfg["model"].get("clip_model_name", "ViT-B-32")
    clip_pretrained = cfg["model"].get("openclip_pretrained", "laion2b_s34b_b79k")
    use_clip_text   = bool(cfg["model"].get("use_clip_text", False))
    use_fp16 = False # надо пофиксить

    image_extractor = ImageEncoder(
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        device=device,
        normalize=True
        #dtype=torch.float16 if use_fp16 else None,  # можно включить для ускорения на GPU
    )
    img_tf = image_extractor.transform
    if use_clip_text:
        clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    # RuBERT text encoder (как было)
    text_encoder = TextEncoder(
        model_name=cfg["model"]["text_model"],
        device=device
    )
    text_encoder.model.eval()
    for p in text_encoder.model.parameters():
        p.requires_grad_(False)

    # --- DATASET / LOADER ---
    dataset = MultimodalDataset(
        df=df,
        images_root=cfg["data"]["images_root"],
        tokenizer=tokenizer,  # нужен для токенов текста
        numeric_cols=numeric_cols,
        image_transform=img_tf,  # ВАЖНО: transform от CLIP
        text_max_len=cfg["model"]["text_max_len"],
        add_has_image=True,
        label_col=cfg["data"].get("label_col", "resolution"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,  # для извлечения фич порядок не важен, но так удобнее для воспроизводимости
        num_workers=cfg["data"]["num_workers"]
    )

    # --- EXTRACT FEATURES (image + text + meta) ---
    imgs, txts, metas, ys = [], [], [], []
    ids_list, item_ids_list = [], []
    clip_txts = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract features"):
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)

            # CLIP image embeddings
            do_flip = bool(cfg.get("features", {}).get("image_tta_flip", False))
            img_emb = encode_image_with_tta(image_extractor, images, do_flip=do_flip)
            if isinstance(img_emb, (tuple, list)):
                img_emb = img_emb[0]
            img_emb = img_emb.to(torch.float32) #  [B, D_img]

            # 1. CLIP Text Encoding
            if use_clip_text:
                texts = batch.get("text_str", None)  # список строк (дефолтный collate не склеивает)
                if texts is not None:
                    clip_tok = clip_tokenizer(texts).to(device)
                    clip_txt = image_extractor.model.encode_text(clip_tok)
                    if isinstance(clip_txt, (tuple, list)):
                        clip_txt = clip_txt[0]
                    clip_txt = F.normalize(clip_txt, dim=-1).to(torch.float32)  # [B, D_clip_text]
                else:
                # на всякий случай — паддинг нулями
                    d = int(getattr(image_extractor.model, "text_embed_dim", 512))
                    clip_txt = torch.zeros(images.size(0), d, device=device)


            # 2. RuBERT mean-pooling
            toks = {k: v.to(device) for k, v in batch["text"].items()}
            out = text_encoder.model(**toks, return_dict=True)
            last_hidden = out.last_hidden_state  # [B, L, 768]
            attn = toks["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
            txt_emb = (last_hidden * attn).sum(1) / attn.sum(1)  # [B, 768]

            imgs.append(img_emb.cpu().numpy())
            txts.append(txt_emb.cpu().numpy())
            metas.append(meta.cpu().numpy())
            ys.append(batch["label"].cpu().numpy())
            if use_clip_text:
                clip_txts.append(clip_txt.cpu().numpy())

            b_ids = batch.get("id", None)
            b_item_ids = batch.get("item_id", None)
            if b_ids is not None:
                if isinstance(b_ids, torch.Tensor):
                    ids_list.extend(b_ids.cpu().numpy().astype(np.int64).tolist())
                else:
                    ids_list.extend([int(x) for x in b_ids])
            if b_item_ids is not None:
                if isinstance(b_item_ids, torch.Tensor):
                    item_ids_list.extend(b_item_ids.cpu().numpy().astype(np.int64).tolist())
                else:
                    item_ids_list.extend([int(x) for x in b_item_ids])

    X_img = np.vstack(imgs).astype(np.float32)
    img_pca_dim = int(cfg.get("features", {}).get("image_pca_dim", 0))
    img_pca_whiten = bool(cfg.get("features", {}).get("image_pca_whiten", False))
    if img_pca_dim and X_img.shape[1] > img_pca_dim:
        pca = PCA(n_components=img_pca_dim, whiten=img_pca_whiten, random_state=cfg.get("seed", 42))
        X_img_pca = pca.fit_transform(X_img)
        joblib.dump(pca, "pca_image.joblib")
        print(f"PCA(image): {X_img.shape} -> {X_img_pca.shape} | whiten={img_pca_whiten}")
    else:
        X_img_pca = X_img
    X_txt = np.vstack(txts).astype(np.float32) #rubert
    if use_clip_text:
        X_clip_txt = np.vstack(clip_txts).astype(np.float32)  # CLIP-text
    X_meta = np.vstack(metas).astype(np.float32) if metas else np.empty((len(df), 0), dtype=np.float32)
    y = np.concatenate(ys).astype(np.int32)

    if use_clip_text:
        X = np.hstack([X_txt, X_clip_txt, X_img_pca, X_meta]).astype(np.float32)
        print("Shapes -> RuBERT:", X_txt.shape, "| CLIP-txt:", X_clip_txt.shape,
              "| image:", X_img_pca.shape, "| meta:", X_meta.shape, "| X:", X.shape, "| y:", y.shape)
    else:
        X = np.hstack([X_txt, X_img_pca, X_meta]).astype(np.float32)
        print("Shapes -> RuBERT:", X_txt.shape, "| image:", X_img_pca.shape,
              "| meta:", X_meta.shape, "| X:", X.shape, "| y:", y.shape)

    # --- SAVE CLIP EMBEDDINGS (.npz) ---
    try:
        ids_np = np.asarray(ids_list, dtype=np.int64) if ids_list else np.arange(len(X_img), dtype=np.int64)
        item_ids_np = np.asarray(item_ids_list, dtype=np.int64) if item_ids_list else np.zeros(len(X_img),
                                                                                               dtype=np.int64)
        np.savez_compressed(
            save_clip_npz,
            id=ids_np,
            item_id=item_ids_np,
            img=X_img,  # [N, D_img] float32
            label=y,  # [N] int32 — таргеты тренировки
        )
        print(f"💾 Saved CLIP embeddings: {save_clip_npz} | img shape={X_img.shape}")
    except Exception as e:
        print(f"⚠️ Failed to save CLIP embeddings to '{save_clip_npz}': {e}")

    # сохраним порядок numeric_cols для predict
    json.dump(numeric_cols, open("numeric_cols.json", "w"))

    def tune_catboost_params(X, y, base_params, n_trials=20, n_folds=3, timeout=None):
        # базовые неизменяемые параметры
        fixed = dict(**base_params)

        # некоторые параметры лучше не тюнить через Optuna
        # (eval_metric/loss_function/iterations/early_stopping/seed/task_type/verbose)
        def objective(trial):
            params = dict(
                depth=trial.suggest_int("depth", 4, 10),
                learning_rate=trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                rsm=trial.suggest_float("rsm", 0.6, 1.0),  # subsample features
                random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
                # попробуем и с авто-весами, и без
                auto_class_weights=trial.suggest_categorical("auto_class_weights", [None, "Balanced"]),
            )
            # на GPU grow_policy игнорируется (обливиус-деревья), поэтому не трогаем
            params = {k: v for k, v in params.items() if v is not None}
            params.update(fixed)

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=fixed.get("random_seed", 42))
            aucs = []
            for tr_idx, va_idx in skf.split(X, y):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]
                model = CatBoostClassifier(**params)
                model.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True, verbose=False)
                probs = model.predict_proba(Xva)[:, 1]
                m = compute_metrics(yva, probs)
                aucs.append(m["roc_auc"])
            return float(np.mean(aucs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        print("Optuna best AUC:", study.best_value)
        print("Optuna best params:", study.best_params)
        best_params = dict(**fixed)
        best_params.update({k: v for k, v in study.best_params.items() if v is not None})
        return best_params, study.best_value

    def find_best_threshold(y_true, probs, grid=None):
        if grid is None:
            # более частая сетка в районе 0–0.2, т.к. класс редкий
            grid = np.r_[np.linspace(0.02, 0.20, 19), np.linspace(0.21, 0.80, 60)]
        best_thr, best_f1 = 0.5, -1.0
        for thr in grid:
            pred = (probs >= thr).astype(np.int8)
            m = compute_metrics(y_true, probs)  # считаем F1 внутри? — лучше по preds:
            # если твой compute_metrics требует probs, то F1 он сам посчитает по thr=0.5.
            # Тогда считаем вручную:
            tp = ((y_true == 1) & (pred == 1)).sum()
            fp = ((y_true == 0) & (pred == 1)).sum()
            fn = ((y_true == 1) & (pred == 0)).sum()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        return best_thr, best_f1

    # --- CATBOOST TRAIN ---
    # --- CATBOOST TRAIN (+ Optuna) ---
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=cfg.get("seed", 42), stratify=y
    )

    base_params = dict(
        iterations=cfg["catboost"]["iterations"],
        learning_rate=cfg["catboost"]["learning_rate"],
        depth=cfg["catboost"]["depth"],
        l2_leaf_reg=cfg["catboost"]["l2_leaf_reg"],
        eval_metric=cfg["catboost"]["eval_metric"],
        loss_function=cfg["catboost"]["loss_function"],
        early_stopping_rounds=cfg["catboost"]["early_stopping_rounds"],
        random_seed=cfg.get("seed", 42),
        task_type="GPU" if (cfg["catboost"]["use_gpu"] and torch.cuda.is_available()) else "CPU",
        verbose=False,
        # полезные дефолты:
        bootstrap_type="Bayesian",  # хорошо работает с bagging_temperature
        od_type="Iter",  # стандартный early stop
    )

    tune_cfg = cfg.get("catboost_tune", {})
    if tune_cfg.get("use", False):
        best_params, _ = tune_catboost_params(
            X, y,
            base_params=base_params,
            n_trials=int(tune_cfg.get("trials", 20)),
            n_folds=int(tune_cfg.get("cv_folds", 3)),
            timeout=tune_cfg.get("timeout", None),
        )
    else:
        best_params = base_params

    print("Final CatBoost params:", {k: best_params[k] for k in [
        "depth", "learning_rate", "l2_leaf_reg", "bagging_temperature", "rsm", "random_strength", "auto_class_weights"
    ] if k in best_params})

    cb = CatBoostClassifier(**best_params)
    cb.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True, verbose=100)

    # сохранение модели
    cb.save_model("cb_model.cbm")

    # метрики и лучший порог
    va_probs = cb.predict_proba(Xva)[:, 1]
    metrics = compute_metrics(yva, va_probs)
    thr, f1_thr = find_best_threshold(yva, va_probs)

    print(f"VAL  AUC={metrics['roc_auc']:.4f}  F1@0.50={metrics['f1']:.4f}  | best_thr={thr:.3f}  F1@best={f1_thr:.4f}")

    # сохраним лучший порог (для predict)
    json.dump({"threshold": float(thr)}, open("cb_threshold.json", "w"))
    print("✔ Saved: cb_model.cbm, numeric_cols.json, cb_threshold.json")


if __name__ == "__main__":
    main()
