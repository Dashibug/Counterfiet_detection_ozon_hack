import yaml, json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import os; os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.data.dataset import MultimodalDataset
from src.image_model.image_encoder import ImageEncoder
from src.text_model.text_encoder import TextEncoder
from src.fusion.multimodal_classifier import MultimodalClassifier
from src.utils.metrics import compute_metrics
from tqdm import tqdm

cb = None

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path="configs/config.yaml"):
    # --- CONFIG ---
    cfg = yaml.safe_load(open(config_path))
    seed_everything(cfg.get("seed", 42))
    device = cfg["train"]["device"]

    # --- DATA ---
    df = pd.read_csv(cfg["data"]["csv_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_model"])

    numeric_cols = [c for c in df.columns if c.startswith("num_")]
    #meta_dim = len(numeric_cols) + 1  # + has_image

    # --- ENCODERS ---
    # CLIP image encoder: берём правильный transform прямо из энкодера
    clip_model_name = cfg["model"].get("clip_model_name", "ViT-B-32")
    clip_pretrained = cfg["model"].get("openclip_pretrained", "laion2b_s34b_b79k")

    image_extractor = ImageEncoder(
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        device=device,
        normalize=True,
        # dtype=torch.float16,  # можно включить для ускорения на GPU
    )
    img_tf = image_extractor.transform

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
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract features"):
            images = batch["image"].to(device)
            meta = batch["meta"].to(device)

            # CLIP image embeddings
            img_emb = image_extractor(images)  # [B, D_img]

            # RuBERT mean-pooling
            toks = {k: v.to(device) for k, v in batch["text"].items()}
            out = text_encoder.model(**toks, return_dict=True)
            last_hidden = out.last_hidden_state  # [B, L, 768]
            attn = toks["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]
            txt_emb = (last_hidden * attn).sum(1) / attn.sum(1)  # [B, 768]

            imgs.append(img_emb.cpu().numpy())
            txts.append(txt_emb.cpu().numpy())
            metas.append(meta.cpu().numpy())
            ys.append(batch["label"].cpu().numpy())

    X_img = np.vstack(imgs).astype(np.float32)
    X_txt = np.vstack(txts).astype(np.float32)
    X_meta = np.vstack(metas).astype(np.float32) if metas else np.empty((len(df), 0), dtype=np.float32)
    y = np.concatenate(ys).astype(np.int32)

    X = np.hstack([X_txt, X_img, X_meta]).astype(np.float32)
    print("Shapes -> text:", X_txt.shape, "| image:", X_img.shape, "| meta:", X_meta.shape, "| X:", X.shape, "| y:",
          y.shape)

    # сохраним порядок numeric_cols для predict
    json.dump(numeric_cols, open("numeric_cols.json", "w"))

    # --- CATBOOST TRAIN ---
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=cfg.get("seed", 42), stratify=y
    )
    params = dict(
        iterations=cfg["catboost"]["iterations"],
        learning_rate=cfg["catboost"]["learning_rate"],
        depth=cfg["catboost"]["depth"],
        l2_leaf_reg=cfg["catboost"]["l2_leaf_reg"],
        eval_metric=cfg["catboost"]["eval_metric"],
        loss_function=cfg["catboost"]["loss_function"],
        early_stopping_rounds=cfg["catboost"]["early_stopping_rounds"],
        random_seed=cfg.get("seed", 42),
        task_type="GPU" if (cfg["catboost"]["use_gpu"] and torch.cuda.is_available()) else "CPU",
        verbose=100,
    )
    cb = CatBoostClassifier(**params)
    cb.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True)
    cb.save_model("cb_model.cbm")
    va_probs = cb.predict_proba(Xva)[:, 1]
    metrics = compute_metrics(yva, va_probs)
    print(f"VAL  AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

    print("✔ Saved: cb_model.cbm, numeric_cols.json")


if __name__ == "__main__":
    main()
