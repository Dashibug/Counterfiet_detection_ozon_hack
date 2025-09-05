# src/predict.py
import os, argparse, yaml, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import joblib


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from catboost import CatBoostClassifier

from src.data.dataset import MultimodalDataset
from src.image_model.image_encoder import ImageEncoder   # CLIP
from src.text_model.text_encoder import TextEncoder      # RuBERT

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


def collate_fn(batch):
    import torch
    images = torch.stack([b["image"] for b in batch])                       # [B,3,H,W]
    metas  = torch.stack([b["meta"]  for b in batch]) if batch[0]["meta"].numel() else torch.empty(len(batch), 0)
    ids    = [int(b["id"]) for b in batch]
    item_ids = [int(b["item_id"]) for b in batch]
    toks = {}
    if "text" in batch[0] and batch[0]["text"]:
        keys = batch[0]["text"].keys()
        toks = {k: torch.stack([b["text"][k] for b in batch]) for k in keys}
    texts_str = [b["text_str"] for b in batch] if "text_str" in batch[0] else None
    out = {"image": images, "meta": metas, "text": toks, "text_str": texts_str, "id": ids, "item_id": item_ids}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out


def save_submission(ids, probs, out_csv, test_csv, thr=0.5):
    ids = np.asarray(ids, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= thr).astype(np.int8)
    sub = pd.DataFrame({"id": ids, "prediction": preds})

    df_test = pd.read_csv(test_csv)
    if "id" not in df_test.columns:
        raise ValueError("В тестовом CSV нет столбца 'id' — он обязателен для сабмита.")
    order = df_test["id"].astype(np.int64).values

    # проверки
    assert sub["id"].is_unique, "id должны быть уникальны"
    assert set(sub["prediction"].unique()).issubset({0, 1})

    sub = sub.set_index("id").loc[order].reset_index()
    sub.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv} | rows={len(sub)} | pos_rate={sub['prediction'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="configs/config.yaml")
    ap.add_argument("--test_csv", default="test_processed.csv")
    ap.add_argument("--out_csv",  default="submission.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = cfg["train"]["device"]

    # --- DATAFRAME ---
    df = pd.read_csv(args.test_csv)

    # --- numeric_cols порядок как в train ---
    with open("numeric_cols.json") as f:
        numeric_cols_train = json.load(f)
    # создаём недостающие num_* в тесте
    for c in numeric_cols_train:
        if c not in df.columns:
            df[c] = 0.0
    numeric_cols = numeric_cols_train

    # --- ENCODERS ---
    # CLIP (берём правильный preprocess из энкодера)
    clip_model_name = cfg["model"].get("clip_model_name", "ViT-B-32")
    clip_pretrained = cfg["model"].get("openclip_pretrained", "laion2b_s34b_b79k")
    use_clip_text = bool(cfg["model"].get("use_clip_text", False))

    image_extractor = ImageEncoder(
        model_name=clip_model_name,
        pretrained=clip_pretrained,
        device=device,
        normalize=True,
        # dtype=torch.float16,  # при желании ускорить на GPU
    )
    img_tf = image_extractor.transform
    image_extractor.eval()

    # RuBERT (как в train)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_model"])
    text_encoder = TextEncoder(model_name=cfg["model"]["text_model"], device=device)
    text_encoder.model.eval()
    for p in text_encoder.model.parameters():
        p.requires_grad_(False)

    if use_clip_text:
        clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    # --- DATASET / LOADER ---
    dataset = MultimodalDataset(
        df=df,
        images_root=cfg["data"]["images_root"],
        tokenizer=tokenizer,
        numeric_cols=numeric_cols,
        image_transform=img_tf,                       # ВАЖНО: CLIP preprocess
        text_max_len=cfg["model"]["text_max_len"],
        label_col=None,                               # тест без лейблов
        add_has_image=True,                           # как в train
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn
    )

    # --- CatBoost ---
    cb = CatBoostClassifier()
    cb.load_model("cb_model.cbm")

    # --- EXTRACT FEATURES ---
    ids, all_img, all_txt, all_meta = [], [], [], []
    all_clip_txt = []  # если use_clip_text
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extract features (image+text+meta)"):
            images = batch["image"].to(device)
            meta   = batch["meta"].to(device)

            # CLIP image embeddings
            do_flip = bool(cfg.get("features", {}).get("image_tta_flip", False))
            img_emb = encode_image_with_tta(image_extractor, images, do_flip=do_flip)
            if isinstance(img_emb, (tuple, list)):
                img_emb = img_emb[0]
            img_emb = img_emb.to(torch.float32)  # как в train.py # [B, D_img]

            # RuBERT mean-pooling
            toks = {k: v.to(device) for k, v in batch["text"].items()} if batch["text"] else {}
            if toks:
                out = text_encoder.model(**toks, return_dict=True)
                last_hidden = out.last_hidden_state
                attn = toks["attention_mask"].unsqueeze(-1).float()
                txt_emb = (last_hidden * attn).sum(1) / attn.sum(1)  # [B, 768]
            else:
                # на всякий случай, если токенов не будет
                txt_emb = torch.zeros(images.size(0), 768, device=device)

            # CLIP text embeddings (опц.)
            if use_clip_text:
                texts = batch.get("text_str", None)
                if texts:
                    clip_tok = clip_tokenizer(texts).to(device)
                    clip_txt = image_extractor.model.encode_text(clip_tok)
                    if isinstance(clip_txt, (tuple, list)):
                        clip_txt = clip_txt[0]
                    clip_txt = torch.nn.functional.normalize(clip_txt, dim=-1).to(torch.float32)
                else:
                    d = int(getattr(image_extractor.model, "text_embed_dim", 512))
                    clip_txt = torch.zeros(images.size(0), d, device=device)

            all_img.append(img_emb.cpu().numpy())
            all_txt.append(txt_emb.cpu().numpy())
            all_meta.append(meta.cpu().numpy())
            if use_clip_text:
                all_clip_txt.append(clip_txt.cpu().numpy())
            ids.extend(batch["id"])

    X_img = np.vstack(all_img).astype(np.float32)
    if os.path.exists("pca_image.joblib"):
        pca = joblib.load("pca_image.joblib")
        X_img = pca.transform(X_img)
        print(f"Applied PCA(image): -> {X_img.shape}")

    X_txt = np.vstack(all_txt).astype(np.float32)  # RuBERT
    X_meta = np.vstack(all_meta).astype(np.float32) if all_meta else np.empty((len(df), 0), dtype=np.float32)
    if use_clip_text:
        X_clip_txt = np.vstack(all_clip_txt).astype(np.float32)
        X = np.hstack([X_txt, X_clip_txt, X_img, X_meta]).astype(np.float32)
    else:
        X = np.hstack([X_txt, X_img, X_meta]).astype(np.float32)

    # --- PREDICT ---
    probs = cb.predict_proba(X)[:, 1]
    print("Rows:", len(ids), "| X:", X.shape, "| prob stats:",
          float(probs.min()), float(probs.mean()), float(probs.max()))

    save_submission(ids, probs, args.out_csv, args.test_csv, thr=float(cfg["catboost"]["threshold"]))


if __name__ == "__main__":
    main()
