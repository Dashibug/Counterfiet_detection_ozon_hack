# src/predict.py
import os, argparse, yaml, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from src.data.dataset import MultimodalDataset
from src.fusion.multimodal_classifier import MultimodalClassifier
from src.image_model.image_encoder import ImageEncoder
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def collate_fn(batch):
    import torch
    images = torch.stack([b["image"] for b in batch])
    metas  = torch.stack([b["meta"]  for b in batch])
    keys   = batch[0]["text"].keys()
    toks   = {k: torch.stack([b["text"][k] for b in batch]) for k in keys}
    ids = [int(b["id"]) for b in batch]  # <-- сабмит-id
    item_ids = [int(b["item_id"]) for b in batch]
    out = {"image": images, "meta": metas, "text": toks, "id": ids, "item_id": item_ids}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch])
    return out

# def save_submission(ids, probs, out_csv, test_csv, thr=0.5):
#     preds = (np.asarray(probs) >= thr).astype(int)
#     sub = pd.DataFrame({"id": np.asarray(ids, dtype=np.int64),
#                         "prediction": preds.astype(np.int8)})
#
#     # проверки формата/полного покрытия
#     assert sub["id"].is_unique, "id должны быть уникальны"
#     assert set(sub["prediction"].unique()).issubset({0, 1})
#     df_test = pd.read_csv(test_csv)
#     order = df_test["ItemID"].astype(np.int64).values
#     missing = set(order) - set(sub["id"])
#     if missing:
#         raise ValueError(f"В сабмите отсутствуют {len(missing)} id, напр. {list(missing)[:5]}")
#
#     sub = sub.set_index("id").loc[order].reset_index()
#     sub.to_csv(out_csv, index=False)
#     print(f"✅ Saved {out_csv} | rows={len(sub)} | pos_rate={sub['prediction'].mean():.4f}")
#     return sub
def save_submission(ids, probs, out_csv, test_csv, thr=0.5):
    ids = np.asarray(ids, dtype=np.int64)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= thr).astype(np.int8)
    sub = pd.DataFrame({"id": ids, "prediction": preds})

    # проверки формата/полного покрытия/типа
    assert sub["id"].is_unique, "id должны быть уникальны"
    assert set(sub["prediction"].unique()).issubset({0, 1})
    df_test = pd.read_csv(test_csv)
    if "id" not in df_test.columns:
        raise ValueError("В тестовом CSV нет столбца 'id' — он обязателен для сабмита.")
    order = df_test["id"].astype(np.int64).values
    missing = set(order) - set(sub["id"])
    if missing:
        raise ValueError(f"В сабмите отсутствуют {len(missing)} id, напр. {list(sorted(missing))[:5]}")

    # приводим к порядку из теста по 'id'
    sub = sub.set_index("id").loc[order].reset_index()
    sub.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv} | rows={len(sub)} | pos_rate={sub['prediction'].mean():.4f}")
    return sub

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt",   default="ckpt_interrupt.pt")
    # если у тебя есть test_processed.csv — укажи его здесь:
    ap.add_argument("--test_csv", default="test_processed.csv")
    ap.add_argument("--out_csv",  default="submission.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = cfg["train"]["device"]

    df = pd.read_csv(args.test_csv)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_model"])

    # numeric_cols должны совпадать с обучением (например, num_0..num_39)
    numeric_cols = [c for c in df.columns if c.startswith("num_")]
    meta_dim = len(numeric_cols) + 1

    img_tf = transforms.Compose([
        transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    dataset = MultimodalDataset(
        df=df,
        images_root=cfg["data"]["images_root"],
        tokenizer=tokenizer,
        numeric_cols=numeric_cols,
        image_transform=img_tf,
        text_max_len=cfg["model"]["text_max_len"],
        label_col=None,
        add_has_image=True,
    )
    loader = DataLoader(dataset,
                        batch_size=cfg["train"]["batch_size"],
                        shuffle=False,
                        num_workers=cfg["data"]["num_workers"],
                        collate_fn=collate_fn)

    # модели
    image_extractor = ImageEncoder("resnet50", pretrained=True, device=device)
    text_model = AutoModel.from_pretrained(cfg["model"]["text_model"]).to(device)
    text_model.eval()
    clf = MultimodalClassifier(image_dim=image_extractor.out_dim,
                               text_dim=768,
                               meta_dim=meta_dim,
                               hidden_size=cfg["model"]["fusion_hidden"]).to(device)

    # загрузка весов
    ckpt = torch.load(args.ckpt, map_location=device)
    clf.load_state_dict(ckpt["clf_state"])
    clf.eval()

    ids, preds = [], []
    # после чтения df теста
    print("Test rows:", len(df))
    print("Has 'id' col:", "id" in df.columns)
    print("Has 'ItemID' col:", "ItemID" in df.columns)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"): #  забираем данные из батча
            images = batch["image"].to(device) # Tensor [B,3,H,W]
            meta = batch["meta"].to(device) # Tensor [B, num_meta]
            toks = {k: v.to(device) for k, v in batch["text"].items()} # dict input_ids, attention_mask

            img_emb = image_extractor(images) # эмбеббинги картинки
            out = text_model(**toks, return_dict=True) # эмбеббинги текста
            last_hidden = out.last_hidden_state # [B, L, 768]
            attn = toks["attention_mask"].unsqueeze(-1).float() # [B, L, 1]
            txt_emb = (last_hidden * attn).sum(1) / attn.sum(1) # [B, 768]

            logits = clf(img_emb, txt_emb, meta) # объединяем модальности и считаем логиты
            probs = torch.sigmoid(logits) # логит в метку
            preds.extend(probs.detach().cpu().numpy().tolist()) # preds - список предсказаний для всех объектов теста.

            ids.extend(batch["id"]) # список id (ключей для сабмита, не путать с ItemID, который нужен только для поиска картинок).
    # после инференса
    print("Pred rows:", len(ids))
    print("Unique pred ids:", len(set(ids)))
    arr = np.asarray(preds, dtype=float)
    print("probs min/mean/median/max:", arr.min(), arr.mean(), np.median(arr), arr.max())
    print("share >= 0.5:", (arr >= 0.5).mean())

    save_submission(ids, preds, args.out_csv, args.test_csv, thr=0.5)

if __name__ == "__main__":
    main()
