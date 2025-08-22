"""
    Вытаскиваем картиноч эмбебберы

    item_id — int64 [N]
    id — int64 [N] (если есть столбец id в CSV)
    emb — float32 [N, D] (где D = encoder.out_dim, для resnet50 обычно 2048)
    has_image — float32 [N,1] новая фича (1.0 если файл найден, иначе 0.0)

"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange
import torch
from torchvision import transforms

from src.image_model.image_encoder import ImageEncoder

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".JPG", ".PNG")

def resolve_image_path(images_root:str, item_id: int):
    base = os.path.join(images_root, str(item_id))
    for ext in  IMG_EXTS:
        p = base + ext
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Путь к CSV (train/test) со столбцом ItemID (и, опц., id)")
    ap.add_argument("--images_root", required=True, help="Корневая папка с изображениями по ItemID")
    ap.add_argument("--out", default="image_embeds.npz", help="Куда сохранить npz с эмбеддингами")
    ap.add_argument("--backbone", default="resnet50", help="Модель timm (по умолчанию resnet50)")
    ap.add_argument("--pretrained", type=int, default=1, help="1/0 — грузить предобученные веса timm")
    ap.add_argument("--device", default="cuda", help="cuda или cpu")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=224, help="Размер картинки на вход модели")
    ap.add_argument("--id_col", default="ItemID", help="Колонка, по которой искать файл картинки (обычно ItemID)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.id_col not in df.columns:
        raise ValueError(f"В CSV нет колонки '{args.id_col}'")

    item_ids = df[args.id_col].astype(np.int64).values
    submit_ids = df["id"].astype(np.int64).values if "id" in df.columns else None

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # инициализация энкодера
    enc = ImageEncoder(backbone=args.backbone, pretrained=bool(args.pretrained), device=device)
    enc.eval()
    # если хочешь другой размер — обновим трансформ у энкодера
    if args.img_size != 224:
        enc.transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    out_dim = enc.out_dim
    print(f"Encoder: {args.backbone}, pretrained={bool(args.pretrained)}, out_dim={out_dim}, device={device}")

    feats = []
    has_flags = []

    # подготовим чёрную заглушку на нужный размер
    def make_stub():
        img = Image.new("RGB", (args.img_size, args.img_size), (0, 0, 0))
        return enc.transform(img) if enc.transform is not None else transforms.ToTensor()(img)

    with torch.no_grad():
        for i in trange(0, len(item_ids), args.batch_size, desc="Extracting"):
            chunk_ids = item_ids[i:i + args.batch_size]
            imgs = []
            for item_id in chunk_ids:
                p = resolve_image_path(args.images_root, int(item_id))
                if p is None:
                    imgs.append(make_stub())
                    has_flags.append(0.0)
                else:
                    img = Image.open(p).convert("RGB")
                    imgs.append(enc.transform(img))
                    has_flags.append(1.0)

            batch = torch.stack(imgs).to(device)  # [B,3,H,W]
            emb = enc(batch).detach().cpu().numpy()  # [B,D]
            feats.append(emb)

    feats = np.vstack(feats).astype(np.float32)
    has = np.asarray(has_flags, dtype=np.float32).reshape(-1, 1)

    # sanity checks
    assert feats.shape[0] == len(item_ids), "Количество эмбеддингов не совпало с количеством строк"
    if submit_ids is not None:
        assert len(submit_ids) == len(item_ids), "Длина submit_id не совпала с ItemID"

    np.savez_compressed(
        args.out,
        item_id=item_ids,
        id=submit_ids if submit_ids is not None else np.array([], dtype=np.int64),
        emb=feats,
        has_image=has,
    )

    miss = int((has == 0).sum())
    print(f"✅ saved: {args.out} | rows={len(item_ids)} | emb_dim={feats.shape[1]} | missing_images={miss}")


if __name__ == "__main__":
    main()