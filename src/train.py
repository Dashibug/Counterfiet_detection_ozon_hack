import yaml
import random
import numpy as np
import pandas as pd
import torch
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

    numeric_cols = [c for c in df.columns if c.startswith("num_")]  # TODO: вынеси в конфиг
    meta_dim = len(numeric_cols) + 1  # + has_image

    img_tf = transforms.Compose([
        transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = MultimodalDataset(
        df=df,
        images_root=cfg["data"]["images_root"],
        tokenizer=tokenizer,
        numeric_cols=numeric_cols,
        image_transform=img_tf,
        text_max_len=cfg["model"]["text_max_len"],
        add_has_image=True,
    )
    loader = DataLoader(dataset,
                        batch_size=cfg["train"]["batch_size"],
                        shuffle=True,
                        num_workers=cfg["data"]["num_workers"],
                        drop_last=True)

    # --- MODELS ---
    image_extractor = ImageEncoder(
        backbone="resnet50",
        pretrained=True,
        device=device
    )
    text_encoder = TextEncoder(
        model_name=cfg["model"]["text_model"],
        device=device
    )
    clf = MultimodalClassifier(
        image_dim=image_extractor.out_dim,
        text_dim=768,
        meta_dim=meta_dim,
        hidden_size=cfg["model"]["fusion_hidden"]
    ).to(device)

    # --- OPTIM / LOSS ---
    optimizer = torch.optim.AdamW(clf.parameters(),
                                  lr=float(cfg["train"]["lr"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- TRAIN LOOP ---
    best_auc = 0.0

    try:
        for epoch in range(cfg["train"]["epochs"]):
            clf.train()
            all_preds, all_labels = [], []
            running_loss = 0.0

            for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")):
                images = batch["image"].to(device)
                meta = batch["meta"].to(device)
                labels = batch["label"].to(device)

                # эмбеддинги (замороженные энкодеры)
                with torch.no_grad():
                    image_emb = image_extractor(images)
                    toks = {k: v.to(device) for k, v in batch["text"].items()}
                    out = text_encoder.model(**toks, return_dict=True)
                    last_hidden = out.last_hidden_state
                    attn = toks["attention_mask"].unsqueeze(-1).float()
                    text_emb = (last_hidden * attn).sum(1) / attn.sum(1)

                # шаг оптимизации
                logits = clf(image_emb, text_emb, meta)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # учёт лосса и метрик ПОСЛЕ каждого шага
                running_loss += loss.item()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.detach().cpu().numpy())

                # периодический лог
                if (step + 1) % 50 == 0:
                    avg = running_loss / 50.0
                    print(f"[epoch {epoch + 1} step {step + 1}] loss={avg:.4f}", flush=True)
                    running_loss = 0.0

            # метрики за эпоху
            metrics = compute_metrics(all_labels, all_preds)
            print(f"Epoch {epoch + 1}/{cfg['train']['epochs']} "
                  f"Loss(last)={loss.item():.4f} AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f}", flush=True)

            # сохранение лучшего по AUC
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                torch.save({
                    "clf_state": clf.state_dict(),
                    "best_auc": best_auc,
                    "epoch": epoch + 1,
                }, "ckpt_best.pt")
                print(f"✔ Saved best checkpoint: AUC={best_auc:.4f}", flush=True)

    except KeyboardInterrupt:
        print("\n⛔️ Interrupted by user. Saving current checkpoint as ckpt_interrupt.pt")
        torch.save({"clf_state": clf.state_dict()}, "ckpt_interrupt.pt")


if __name__ == "__main__":
    main()
