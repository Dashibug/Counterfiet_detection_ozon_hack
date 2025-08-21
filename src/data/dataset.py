"""
    Мультимодальный датасет
"""
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class MultimodalDataset(Dataset):
    """
    df - исходная таблица
    image_root - пути до картинок
    tokenizer - токенизатор
    numeric_cols - числовые столбцы
    image_transform - преобразование изображений(один размер,аугментации)
    text_max_len - макс длина текста
    add_has_image - новая фича: некоторые товары без картинок, я добавил флаг, мб его ваще не использовать
    """
    def __init__(self, df, images_root, tokenizer, numeric_cols,
                 image_transform = None, text_max_len = 128, label_col="resolution", add_has_image=False):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.numeric_cols = numeric_cols
        self.image_transform = image_transform
        self.text_max_len = text_max_len
        self.label_col = label_col
        self.add_has_image = add_has_image

    def __len__(self):
        return len(self.df)

    #так как некоторых картинок нет
    def _resolve_image_path(self, item_id):
        base = os.path.join(self.images_root, str(item_id))
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".JPG", ".PNG"):
            p = base + ext
            if os.path.exists(p):
                return p
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- IMAGE ---
        # img_path = os.path.join(self.images_root, f"{row['ItemID']}.png")
        # image = Image.open(img_path).convert("RGB")
        # if self.image_transform:
        #     image = self.image_transform(image)
        # --- IDs ---
        if "id" in self.df.columns:
            id_submit = int(row["id"])
        else:
            id_submit = int(row.name)

        item_id = int(row["ItemID"])

        img_path = self._resolve_image_path(item_id)
        has_image = 1.0 if img_path is not None else 0.0
        if img_path is not None:
            image = Image.open(img_path).convert("RGB")
            if self.image_transform:
                image = self.image_transform(image)
        else:
            # заглушка того же размера, что обычное изображение
            from torchvision.transforms import ToTensor, Resize
            _tf = transforms.Compose([
                Resize((224, 224)),
                ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = _tf(Image.new("RGB", (224, 224), (0, 0, 0)))

        # --- TEXT ---
        text_parts = [
            str(row.get('name_rus', '')),
            str(row.get('description', '')),
            str(row.get('brand_name', '')),
            str(row.get('CommercialTypeName4', ''))# безопасно вытаскиваем название
        ]
        text = " ".join([part for part in text_parts if part != 'nan'])
        toks = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.text_max_len,
            return_tensors="pt"
        )
        toks = {k: v.squeeze(0) for k, v in toks.items()}


        # --- NUMERIC META ---
        meta = torch.tensor(row[self.numeric_cols].values.astype(float), dtype = torch.float32)
        if self.add_has_image:
            meta = torch.cat([meta, torch.tensor([has_image], dtype=torch.float32)], dim=0)

        # --- LABEL ---
        #label = torch.tensor(row["resolution"], dtype=torch.float)
        out = {
            "id": id_submit,
            "item_id": item_id,
            "image": image,
            "text": toks,  # dict тензоров [L], без лишних unsqueeze
            "meta": meta
        }
        if self.label_col is not None and self.label_col in self.df.columns:
            out["label"] = torch.tensor(row[self.label_col], dtype=torch.float32)
        return out

        # return {
        #     "id": int(row["ItemID"]), # закомментируй для трейна
        #     "image": image,
        #     "text": toks,
        #     "meta": meta,
        #     #"label": label # закомментируй для трейна
        # }



