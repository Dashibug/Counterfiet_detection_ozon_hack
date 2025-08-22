"""
ImageEncoder — модуль, который даёт image-embedding.

"""

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import open_clip
from typing import Optional

class ImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        normalize: bool = True,
        dtype: Optional[torch.dtype] = None,  # можно передать torch.float16 для ускорения
    ):
        """
        Args:
            model_name: имя CLIP-архитектуры (напр. "ViT-B-32", "ViT-L-14", "ViT-B-16-plus-240")
            pretrained: имя чекпойнта open-clip (напр. "laion2b_s34b_b79k", "openai", "datacomp_xl_s13b_b90k")
            normalize: L2-нормализовать ли выходные эмбеддинги
            dtype: принудительный dtype (напр. torch.float16). По умолчанию — как у модели.
        """
        super().__init__()
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.normalize = normalize

        # Модель + валид-трансформы от open-clip
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model = model.eval()  # encode_image внутри модели
        self.transform = preprocess_val  # корректный валид-трансформ
        # Размер эмбеддинга (у ViT-B/32 = 512, у L/14 = 768 и т.д.)
        # open-clip хранит embed_dim на корневой модели, а у visual — output_dim
        self.out_dim = int(getattr(self.model, "embed_dim", getattr(self.model.visual, "output_dim")))

        # фиксируем dtype, чтобы forward мог приводить входы
        self.dtype = next(self.model.parameters()).dtype  # dtype модели после кастов
        # если явно передан dtype — всё равно он совпадёт с параметрами

        if dtype is not None:
            self.model.to(dtype=dtype)

        # выключаем все градиенты
        for p in self.model.parameters():
            p.requires_grad_(False) # размер эмбеддинга

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
                images: тензор [B, 3, H, W], уже прогнанный через self.transform
                return: эмбеддинги [B, out_dim] (L2-нормированные, если normalize=True)
        """
        images = images.to(self.device)
        if images.dtype != self.dtype:
            images = images.to(self.dtype)
        feats = self.model.encode_image(images.to(self.device))
        if self.normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats
