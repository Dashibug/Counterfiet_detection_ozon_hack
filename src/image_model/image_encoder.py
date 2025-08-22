"""
ImageEncoder — модуль, который даёт image-embedding.

"""

import torch
import torch.nn as nn
from torchvision import transforms
import timm

class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, device='cuda'):
        super().__init__()
        # backbone без финального классификатора
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0  # выдаёт эмбеддинги вместо logits
        ).to(device)
        self.model.eval()
        self.device = device

        # Регистрируем hook на предпоследний слой (neck)
        #self.handle = self.model.model[-2].register_forward_hook(self._hook_fn)

        # Трансформ для подготовки изображения (как в resnet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.out_dim = self.model.num_features  # размер эмбеддинга

    def forward(self, images: torch.Tensor):
        """
        images: [B, 3, H, W], нормализованные изображения
        → возвращает [B, out_dim]
        """
        with torch.no_grad():
            return self.model(images)
