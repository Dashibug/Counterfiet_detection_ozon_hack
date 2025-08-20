import torch
import torch.nn as nn

class MultimodalClassifier(nn.Module):
    """
    Мультимодальная модель:
      - image_emb [B, image_dim]
      - text_emb [B, text_dim]
      - meta_features [B, meta_dim]
    Конкатенируем → MLP → вероятность подделки (0/1).
    """

    def __init__(self, image_dim: int, text_dim: int, meta_dim: int = 0,
                 hidden_size: int = 512, dropout: float = 0.3):
        super().__init__()

        # Если есть табличные данные — отдельный "проектор"
        if meta_dim > 0:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            meta_out_dim = hidden_size // 2
        else:
            self.meta_proj = None
            meta_out_dim = 0

        # Итоговый размер после конкатенации
        in_dim = image_dim + text_dim + meta_out_dim

        # Основной классификатор
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # бинарная классификация
        )

    def forward(self, image_emb, text_emb, meta_features=None):
        """
        image_emb: [B, image_dim]
        text_emb: [B, text_dim]
        meta_features: [B, meta_dim] (опционально)
        """
        if meta_features is not None and self.meta_proj is not None:
            meta_emb = self.meta_proj(meta_features)
            x = torch.cat([image_emb, text_emb, meta_emb], dim=1)
        else:
            x = torch.cat([image_emb, text_emb], dim=1)

        logits = self.classifier(x)  # [B, 1]
        return logits.squeeze(1)  # [B]
