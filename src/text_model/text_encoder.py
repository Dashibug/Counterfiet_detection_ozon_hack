import torch
from transformers import AutoTokenizer, AutoModel
"""
    Обертка над HuggingFace encoder.
    Возвращает эмбеддинги текста фиксированной размерности.
"""

class TextEncoder(torch.nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased", max_length=128, device="cuda"):
        super().__init__()
        self.device = device
        self.max_length = max_length

        # Загружаем токенайзер и модель
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()  # encoder не обучаем по умолчанию (можно включить fine-tuning)

        # Сохраняем размер эмбеддинга
        self.hidden_size = self.model.config.hidden_size

    def forward(self, texts: list[str]):
        """
        texts: список строк (batch).
        Возвращает тензор [B, hidden_size].
        """
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():  # если не хотим fine-tune
            outputs = self.model(**toks)

        # Попробуем mean pooling (обычно лучше, чем CLS/pooler)
        last_hidden = outputs.last_hidden_state  # [B, seq_len, hidden]
        attention_mask = toks["attention_mask"]  # [B, seq_len]

        # Маскированное среднее
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = (last_hidden * mask_expanded).sum(1)
        lengths = mask_expanded.sum(1)
        embeddings = sum_hidden / lengths  # [B, hidden]

        return embeddings