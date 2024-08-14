import torch
import torch.nn as nn

from src.models.classification_head import ClassificationHead


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, tokenized_vector_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(tokenized_vector_len, embed_size)
        self.encoding.requires_grad = False

        pos = torch.arange(0, tokenized_vector_len).unsqueeze(1).float()
        _2i = torch.arange(0, embed_size, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return x + self.encoding[:seq_len, :].to(x.device)


class SkincareClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels, max_len, embed_size, nhead, num_layers):
        super(SkincareClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead),
            num_layers=num_layers
        )
        self.classifier = ClassificationHead(embed_size, num_labels, hidden_sizes=[embed_size // 2], dropout=0.2)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = self.positional_encoding(embedded)

        transformer_output = self.transformer(embedded)
        pooled_output = transformer_output.mean(dim=1)  # Pooling
        logits = self.classifier(pooled_output)

        return logits
