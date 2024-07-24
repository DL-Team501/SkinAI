import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len).unsqueeze(1).float()
        _2i = torch.arange(0, embed_size, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return x + self.encoding[:seq_len, :].to(x.device)


class SkincareClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_labels, max_len=216):
        super(SkincareClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=8),
            num_layers=3
        )
        self.classifier = nn.Linear(embed_size, num_labels)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        embedded = self.positional_encoding(embedded)
        transformer_output = self.transformer(embedded)
        pooled_output = transformer_output.mean(dim=1)  # Pooling
        logits = self.classifier(pooled_output)
        return logits
