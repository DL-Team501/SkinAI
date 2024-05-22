import torch
import torch.nn as nn
import math

from src.training_config import training_device


class TransformerEncoderModel(nn.Module):
    def __init__(self, max_length, num_transformer_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()

        # Transformer Encoder Layer (unchanged)
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout).to(training_device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers).to(training_device)

    def forward(self, ingredient_tokens):
        # Assuming ingredient_tokens are already integer-encoded

        # Get positional embeddings using sin/cos encoding
        batch_size, sequence_length = ingredient_tokens.size()
        pos_embeddings = self.get_positional_encoding(sequence_length, self.d_model).to(ingredient_tokens.device)
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Type cast ingredient_tokens to float for compatibility with positional embeddings
        ingredient_tokens = ingredient_tokens.float()

        # Expand ingredient_tokens to have d_model dimensions
        ingredient_tokens = ingredient_tokens.unsqueeze(-1).expand(-1, -1, self.d_model)

        # Add positional embeddings directly to integer-encoded ingredient tokens
        embedded_ingredients = ingredient_tokens + pos_embeddings

        # Apply transformer encoder layer
        output = self.transformer_encoder(embedded_ingredients)
        return output

    def get_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
