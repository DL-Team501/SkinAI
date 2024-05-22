import torch.nn as nn
from src.training_config import training_device


class TransformerEncoderModel(nn.Module):
    def __init__(self, num_transformer_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # Transformer Encoder Layer (unchanged)
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout).to(training_device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers).to(training_device)

    def forward(self, ingredient_tokens):
        # Assuming ingredient_tokens are already integer-encoded

        # Get positional embeddings using sin/cos encoding
        batch_size, sequence_length = ingredient_tokens.size()

        # Type cast ingredient_tokens to float for compatibility with positional embeddings
        ingredient_tokens = ingredient_tokens.float()

        # Expand ingredient_tokens to have d_model dimensions
        ingredient_tokens = ingredient_tokens.unsqueeze(-1).expand(-1, -1, self.d_model)
        # Apply transformer encoder layer
        output = self.transformer_encoder(ingredient_tokens)

        return output
