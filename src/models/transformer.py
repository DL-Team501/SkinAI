import torch
import torch.nn as nn
from transformers import BertModel


class TransformerEncoderModel(nn.Module):
    def __init__(self, max_length, num_transformer_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # Or another suitable BERT model
        self.positional_embedding = nn.Embedding(max_length, d_model)  # For positional information

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

    def forward(self, ingredient_tokens):
        pos_embeddings = self.positional_embedding(torch.arange(ingredient_tokens.size(1)))
        embedded_ingredients = self.bert(ingredient_tokens)[0] + pos_embeddings
        output = self.transformer_encoder(embedded_ingredients)

        return output
