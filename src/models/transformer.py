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
        # Obtain the output of the BERT model
        bert_output = self.bert(ingredient_tokens)[0]  # [batch_size, sequence_length, hidden_size]

        # Obtain positional embeddings
        batch_size, sequence_length, hidden_size = bert_output.size()
        pos_embeddings = self.positional_embedding(torch.arange(sequence_length).to(ingredient_tokens.device))
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Expand to match batch size

        # Add positional embeddings to BERT output
        embedded_ingredients = bert_output + pos_embeddings

        # Apply transformer encoder
        output = self.transformer_encoder(embedded_ingredients)

        return output
