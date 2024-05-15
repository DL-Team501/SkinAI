import torch
import torch.nn as nn

from src.models.ingredients_tokenizer import CosmeticIngredientTokenizer
from src.models.transformer import TransformerEncoderModel
from src.models.classification_head import ClassificationHead


def preprocess_ingredients(ingredient_list, tokenizer, max_length):
    token_ids = tokenizer.tokenize(ingredient_list)
    token_ids = token_ids[:max_length]  # Truncate if necessary
    padding_length = max_length - len(token_ids)
    token_ids += [0] * padding_length  # Padding with zeros

    return torch.tensor(token_ids)


class CosmeticEfficacyModel(nn.Module):
    def __init__(self, ingredient_dict, num_classes, max_length, **kwargs):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = CosmeticIngredientTokenizer(ingredient_dict)  # Load your ingredient dictionary
        self.transformer_encoder = TransformerEncoderModel(max_length, **kwargs)
        self.classification_head = ClassificationHead(kwargs['d_model'], num_classes)

    def forward(self, ingredient_list):
        ingredient_tokens = preprocess_ingredients(ingredient_list, self.tokenizer, self.max_length)
        encoded_ingredients = self.transformer_encoder(ingredient_tokens)
        # Use encoded_ingredients for classification (e.g., take mean or max-pool)
        output = self.classification_head(encoded_ingredients[:, 0, :])  # Example: using the first token
        return output
