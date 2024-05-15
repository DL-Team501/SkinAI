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

    def forward(self, ingredient_lists):
        token_ids_list = [preprocess_ingredients(ingredient_list, self.tokenizer, self.max_length)
                          for ingredient_list in ingredient_lists]
        batch_input_ids = torch.stack(token_ids_list, dim=0)  # Stack the token IDs along batch dimension

        encoded_ingredients = self.transformer_encoder(batch_input_ids)
        # Use encoded_ingredients for classification (e.g., take mean or max-pool)
        output = self.classification_head(encoded_ingredients[:, 0, :])  # Example: using the first token

        return output
