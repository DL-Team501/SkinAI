import torch.nn as nn

from src.models.transformer import TransformerEncoderModel
from src.models.classification_head import ClassificationHead
from src.training_config import training_device


class CosmeticEfficacyModel(nn.Module):
    def __init__(self, ingredients_vector_size, num_classes, **kwargs):
        super().__init__()
        self.ingredients_vector_size = ingredients_vector_size
        self.transformer_encoder = TransformerEncoderModel(ingredients_vector_size, **kwargs).to(training_device)
        self.classification_head = ClassificationHead(kwargs['d_model'], num_classes).to(training_device)

    def forward(self, ingredient_lists):
        batch_input_ids = ingredient_lists
        encoded_ingredients = self.transformer_encoder(batch_input_ids)
        # Use encoded_ingredients for classification (e.g., take mean or max-pool)
        output = self.classification_head(encoded_ingredients[:, 0, :])  # Example: using the first token

        return output
