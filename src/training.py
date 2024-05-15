import os

import pandas as pd
import re
import torch
import torch.nn as nn
import mlflow

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from src.models.ingredients_classification_model import CosmeticEfficacyModel, preprocess_ingredients
from src.models.ingredients_tokenizer import CosmeticIngredientTokenizer


class CosmeticIngredientsDataset(Dataset):
    def __init__(self, ingredient_lists, labels, ingredient_dict):
        self.ingredient_lists = ingredient_lists
        self.labels = labels
        self.tokenizer = CosmeticIngredientTokenizer(ingredient_dict)  # Assuming you have this

    def __len__(self):
        return len(self.ingredient_lists)

    def __getitem__(self, index):
        ingredients = self.ingredient_lists[index]
        label = self.labels[index]

        token_ids = preprocess_ingredients(ingredients, self.tokenizer, max_length)
        return token_ids, label


# *2. Data Loading Functions*
def get_dataloaders(dataset, batch_size=32, train_split=0.8):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


def clean_ingredients(text):
    """
  Cleans the ingredient text.

  Args:
      text: String containing the ingredient list.

  Returns:
      A string with cleaned ingredients in lowercase, with extra spaces removed.
  """
    # Lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cosmetic.csv')
    df = pd.read_csv(data_dir)
    skin_type_cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    df_skin_types = df[skin_type_cols]
    all_zero_skin_types = (df_skin_types == 0).all(axis=1).sum()
    df.loc[(df_skin_types == 0).all(axis=1), 'Normal'] = 1
    # Apply the cleaning function to the 'ingredients list' column
    df['clean_ingredients'] = df['ingredients'].apply(clean_ingredients)
    df['clean_ingredients_lists'] = df['clean_ingredients'].str.split(',')
    unique_ingredients = set()

    for ingredient_list in df['clean_ingredients']:
        ingredients = ingredient_list.split(",")
        unique_ingredients.update(ingredient.strip() for ingredient in ingredients)

    ingredients_dict = {ingredient: idx for idx, ingredient in enumerate(unique_ingredients)}

    return ingredients_dict, df


# **Training Function**
def train_model(model, train_loader, val_loader, epochs, learning_rate, experiment_name):
    """Trains the model and logs metrics to MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    mlflow.set_experiment(experiment_name)  # Create or use existing experiment

    with mlflow.start_run() as run:
        mlflow.log_params(model.transformer_encoder.get_config())  # Log model hyperparameters

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                ingredients, labels = batch
                ingredients, labels = ingredients.to(device), labels.to(device)
                outputs = model(ingredients)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients

                train_loss += loss.item() * ingredients.size(0)

            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation for efficiency
                for batch in val_loader:
                    ingredients, labels = batch
                    ingredients, labels = ingredients.to(device), labels.to(device)
                    outputs = model(ingredients)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * ingredients.size(0)

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss / len(train_loader.dataset), epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader.dataset), epoch)
            # ... log other relevant metrics


# **Main Execution**
if __name__ == "__main__":
    # Transformer encoder parameters
    model_args = {
        'd_model': 256,  # Embedding dimension
        'nhead': 8,  # Number of heads in multi-head attention
        'num_transformer_layers': 6,  # Number of transformer encoder layers
        'dim_feedforward': 1024,  # Dimension of feedforward network in the transformer
        'dropout': 0.1,  # Dropout probability
    }
    experiment_name = "cosmetic_efficacy_experiment"
    ingredients_dict, dataset_df = load_data()
    max_length = max([len(ingredient_list.split(",")) for ingredient_list in list(dataset_df['clean_ingredients'])])

    # Initialize your tokenizer using your ingredient dictionary
    tokenizer = CosmeticIngredientTokenizer(ingredients_dict)

    model = CosmeticEfficacyModel(ingredients_dict, 5, max_length, **model_args)
    dataset = CosmeticIngredientsDataset(dataset_df['clean_ingredients_lists'], dataset_df['skiny_type_label'],
                                         ingredients_dict)
    train_dataloader, val_dataloader = get_dataloaders(dataset, train_split=0.8)

    train_model(model, train_dataloader, val_dataloader, epochs=10, learning_rate=0.001,
                experiment_name=experiment_name)
