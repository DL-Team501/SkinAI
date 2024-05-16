import os

import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import mlflow

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

from src.models.ingredients_classification_model import CosmeticEfficacyModel, preprocess_ingredients
from src.models.ingredients_tokenizer import CosmeticIngredientTokenizer
from src.training_config import training_device


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
    text = text.rstrip('.')

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
    df['clean_ingredients_lists'] = df['clean_ingredients'].str.split(',').map(lambda x: [item.strip() for item in x])
    unique_ingredients = set()

    for ingredient_list in df['clean_ingredients']:
        ingredients = ingredient_list.split(",")
        unique_ingredients.update(ingredient.strip() for ingredient in ingredients)

    unique_ingredients = list(unique_ingredients)
    unique_ingredients.append('<UNK>')
    ingredients_dict = {ingredient: idx for idx, ingredient in enumerate(unique_ingredients)}

    return ingredients_dict, df


# **Training Function**
def train_model(model, train_loader, val_loader, epochs, learning_rate, experiment_name):
    """Trains the model and logs metrics to MLflow."""
    print(training_device)
    model.to(training_device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    mlflow.set_experiment(experiment_name)  # Create or use existing experiment

    with mlflow.start_run() as run:
        # mlflow.log_params(model.transformer_encoder.get_config())  # Log model hyperparameters

        for epoch in range(epochs):
            print(f"Epoch number : {epoch}")
            model.train()
            train_loss = 0.0
            train_total = 0
            train_correct = 0

            for batch in train_loader:
                print('1')
                ingredients, labels = batch
                ingredients, labels = ingredients.to(training_device), labels.to(training_device)
                outputs = model(ingredients)

                predicted = (outputs >= 0.5).float()  # Apply a threshold for multi-label

                train_total += labels.numel()  # Total number of labels (elements)
                train_correct += (predicted == labels).sum().item()

                labels = labels.float()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients

                train_loss += loss.item() * ingredients.size(0)
                print('2')

            model.eval()  # Set model to evaluation mode
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():  # Disable gradient calculation for efficiency
                for batch in val_loader:
                    ingredients, labels = batch
                    ingredients, labels = ingredients.to(training_device), labels.to(training_device)
                    outputs = model(ingredients)

                    predicted = (outputs >= 0.5).float()  # Apply a threshold for multi-label

                    val_total += labels.numel()  # Total number of labels (elements)
                    val_correct += (predicted == labels).sum().item()

                    labels = labels.float()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * ingredients.size(0)

            val_accuracy = val_correct / val_total
            train_accuracy = train_correct / train_total

            # Log metrics to MLflow
            print("train_loss", train_loss / len(train_loader.dataset), epoch)
            print("val_loss", val_loss / len(val_loader.dataset), epoch)
            print(f'val_accuracy = {val_accuracy:.4f}')
            print(f'train_accuracy = {train_accuracy:.4f}')
            mlflow.log_metric("train_loss", train_loss / len(train_loader.dataset), epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader.dataset), epoch)
            # ... log other relevant metrics


# **Main Execution**
if __name__ == "__main__":
    # Transformer encoder parameters
    model_args = {
        'd_model': 768,  # Embedding dimension
        'nhead': 8,  # Number of heads in multi-head attention
        'num_transformer_layers': 6,  # Number of transformer encoder layers
        'dim_feedforward': 1024,  # Dimension of feedforward network in the transformer
        'dropout': 0.1,  # Dropout probability
    }
    experiment_name = "cosmetic_efficacy_experiment"
    ingredients_dict, dataset_df = load_data()
    max_length = max([len(ingredient_list.split(",")) for ingredient_list in list(dataset_df['clean_ingredients'])])

    skin_type_cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']

    # Convert each selected column into a NumPy array
    arrays_list = [np.array(dataset_df[col]) for col in skin_type_cols]

    # Vertically stack the arrays
    labels = np.vstack(arrays_list).T

    # Initialize your tokenizer using your ingredient dictionary
    tokenizer = CosmeticIngredientTokenizer(ingredients_dict)
    model = CosmeticEfficacyModel(ingredients_dict, 5, max_length, **model_args)
    dataset = CosmeticIngredientsDataset(list(dataset_df['clean_ingredients_lists']), labels, ingredients_dict)
    train_dataloader, val_dataloader = get_dataloaders(dataset, train_split=0.8)

    train_model(model, train_dataloader, val_dataloader, epochs=10, learning_rate=0.001,
                experiment_name=experiment_name)
