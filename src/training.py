import pandas as pd
import re
import torch
import torch.nn as nn
import mlflow
from mlflow.tracking import MlflowClient
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from src.models.ingredients_classification_model import CosmeticEfficacyModel
from src.models.ingredients_tokenizer import CosmeticIngredientTokenizer


class CosmeticIngredientsDataset(Dataset):
    def _init_(self, ingredient_lists, labels, ingredient_dict):
        self.ingredient_lists = ingredient_lists
        self.labels = labels
        self.tokenizer = CosmeticIngredientTokenizer(ingredient_dict)  # Assuming you have this

    def _len_(self):
        return len(self.ingredient_lists)

    def _getitem_(self, index):
        ingredients = self.ingredient_lists[index]
        label = self.labels[index]

        token_ids = preprocess_ingredients(ingredients, self.tokenizer, max_length)
        return token_ids, label


# *2. Data Loading Functions*
def get_train_dataloader(dataset, batch_size=32, train_split=0.8):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def get_val_dataloader(dataset, batch_size=32):
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return val_dataloader


# *Example Usage (Assuming you've loaded your data)*
all_ingredient_lists = ...  # Load your list of ingredient lists
all_labels = ...  # Load corresponding labels
dataset = CosmeticIngredientsDataset(all_ingredient_lists, all_labels)
train_dataloader = get_train_dataloader(dataset)
val_dataloader = get_val_dataloader(dataset)


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
    df = pd.read_csv('../data/cosmetic.csv')
    skin_type_cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    df_skin_types = df[skin_type_cols]
    all_zero_skin_types = (df_skin_types == 0).all(axis=1).sum()
    df.loc[(df_skin_types == 0).all(axis=1), 'Normal'] = 1
    # Apply the cleaning function to the 'ingredients list' column
    df['clean_ingredients'] = df['ingredients'].apply(clean_ingredients)
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

            # ... (Add validation loop similar to the above)

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

        # Other parameters
        'num_classes': 5  # Number of output classes (for your efficacy labels)
    }
    experiment_name = "cosmetic_efficacy_experiment"
    ingredients_dict, df = load_data()
    max_length = max([len(ingredient_list.split(",")) for ingredient_list in list(df['clean_ingredients'])])

    # Initialize your tokenizer using your ingredient dictionary
    tokenizer = CosmeticIngredientTokenizer(ingredients_dict)

    model = CosmeticEfficacyModel(**model_args)
    train_dataloader = get_train_dataloader()
    val_dataloader = get_val_dataloader()

    train_model(model, train_dataloader, val_dataloader, epochs=10, learning_rate=0.001,
                experiment_name=experiment_name)
