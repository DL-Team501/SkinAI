import json

import torch
import torch.nn as nn
import mlflow

from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.ingredients_classification_model import SkincareClassifier
from src.training_config import training_device
from src.data.dataset import create_dataloaders
from src.consts import INGREDIENT_LIST_CLASSIFICATION_LABELS


def train_model(model, train_loader, val_loader, epochs, learning_rate, experiment_name, model_args):
    """Trains the model and logs metrics to MLflow."""
    print(f"Running on {training_device}")
    model.to(training_device)

    # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)  # Create or use existing experiment

    with mlflow.start_run():
        mlflow.log_params(model_args)

        for epoch in range(epochs):
            print(f"Epoch number : {epoch}")
            model.train()
            train_loss = 0.0
            train_total = 0
            train_correct = 0

            for batch in train_loader:
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
            print("train_loss", train_loss / len(train_loader.dataset))
            print("val_loss", val_loss / len(val_loader.dataset))
            print(f'val_accuracy = {val_accuracy:.4f}')
            print(f'train_accuracy = {train_accuracy:.4f}')
            mlflow.log_metric("train_loss", train_loss / len(train_loader.dataset), epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader.dataset), epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, epoch)

            # scheduler.step(val_loss / len(val_loader.dataset))

    scripted_model = torch.jit.script(model)
    scripted_model.save('ingredients_classifier.pt')


if __name__ == "__main__":
    # Run in terminal: mlflow server --host 127.0.0.1 --port 8080
    experiment_name = "old_dataset"
    train_dataloader, val_dataloader, skin_care_data = create_dataloaders()

    vocab_size = len(skin_care_data.ingredient_index_dict)
    ingredients_vector_len = skin_care_data.data['tokenized_ingredients'].apply(len).max()
    num_labels = len(INGREDIENT_LIST_CLASSIFICATION_LABELS)  # Number of skin types

    with open("ingredient_index_dict.json", "w") as outfile:
        outfile.write(json.dumps(skin_care_data.ingredient_index_dict))

    with open("ingredients_vector_len.json", "w") as outfile:
        outfile.write(json.dumps({"ingredients_vector_len": int(ingredients_vector_len)}))

    # Transformer encoder parameters
    model_args = {
        'embed_size': 128,  # Embedding dimension
        'nhead': 2,  # Number of heads in multi-head attention
        'num_layers': 2,  # Number of transformer encoder layers
    }

    model = SkincareClassifier(vocab_size, num_labels, ingredients_vector_len, **model_args)

    train_model(model, train_dataloader, val_dataloader, epochs=30, learning_rate=0.001,
                experiment_name=experiment_name, model_args=model_args)
