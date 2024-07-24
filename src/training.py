import torch
import torch.nn as nn
import mlflow

from torch.optim import Adam

from src.models.ingredients_classification_model import SkincareClassifier
from src.training_config import training_device
from src.data.dataset import create_dataloaders


def train_model(model, train_loader, val_loader, epochs, learning_rate, experiment_name):
    """Trains the model and logs metrics to MLflow."""
    print(f"Running on {training_device}")
    model.to(training_device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

    mlflow.set_experiment(experiment_name)  # Create or use existing experiment

    with mlflow.start_run() as run:
        # mlflow.log_params(model_args)

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
            print("train_loss", train_loss / len(train_loader.dataset), epoch)
            print("val_loss", val_loss / len(val_loader.dataset), epoch)
            print(f'val_accuracy = {val_accuracy:.4f}')
            print(f'train_accuracy = {train_accuracy:.4f}')
            mlflow.log_metric("train_loss", train_loss / len(train_loader.dataset), epoch)
            mlflow.log_metric("val_loss", val_loss / len(val_loader.dataset), epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, epoch)


if __name__ == "__main__":
    experiment_name = "cosmetic_efficacy_experiment"
    train_dataloader, val_dataloader = create_dataloaders()

    vocab_size = 5868
    embed_size = 128
    num_labels = 5  # Number of skin types

    # Transformer encoder parameters
    # model_args = {
    #     'd_model': 768,  # Embedding dimension
    #     'nhead': 2,  # Number of heads in multi-head attention
    #     'num_transformer_layers': 6,  # Number of transformer encoder layers
    #     'dim_feedforward': 1024,  # Dimension of feedforward network in the transformer
    #     'dropout': 0.1,  # Dropout probability
    # }
    #
    # ingredients_vector_len = 216

    model = SkincareClassifier(vocab_size, embed_size, num_labels)
    # model = CosmeticEfficacyModel(ingredients_vector_len, 5, [768 // 2, (768 // 2) // 2], **model_args)

    train_model(model, train_dataloader, val_dataloader, epochs=50, learning_rate=0.001,
                experiment_name=experiment_name)
