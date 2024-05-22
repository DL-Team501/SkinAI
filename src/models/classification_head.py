import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()  # Add ReLU activation
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # Apply ReLU
        x = self.fc2(x)
        return self.sigmoid(x)
