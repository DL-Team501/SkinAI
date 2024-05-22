import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=None, dropout=0.0):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [input_size // 2]  # Default to one hidden layer

        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        if dropout > 0.0:
            self.layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())
            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.sigmoid(x)
