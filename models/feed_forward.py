"""
Feedforward neural network model.
"""
import torch.nn as nn

class FF_1Network(nn.Module): # ff_1
    def __init__(self, input_size, hidden_size, output_size):
        super(FF_1Network, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class FF_2Network(nn.Module): # ff_2
    def __init__(self, input_size, hidden_size, output_size):
        super(FF_2Network, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.layer_norm3 = nn.LayerNorm(hidden_size * 4)
        self.layer4 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.layer_norm4 = nn.LayerNorm(hidden_size * 2)
        self.layer5 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm5 = nn.LayerNorm(hidden_size)
        self.layer6 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First block
        x = self.layer1(x)
        x = self.layer_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Second block
        x = self.layer2(x)
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Third block
        x = self.layer3(x)
        x = self.layer_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fourth block
        x = self.layer4(x)
        x = self.layer_norm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fifth block
        x = self.layer5(x)
        x = self.layer_norm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.layer6(x)
        return x