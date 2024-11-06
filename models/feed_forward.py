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

class FF_3Network(nn.Module): # ff_3
    def __init__(self, input_size, hidden_size, output_size):
        super(FF_3Network, self).__init__()
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
        x = self.relu(x)
        x = self.dropout(x)

        # Second block
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Third block
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fourth block
        x = self.layer4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fifth block
        x = self.layer5(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.layer6(x)
        return x


class FF_3Network(nn.Module): # ff_3
    def __init__(self, input_size, hidden_size, output_size):
        super(FF_3Network, self).__init__()
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
        x = self.relu(x)
        x = self.dropout(x)

        # Second block
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Third block
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fourth block
        x = self.layer4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Fifth block
        x = self.layer5(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer
        x = self.layer6(x)
        return x

class FF_4Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FF_4Network, self).__init__()

        # Layer 1
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        # Layer 2
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)

        # Layer 3
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.layer_norm3 = nn.LayerNorm(hidden_size * 2)

        # Layer 4
        self.layer4 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.layer_norm4 = nn.LayerNorm(hidden_size * 2)

        # Layer 5
        self.layer5 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.layer_norm5 = nn.LayerNorm(hidden_size * 2)

        # Layer 6
        self.layer6 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.layer_norm6 = nn.LayerNorm(hidden_size * 2)

        # Layer 7
        self.layer7 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm7 = nn.LayerNorm(hidden_size)

        # Layer 8 (Output layer)
        self.layer8 = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_norm1(self.layer1(x)))
        x = self.dropout(x)

        x = self.relu(self.layer_norm2(self.layer2(x)))
        x = self.dropout(x)

        x = self.relu(self.layer_norm3(self.layer3(x)))
        x = self.dropout(x)

        x = self.relu(self.layer_norm4(self.layer4(x)))
        x = self.dropout(x)

        x = self.relu(self.layer_norm5(self.layer5(x)))
        x = self.dropout(x)

        x = self.relu(self.layer_norm6(self.layer6(x)))
        x = self.dropout(x)

        x = self.relu(self.layer_norm7(self.layer7(x)))
        x = self.dropout(x)

        x = self.layer8(x)
        return x


class FF_5Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(FF_5Network, self).__init__()

        # Architecture with increasing then decreasing sizes
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)

        self.layer3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)

        self.layer4 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.layer5 = nn.Linear(hidden_size, output_size)

        # Different dropout rates for different depths
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 1.5)
        self.dropout3 = nn.Dropout(dropout_rate * 2)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU()

        # Optional: Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # First block
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Second block
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Third block (with residual)
        identity3 = x
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = x + identity3  # Same dimensions, safe to add

        # Fourth block
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        # Output layer (no activation/normalization/dropout)
        x = self.layer5(x)
        return x