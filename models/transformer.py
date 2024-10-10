"""
Transformer encoder based model architecture.
"""
import torch.nn as nn

class TransformerRegressor(nn.Module): # tf_1
    def __init__(self, input_size, hidden_size=256, num_layers=4, num_heads=8, output_size=1, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.linear = nn.Linear(1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(hidden_size * input_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size * 4)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.linear(x)  # Add sequence dimension
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x