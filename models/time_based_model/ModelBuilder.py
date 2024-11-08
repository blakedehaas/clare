# ModelBuilder.py

import torch
import torch.nn as nn

class ModelBuilder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        """
        Initializes the ModelBuilder.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in LSTM.
            output_size (int): Number of output features.
            dropout (float): Dropout rate.
        """
        super(ModelBuilder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        # Use the last time step's output
        last_time_step = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(last_time_step)  # (batch_size, output_size)
        return out
