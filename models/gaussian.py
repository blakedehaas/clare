"""
Gaussian neural network model.
2 output logits to represent mean and variance.
"""
import torch.nn as nn

class GaussianNetwork(nn.Module): # gaussian_1
    def __init__(self, input_size, hidden_size, output_size):
        super(GaussianNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, output_size)
        self.var_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        mean = self.mean_layer(x)
        var = self.relu(self.var_layer(x))
        return mean, var
