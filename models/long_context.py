import torch
import torch.nn as nn


class LongContextTransformer(nn.Module):
    def __init__(self, input_size, context_length, hidden_size=256, num_layers=4, num_heads=8, output_size=150):
        super().__init__()
        self.context_length = context_length
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, context_length + 1, hidden_size))
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size))

    def forward(self, tokens):
        if tokens.shape[1] != self.context_length + 1:
            raise ValueError(f"expected {self.context_length + 1} tokens, got {tokens.shape[1]}")
        hidden = self.input_projection(tokens) + self.position
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1], device=tokens.device)
        return self.output(self.encoder(hidden, mask=causal_mask)[:, -1])
