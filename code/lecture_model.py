import torch
from torch import nn


class LoopedVerifier(nn.Module):
    def __init__(self, dim, width=128):
        super().__init__()
        self.projection = nn.Linear(dim, width)
        self.block = nn.TransformerEncoderLayer(
            width,
            4,
            width * 2,
            0.3,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        self.skip = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        e = self.projection(x)
        h = e
        for _ in range(3):
            h = self.block(self.norm(h + 0.25 * e))
        return (self.head(self.norm(h)) + self.skip(x)).squeeze(-1)
