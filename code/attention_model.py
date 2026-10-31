import numpy as np
import torch
from torch import nn
from baseline_submission import features
from math_groups import arithmetic


class SetVerifier(nn.Module):
    def __init__(self, dim, width=96):
        super().__init__()
        self.projection = nn.Linear(dim, width)
        layer = nn.TransformerEncoderLayer(
            width,
            4,
            2 * width,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(width, 1)
        self.skip = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.skip.weight)

    def forward(self, x):
        return (self.head(self.blocks(self.projection(x))) + self.skip(x)).squeeze(-1)


def inputs(bank, arrays):
    x = features(bank, arrays).reshape(len(bank), 4, -1)
    m = np.array(
        [[arithmetic(c["text"]) for c in q["candidates"]] for q in bank],
        dtype=np.float32,
    )
    return np.concatenate([x, m], axis=-1)


def pooled(p, answers, weight):
    p = p.copy()
    for k, aa in enumerate(answers):
        for a in set(aa) - {None}:
            ids = [i for i, b in enumerate(aa) if b == a]
            p[k, ids] = (1 - weight) * p[k, ids] + weight * p[k, ids].mean()
    return p


def probabilities(logits, valid):
    logits = np.where(valid, logits, -1e4)
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)
