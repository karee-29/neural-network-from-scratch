import numpy as np
from .module import Module, Parameter
from .tensor import Tensor

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed + in_features + out_features)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(rng.uniform(-limit, limit, (in_features, out_features)), "weight")
        self.bias = Parameter(np.zeros(out_features), "bias") if bias else None

    def forward(self, x):
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out

class Dropout(Module):
    def __init__(self, p=0.1, seed=42):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = np.random.default_rng(seed)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (self.rng.random(x.shape) >= self.p).astype(float) / (1 - self.p)
        return x * Tensor(mask)
