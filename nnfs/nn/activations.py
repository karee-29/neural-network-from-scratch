import numpy as np
from .module import Module
from .tensor import Tensor

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()

class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        # Stable softmax. max is detached for numerical stabilisation.
        shifted = x.data - np.max(x.data, axis=self.axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=self.axis, keepdims=True)
        out = Tensor(probs, x.requires_grad, (x,), "softmax")
        def _backward():
            if x.requires_grad:
                dot = np.sum(out.grad * probs, axis=self.axis, keepdims=True)
                x.grad += probs * (out.grad - dot)
        out._backward = _backward
        return out
