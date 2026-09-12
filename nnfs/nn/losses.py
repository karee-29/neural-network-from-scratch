import numpy as np
from .module import Module
from .tensor import Tensor

class CrossEntropyLoss(Module):
    """Numerically stable softmax + cross entropy for integer class labels."""
    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be mean or sum")
        self.reduction = reduction

    def forward(self, logits, targets):
        y = np.asarray(targets, dtype=int)
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        n = logits.data.shape[0]
        loss_values = -np.log(np.clip(probs[np.arange(n), y], 1e-12, None))
        loss_value = loss_values.mean() if self.reduction == "mean" else loss_values.sum()
        out = Tensor(loss_value, logits.requires_grad, (logits,), "cross_entropy")
        def _backward():
            if logits.requires_grad:
                grad = probs.copy()
                grad[np.arange(n), y] -= 1
                if self.reduction == "mean":
                    grad /= n
                logits.grad += out.grad * grad
        out._backward = _backward
        return out

class MSELoss(Module):
    def forward(self, pred, target):
        target = target if isinstance(target, Tensor) else Tensor(target)
        diff = pred - target
        return (diff * diff).mean()
