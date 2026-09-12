import numpy as np
from nnfs.nn import Tensor

def numerical_gradient(fn, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        plus = fn(x)
        x[idx] = old - eps
        minus = fn(x)
        x[idx] = old
        grad[idx] = (plus-minus)/(2*eps)
        it.iternext()
    return grad

rng = np.random.default_rng(7)
x = Tensor(rng.normal(size=(3,4)), requires_grad=True)
w = Tensor(rng.normal(size=(4,2)), requires_grad=True)

out = ((x @ w).relu()).sum()
out.backward()

analytic_x = x.grad.copy()
raw_x = x.data.copy()
numerical_x = numerical_gradient(
    lambda arr: np.maximum(arr @ w.data, 0).sum(), raw_x
)

error = np.max(np.abs(analytic_x-numerical_x))
print("max absolute gradient error:", error)
assert error < 1e-6, f"gradient check failed: {error}"
print("gradient check passed")
