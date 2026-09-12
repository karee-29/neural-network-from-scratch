import numpy as np

_DTYPE = np.float64

def _ensure_array(x):
    if isinstance(x, Tensor):
        return x.data
    return np.asarray(x, dtype=_DTYPE)

def _unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=_DTYPE)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)

class Tensor:
    """NumPy-backed scalar/tensor with reverse-mode automatic differentiation."""

    def __init__(self, data, requires_grad=False, _children=(), _op="", name=None):
        self.data = _ensure_array(data).copy()
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def zero_grad(self):
        if self.requires_grad:
            self.grad.fill(0.0)

    def detach(self):
        return Tensor(self.data.copy(), requires_grad=False)

    def item(self):
        return float(self.data.item())

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}, op='{self._op}')"

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor without requires_grad=True")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be provided for non-scalar outputs")
            grad = np.ones_like(self.data)
        self.grad = self.grad + np.asarray(grad, dtype=_DTYPE)

        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad, (self, other), "+")
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -Tensor(other))

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad, (self, other), "*")
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1)

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def pow(self, exponent):
        out = Tensor(self.data ** exponent, self.requires_grad, (self,), f"pow({exponent})")
        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(
                    out.grad * exponent * (self.data ** (exponent - 1)), self.shape
                )
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other), "@")
        def _backward():
            if self.requires_grad:
                gx = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                self.grad += _unbroadcast(gx, self.shape)
            if other.requires_grad:
                gw = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                other.grad += _unbroadcast(gw, other.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")
        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    for a in sorted([a if a >= 0 else self.ndim + a for a in axes]):
                        g = np.expand_dims(g, a)
                self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in ((axis,) if isinstance(axis, int) else tuple(axis))]
        )
        return self.sum(axis=axis, keepdims=keepdims) / float(denom)

    def exp(self):
        val = np.exp(self.data)
        out = Tensor(val, self.requires_grad, (self,), "exp")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * val
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0),
                     self.requires_grad, (self,), "relu")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -60, 60)))
        out = Tensor(s, self.requires_grad, (self,), "sigmoid")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad * s * (1 - s)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), self.requires_grad, (self,), "transpose")
        inverse = np.argsort(axes) if axes is not None else None
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(inverse) if inverse is not None else out.grad.T
        out._backward = _backward
        return out

    T = property(lambda self: self.transpose())

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.requires_grad, (self,), "slice")
        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

def tensor(data, requires_grad=False, name=None):
    return Tensor(data, requires_grad=requires_grad, name=name)
