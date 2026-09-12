import numpy as np
from nnfs.nn import Tensor

def test_add_backward():
    x = Tensor(np.array([[1.,2.],[3.,4.]]), requires_grad=True)
    y = Tensor(np.array([[2.,4.],[6.,8.]]), requires_grad=True)
    z = (x*y).sum()
    z.backward()
    np.testing.assert_allclose(x.grad, y.data)
    np.testing.assert_allclose(y.grad, x.data)

def test_matmul_backward():
    rng = np.random.default_rng(1)
    x = Tensor(rng.normal(size=(4,3)), requires_grad=True)
    w = Tensor(rng.normal(size=(3,2)), requires_grad=True)
    z = (x @ w).sum()
    z.backward()
    np.testing.assert_allclose(x.grad, np.ones((4,2)) @ w.data.T)
    np.testing.assert_allclose(w.grad, x.data.T @ np.ones((4,2)))

def test_relu_backward():
    x = Tensor(np.array([-2., -1., 0., 2.]), requires_grad=True)
    y = x.relu().sum()
    y.backward()
    np.testing.assert_allclose(x.grad, np.array([0.,0.,0.,1.]))

def test_broadcast_backward():
    x = Tensor(np.ones((3,4)), requires_grad=True)
    b = Tensor(np.ones(4), requires_grad=True)
    y = (x+b).sum()
    y.backward()
    np.testing.assert_allclose(b.grad, np.ones(4)*3)
