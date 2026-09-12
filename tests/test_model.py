import numpy as np
from nnfs.nn import Tensor, Sequential, Linear, ReLU, CrossEntropyLoss, Adam

def test_mlp_backward_and_update():
    rng = np.random.default_rng(2)
    X = Tensor(rng.normal(size=(12,5)))
    y = np.arange(12) % 3
    model = Sequential(Linear(5,8), ReLU(), Linear(8,3))
    loss_fn = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=1e-2)

    before = [p.data.copy() for p in model.parameters()]
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    after = [p.data.copy() for p in model.parameters()]
    assert all(np.any(a != b) for a,b in zip(before,after))

def test_cross_entropy_finite():
    logits = Tensor(np.array([[1000.,0.,-1000.],[-1000.,1000.,0.]]), requires_grad=True)
    loss = CrossEntropyLoss()(logits, np.array([0,1]))
    assert np.isfinite(loss.item())
    loss.backward()
    assert np.all(np.isfinite(logits.grad))
