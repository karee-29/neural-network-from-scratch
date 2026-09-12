import numpy as np
from nnfs.nn import Tensor, Sequential, Linear, ReLU, Sigmoid, CrossEntropyLoss, Adam

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0,1,1,0])

model = Sequential(
    Linear(2, 16),
    ReLU(),
    Linear(16, 8),
    ReLU(),
    Linear(8, 2),
)
loss_fn = CrossEntropyLoss()
opt = Adam(model.parameters(), lr=0.03)

for epoch in range(1500):
    opt.zero_grad()
    logits = model(Tensor(X))
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()

pred = np.argmax(model(Tensor(X)).data, axis=1)
print("XOR predictions:", pred.tolist())
print("XOR accuracy:", float((pred == y).mean()))
