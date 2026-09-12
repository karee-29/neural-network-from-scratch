# Implementation Notes

## Autograd engine

`Tensor` stores:
- NumPy data
- gradient buffer
- parent tensors
- operation name
- backward closure

`backward()` constructs a topological ordering of the dynamic computation graph
and applies reverse-mode differentiation.

Supported differentiable operations:
- addition
- subtraction
- multiplication
- division
- power
- matrix multiplication
- sum / mean
- exp / log
- ReLU
- sigmoid
- reshape
- transpose
- indexing

Broadcasted gradients are reduced back to their original shapes.

## Neural-network abstraction

`Module` provides:
- `forward`
- `train` / `eval`
- parameter discovery
- state dict export/import

`Linear`, `Dropout`, `ReLU`, `Sigmoid`, `Softmax` and loss modules are implemented
without PyTorch or TensorFlow.

## Numerical stability

Cross entropy uses the log-sum-exp / shifted-softmax pattern and clips the
selected probability before taking a logarithm.

## Optimisation

SGD supports momentum, weight decay and gradient clipping.

Adam implements first/second moment estimates with bias correction, plus weight
decay and gradient clipping.

A cosine learning-rate scheduler is included.

## Validation

The repository includes analytical-vs-numerical gradient checking. This is a
critical sanity test for an autograd implementation because a network can train
while still containing subtle derivative bugs.
