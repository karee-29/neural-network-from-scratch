# Interview Guide

### What did you build?

A NumPy-based neural-network framework with a dynamic reverse-mode autograd
engine, trainable modules, loss functions and optimizers. I used the framework
to train an MLP on MNIST without PyTorch or TensorFlow.

### How does backpropagation work?

Each operation creates a Tensor node containing references to its parents and a
local backward function. Calling `backward()` topologically orders the graph and
propagates gradients from the output back to leaf parameters using the chain
rule.

### Why implement cross entropy separately?

The loss combines stable softmax computation with negative log likelihood. This
avoids overflow/underflow issues that appear if naive exponentials and logs are
used independently.

### Why temporal/gradient testing?

For numerical ML systems, a gradient can be silently wrong while code still
runs. Numerical finite-difference checks compare the implemented derivative with
a local approximation and catch implementation errors.

### Why Adam?

Adam adapts the learning rate per parameter using first and second moments.
The implementation also includes bias correction, gradient clipping and weight
decay.

### Why not PyTorch?

The purpose of the project is to demonstrate understanding of the mechanics
underneath high-level deep-learning frameworks. PyTorch would be appropriate for
production model development, but intentionally isn't used in this framework.
