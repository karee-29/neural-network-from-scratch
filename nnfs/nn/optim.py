import numpy as np

class Optimizer:
    def __init__(self, params, lr=1e-3, weight_decay=0.0, grad_clip=None):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def _gradient(self, p):
        g = p.grad
        if self.grad_clip is not None:
            norm = np.linalg.norm(g)
            if norm > self.grad_clip:
                g = g * (self.grad_clip / (norm + 1e-12))
        if self.weight_decay:
            g = g + self.weight_decay * p.data
        return g

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, grad_clip=5.0):
        super().__init__(params, lr, weight_decay, grad_clip)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            g = self._gradient(p)
            self.velocity[i] = self.momentum * self.velocity[i] + g
            p.data -= self.lr * self.velocity[i]

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=0.0, grad_clip=5.0):
        super().__init__(params, lr, weight_decay, grad_clip)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = self._gradient(p)
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*g
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(g*g)
            mhat = self.m[i] / (1-self.beta1**self.t)
            vhat = self.v[i] / (1-self.beta2**self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

class CosineAnnealing:
    def __init__(self, optimizer, max_lr, min_lr, total_steps):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = max(1, total_steps)
        self.step_count = 0

    def step(self):
        self.step_count += 1
        ratio = min(self.step_count, self.total_steps) / self.total_steps
        self.optimizer.lr = self.min_lr + 0.5*(self.max_lr-self.min_lr)*(1+np.cos(np.pi*ratio))
