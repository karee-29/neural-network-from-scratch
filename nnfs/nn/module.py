from collections import OrderedDict
import numpy as np
from .tensor import Tensor

class Module:
    def __init__(self):
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True
        for m in self.modules():
            m.training = True
        return self

    def eval(self):
        self.training = False
        for m in self.modules():
            m.training = False
        return self

    def modules(self):
        for value in self.__dict__.values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Module):
                        yield v

    def parameters(self):
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if isinstance(v, Module):
                        params.extend(v.parameters())
                    elif isinstance(v, Parameter):
                        params.append(v)
        return params

    def state_dict(self):
        state = OrderedDict()
        def walk(obj, prefix=""):
            for k, v in obj.__dict__.items():
                if isinstance(v, Parameter):
                    state[prefix+k] = v.data.copy()
                elif isinstance(v, Module):
                    walk(v, prefix+k+".")
                elif isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        if isinstance(item, Module):
                            walk(item, prefix+f"{k}.{i}.")
        walk(self)
        return state

    def load_state_dict(self, state):
        def walk(obj, prefix=""):
            for k, v in obj.__dict__.items():
                if isinstance(v, Parameter):
                    if prefix+k in state:
                        v.data[...] = state[prefix+k]
                elif isinstance(v, Module):
                    walk(v, prefix+k+".")
                elif isinstance(v, (list, tuple)):
                    for i, item in enumerate(v):
                        if isinstance(item, Module):
                            walk(item, prefix+f"{k}.{i}.")
        walk(self)

class Parameter(Tensor):
    def __init__(self, data, name=None):
        super().__init__(data, requires_grad=True, name=name)

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def modules(self):
        for layer in self.layers:
            yield layer
            if isinstance(layer, Module):
                yield from layer.modules()

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params
