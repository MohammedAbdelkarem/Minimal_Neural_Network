import numpy as np
from .base import Layer

class ReLU(Layer):
    """
    ReLU activation: max(0, x)
    """

    def __init__(self):
        self.mask = None

    def forward(self, x, train_flg=True):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(Layer):
    """
    Sigmoid activation.
    """

    def __init__(self):
        self.out = None

    def forward(self, x, train_flg=True):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Tanh(Layer):
    """
    Tanh activation.
    """

    def __init__(self):
        self.out = None

    def forward(self, x, train_flg=True):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1 - self.out ** 2)
        return dx


class Linear(Layer):
    """
    Linear activation (identity), useful as a placeholder.
    """

    def forward(self, x, train_flg=True):
        return x

    def backward(self, dout):
        return dout
