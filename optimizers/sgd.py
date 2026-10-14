from .base import Optimizer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
