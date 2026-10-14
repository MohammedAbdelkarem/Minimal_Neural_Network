import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SoftmaxCrossEntropy:
    """
    Softmax with cross entropy loss combined.
    """

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        """
        Parameters:
            x: logits
            t: labels (one-hot or class indices)
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1.0):
        batch_size = self.t.shape[0]

        if self.t.size == self.y.size:
            t = self.t.argmax(axis=1)
        else:
            t = self.t

        dx = self.y.copy()
        dx[np.arange(batch_size), t] -= 1
        dx = dx / batch_size
        return dout * dx
