import numpy as np

class MeanSquaredError:
    """
    Mean Squared Error loss.
    """

    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        """
        Parameters:
            y: predictions
            t: targets
        """
        self.y = y
        self.t = t
        loss = 0.5 * np.mean((y - t) ** 2)
        return loss

    def backward(self, dout=1.0):
        """
        Gradient of MSE w.r.t y.
        """
        batch_size = self.t.shape[0]
        dy = (self.y - self.t) / batch_size
        return dout * dy
