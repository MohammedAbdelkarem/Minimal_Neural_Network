import abc

class Layer(abc.ABC):
    """
    Base class for all layers in the network.
    Each layer must implement forward and backward.
    """

    @abc.abstractmethod
    def forward(self, x, train_flg=True):
        """
        Perform forward pass.

        Parameters:
            x: input array
            train_flg: whether we are in training mode (some layers behave differently in inference)

        Returns:
            Output of the layer.
        """
        pass

    @abc.abstractmethod
    def backward(self, dout):
        """
        Perform backward pass.

        Parameters:
            dout: gradient w.r.t layer output

        Returns:
            Gradient w.r.t layer input.
        """
        pass
