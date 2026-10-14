import abc

class Optimizer(abc.ABC):
    """
    Base class for all optimizers.
    """

    @abc.abstractmethod
    def update(self, params, grads):
        """
        Update parameters in-place using the provided gradients.
        

        Parameters:
            params: dictionary of parameters (e.g., weights, biases)
            grads: dictionary of gradients (same keys as params)
        """
        pass
