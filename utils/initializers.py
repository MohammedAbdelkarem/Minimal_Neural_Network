import numpy as np

def xavier_init(in_dim, out_dim):
    """
    Xavier initialization for sigmoid/tanh.
    """
    return np.random.randn(in_dim, out_dim) / np.sqrt(in_dim)

def he_init(in_dim, out_dim):
    """
    He initialization for ReLU.
    """
    return np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)

def zeros_bias(dim):
    return np.zeros(dim)
