import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_mnist(test_size=10000, normalize=True):
    """
    Load MNIST using sklearn's fetch_openml.

    Returns:
        x_train, t_train, x_test, t_test
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    x_train, x_test, t_train, t_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    t_train = t_train.astype(int)
    t_test = t_test.astype(int)

    encoder = OneHotEncoder(sparse_output=False)
    t_train = encoder.fit_transform(t_train.reshape(-1, 1))
    t_test = encoder.transform(t_test.reshape(-1, 1))

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    else:
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

    return x_train, t_train, x_test, t_test
