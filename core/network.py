import numpy as np
from collections import OrderedDict

from MiniNN.layers import Affine, ReLU, Sigmoid, Tanh, Linear, BatchNormalization, Dropout
from MiniNN.losses import MeanSquaredError, SoftmaxCrossEntropy
from MiniNN.utils import xavier_init, he_init, zeros_bias

class NeuralNetwork:
    """
    General feed-forward neural network composed of layers.
    Supports fully connected networks with activation, BatchNorm, Dropout, and loss layers.
    """

    def __init__(self, input_size, layers_config, loss_type="softmax_ce", weight_init="xavier"):
        """
        Parameters:
            input_size: dimension of input features
            layers_config: list of dicts defining each layer, e.g.
                [
                    {"type": "dense", "units": 100, "activation": "sigmoid", "batchnorm": True},
                    {"type": "dense", "units": 50, "activation": "relu"},
                    {"type": "dense", "units": 10}
                ]
            loss_type: "softmax_ce" or "mse"
            weight_init: "xavier" or "he"
        """
        self.input_size = input_size
        self.layers_config = layers_config
        self.weight_init = weight_init

        self.params = {}
        self.layers = OrderedDict()
        self.loss_layer = None

        self._build_network(loss_type)

    def _init_weights(self, in_dim, out_dim, idx):
        if self.weight_init == "he":
            W = he_init(in_dim, out_dim)
        else:
            W = xavier_init(in_dim, out_dim)

        b = zeros_bias(out_dim)
        self.params[f"W{idx}"] = W
        self.params[f"b{idx}"] = b
        return W, b

    def _build_network(self, loss_type):
        """
        Build the network from layers_config.
        """
        prev_dim = self.input_size
        affine_count = 0
        bn_count = 0

        for i, conf in enumerate(self.layers_config):
            if conf["type"] == "dense":
                units = conf["units"]
                affine_count += 1
                W, b = self._init_weights(prev_dim, units, affine_count)
                self.layers[f"Affine{affine_count}"] = Affine(W, b)
                prev_dim = units

                if conf.get("batchnorm", False):
                    bn_count += 1
                    gamma = np.ones(units)
                    beta = np.zeros(units)
                    self.params[f"gamma{bn_count}"] = gamma
                    self.params[f"beta{bn_count}"] = beta
                    self.layers[f"BatchNorm{bn_count}"] = BatchNormalization(
                        self.params[f"gamma{bn_count}"],
                        self.params[f"beta{bn_count}"]
                    )

                activation = conf.get("activation", None)
                if activation is not None:
                    if activation.lower() == "relu":
                        self.layers[f"ReLU{i+1}"] = ReLU()
                    elif activation.lower() == "sigmoid":
                        self.layers[f"Sigmoid{i+1}"] = Sigmoid()
                    elif activation.lower() == "tanh":
                        self.layers[f"Tanh{i+1}"] = Tanh()
                    elif activation.lower() == "linear":
                        self.layers[f"Linear{i+1}"] = Linear()

                if conf.get("dropout", None) is not None:
                    ratio = conf["dropout"]
                    self.layers[f"Dropout{i+1}"] = Dropout(dropout_ratio=ratio)

        if loss_type == "softmax_ce":
            self.loss_layer = SoftmaxCrossEntropy()
        elif loss_type == "mse":
            self.loss_layer = MeanSquaredError()
        else:
            raise ValueError("Unsupported loss_type: {}".format(loss_type))

    def predict(self, x, train_flg=False):
        """
        Forward pass through all layers (without applying loss).
        """
        for name, layer in self.layers.items():
            x = layer.forward(x, train_flg=train_flg)
        return x

    def loss(self, x, t, train_flg=True):
        """
        Compute loss given inputs and targets.
        """
        y = self.predict(x, train_flg=train_flg)
        return self.loss_layer.forward(y, t)

    def accuracy(self, x, t):
        """
        Compute classification accuracy for classification tasks.
        For SoftmaxCrossEntropy, t may be one-hot or label indices.
        """
        y = self.predict(x, train_flg=False)

        if isinstance(self.loss_layer, SoftmaxCrossEntropy):
            y = np.argmax(y, axis=1)
            if t.ndim != 1:
                t = np.argmax(t, axis=1)
            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy
        else:
            return None

    def gradient(self, x, t):
        """
        Compute gradients of the loss with respect to all parameters using backpropagation.
        """
        self.loss(x, t, train_flg=True)

        dout = 1
        dout = self.loss_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        affine_idx = 0
        bn_idx = 0

        for name, layer in self.layers.items():
            if isinstance(layer, Affine):
                affine_idx += 1
                grads[f"W{affine_idx}"] = layer.dW
                grads[f"b{affine_idx}"] = layer.db
            if isinstance(layer, BatchNormalization):
                bn_idx += 1
                grads[f"gamma{bn_idx}"] = layer.dgamma
                grads[f"beta{bn_idx}"] = layer.dbeta

        return grads
