import copy
import numpy as np
from .trainer import Trainer

class HyperparameterTuner:
    """
    Basic hyperparameter tuning using grid search over a parameter grid.
    """

    def __init__(self, base_network_config, param_grid, x_train, t_train, x_val, t_val, optimizer_class):
        """
        Parameters:
            base_network_config: dict with base config for NeuralNetwork (excluding hyperparameters to search)
            param_grid: dict of lists, e.g.
                {
                    "learning_rate": [0.1, 0.01],
                    "batch_size": [64, 128],
                    "epochs": [5, 10]
                }
            x_train, t_train: training data
            x_val, t_val: validation data
            optimizer_class: optimizer class (e.g., Adam, SGD)
        """
        self.base_network_config = base_network_config
        self.param_grid = param_grid
        self.x_train = x_train
        self.t_train = t_train
        self.x_val = x_val
        self.t_val = t_val
        self.optimizer_class = optimizer_class

        self.results = []
        self.best_config = None
        self.best_score = -np.inf

    def _iterate_grid(self):
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        def generate(idx, current):
            if idx == len(keys):
                yield dict(zip(keys, current))
            else:
                for v in values[idx]:
                    yield from generate(idx + 1, current + [v])

        return generate(0, [])

    def run(self, network_class):
        for config in self._iterate_grid():
            network_config = copy.deepcopy(self.base_network_config)
            network = network_class(**network_config)

            lr = config.get("learning_rate", 0.001)
            batch_size = config.get("batch_size", 64)
            epochs = config.get("epochs", 5)

            optimizer = self.optimizer_class(lr=lr)

            trainer = Trainer(
                network=network,
                optimizer=optimizer,
                x_train=self.x_train,
                t_train=self.t_train,
                x_test=self.x_val,
                t_test=self.t_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=False
            )

            trainer.fit()
            val_acc = network.accuracy(self.x_val, self.t_val)
            self.results.append((config, val_acc))

            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_config = config

        return self.best_config, self.best_score
