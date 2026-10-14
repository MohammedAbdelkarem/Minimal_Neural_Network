import numpy as np

class Trainer:
    """
    Trainer class to handle the training loop of a NeuralNetwork.
    """

    def __init__(self, network, optimizer, x_train, t_train,
                 x_test=None, t_test=None,
                 epochs=20, batch_size=100, verbose=True):
        self.network = network
        self.optimizer = optimizer
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // self.batch_size, 1)
        self.max_iter = self.epochs * self.iter_per_epoch

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch, train_flg=True)
        self.train_loss_list.append(loss)

    def fit(self):
        for i in range(self.max_iter):
            self.train_step()

            if self.verbose and (i % self.iter_per_epoch == 0):
                epoch = i // self.iter_per_epoch
                train_acc = self.network.accuracy(self.x_train, self.t_train)
                self.train_acc_list.append(train_acc)

                if self.x_test is not None and self.t_test is not None:
                    test_acc = self.network.accuracy(self.x_test, self.t_test)
                    self.test_acc_list.append(test_acc)
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Loss: {self.train_loss_list[-1]:.4f} - "
                          f"Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Loss: {self.train_loss_list[-1]:.4f} - "
                          f"Train Acc: {train_acc:.4f}")
