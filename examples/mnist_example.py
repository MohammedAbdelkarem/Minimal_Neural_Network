import matplotlib.pyplot as plt

from MiniNN.core import NeuralNetwork, Trainer
from MiniNN.optimizers import Adam
from MiniNN.utils import load_mnist

def main():
    x_train, t_train, x_test, t_test = load_mnist(test_size=10000, normalize=True)

    layers_config = [
        {"type": "dense", "units": 100, "activation": "sigmoid", "batchnorm": True},
        {"type": "dense", "units": 50, "activation": "relu"},
        {"type": "dense", "units": 10}  # final logits layer
    ]

    network = NeuralNetwork(
        input_size=784,
        layers_config=layers_config,
        loss_type="softmax_ce",
        weight_init="xavier"
    )

    optimizer = Adam(lr=0.001)

    trainer = Trainer(
        network=network,
        optimizer=optimizer,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=5,
        batch_size=128,
        verbose=True
    )

    trainer.fit()

    plt.plot(trainer.train_loss_list, label="Train Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.show()

    print("Final Train Accuracy:", network.accuracy(x_train, t_train))
    print("Final Test Accuracy:", network.accuracy(x_test, t_test))

if __name__ == "__main__":
    main()
