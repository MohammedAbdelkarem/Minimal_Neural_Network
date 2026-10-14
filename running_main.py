import sys

sys.path.append("D:/college/Nueral Networks/homeworks")


from MiniNN.core import NeuralNetwork, Trainer
from MiniNN.optimizers import Adam
from MiniNN.utils import load_mnist



x_train, t_train, x_test, t_test = load_mnist()


# Define the network architecture
# The architecture is defined using a list of dictionaries.
# Each dictionary represents one layer.


layers_config = [
    {"type": "dense", "units": 100, "activation": "relu", "batchnorm": True},
    {"type": "dense", "units": 50, "activation": "relu"},
    {"type": "dense", "units:": 50, "acitvation": "relu"},
    {"type": "dense", "units": 10}
]


# Supported keys:

# type: always "dense" in this version

# units: number of neurons

# activation: "relu", "sigmoid", "tanh", "linear"

# batchnorm: True/False

# dropout: dropout ratio (optional)

# You can add any number of layers.


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
    epochs=3,
    batch_size=128
)

trainer.fit()



# During training, the trainer will:

# Perform forward and backward passes

# Update weights

# Compute loss

# Compute train/test accuracy

# Print progress every epoch

print("training finished")

