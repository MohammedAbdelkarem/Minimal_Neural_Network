MiniNN
=================

A minimal neural network library built entirely from scratch using NumPy, inspired by frameworks like TensorFlow and PyTorch, and designed for educational purposes.
This project demonstrates how deep learning frameworks work internally, including:

Forward propagation

Backpropagation

Parameter updates using optimizers

Batch Normalization and Dropout

Training loops and hyperparameter tuning

Modular layer-based architecture

The library is fully extensible and easy to modify or expand.

Features
Layers
Fully connected (Affine/Dense)

Activation functions:

ReLU

Sigmoid

Tanh

Linear

Batch Normalization

Dropout

Loss Functions
Mean Squared Error (MSE)

Softmax Cross Entropy

Optimizers
SGD

Momentum

AdaGrad

Adam

Core Components
NeuralNetwork class for building arbitrary architectures

Trainer class for training loops, metrics, and logging

HyperparameterTuner for basic grid search

MNIST loader using sklearn

Project Structure
Code
MiniNN/
│
├── layers/
├── losses/
├── optimizers/
├── core/
├── utils/
├── examples/
└── README.md
Each folder contains modular components that can be combined to build any feed‑forward neural network.

Getting Started
This section explains how to import the library, load data, define a model, choose an optimizer, and train the network.

You can run this code inside VS Code, Jupyter Notebook, or any Python environment.

1. Add the package to your Python path (if using Jupyter)
```python
import sys
sys.path.append("path/to/your/project")
```
Example:

```python
sys.path.append("D:/college/Nueral Networks/homeworks")
```

2. Import the required components
```python
from MiniNN.core import NeuralNetwork, Trainer
from MiniNN.optimizers import Adam
from MiniNN.utils import load_mnist
```

3. Load the MNIST dataset
This function automatically:

Loads MNIST using sklearn

Normalizes pixel values

One‑hot encodes labels

Splits into train/test

```python
x_train, t_train, x_test, t_test = load_mnist()
```

4. Define the network architecture
The architecture is defined using a list of dictionaries.
Each dictionary represents one layer.

```python
layers_config = [
    {"type": "dense", "units": 100, "activation": "sigmoid", "batchnorm": True},
    {"type": "dense", "units": 50, "activation": "relu"},
    {"type": "dense", "units": 10}
]
```

Supported keys:

type: always "dense" in this version

units: number of neurons

activation: "relu", "sigmoid", "tanh", "linear"

batchnorm: True/False

dropout: dropout ratio (optional)

You can add any number of layers.

5. Create the neural network

```python
network = NeuralNetwork(
    input_size=784,
    layers_config=layers_config,
    loss_type="softmax_ce",
    weight_init="xavier"
)
```

6. Choose an optimizer
Example using Adam:

```python
optimizer = Adam(lr=0.001)
Other options:

SGD(lr=...)

Momentum(lr=..., momentum=...)

AdaGrad(lr=...)
```

7. Train the network

```python
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
```


During training, the trainer will:

Perform forward and backward passes

Update weights

Compute loss

Compute train/test accuracy

Print progress every epoch

8. Done!

```python
print("training finished")
```

Example Output
Code
Epoch 1/3 - Loss: 0.4321 - Train Acc: 0.9123 - Test Acc: 0.9055
Epoch 2/3 - Loss: 0.2987 - Train Acc: 0.9351 - Test Acc: 0.9280
Epoch 3/3 - Loss: 0.2456 - Train Acc: 0.9478 - Test Acc: 0.9402
training finished

A full example is available in:

MiniNN/examples/mnist_example.py