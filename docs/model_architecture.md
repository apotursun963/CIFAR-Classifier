# Neural Network Model Architecture

## Overview
This document provides an overview of the custom neural network model implemented from scratch. The model consists of an input layer, multiple hidden layers, and an output layer, with each layer having specific parameters. The architecture leverages the ReLU activation function for hidden layers and the softmax activation for the output layer.

## Model Class: `NeuralNet`
The neural network is implemented as a Python class `NeuralNet` with methods to initialize the network, perform feedforward computation, backpropagation, and update parameters.

### Initialization
The network is initialized with the following parameters:
- **Input Unit**: The number of input features.
- **Hidden Units**: A list specifying the number of neurons in each hidden layer.
- **Output Unit**: The number of neurons in the output layer.

### Key Components
- **Weights**: Initialized using the He initialization method, which scales the weights by the square root of 2 divided by the sum of the input and output units.
- **Biases**: Initialized to zero for all layers.

## Layer-wise Architecture
The model consists of:
- **Input Layer**: Accepts input features.
- **Hidden Layers**: Multiple hidden layers where each layer applies the ReLU activation function.
- **Output Layer**: Applies the softmax activation function to produce the final output.

### Initialization of Parameters
The weights and biases for each layer are initialized as follows:
1. **Input to First Hidden Layer**: 
   - Weights are initialized with random values, scaled by `sqrt(2 / (input_units + first_hidden_units))`.
   - Biases are initialized to zero.
2. **Hidden Layers**:
   - Weights are initialized similarly to the input-hidden layer connection, using the size of adjacent layers.
   - Biases are initialized to zero.
3. **Last Hidden Layer to Output Layer**:
   - Weights and biases are initialized in the same way as above.

## Methods

### `initialize_parameters()`
This method initializes the weights and biases for all layers in the network, based on the size of the input, hidden, and output layers. The weight matrices are initialized using a random normal distribution, and biases are set to zero.

### `feedforward(X)`
This method computes the forward pass through the network. It calculates the activations for all layers as follows:
- **First Hidden Layer**: The input `X` is multiplied by the weights and biases are added, followed by applying the ReLU activation function.
- **Subsequent Hidden Layers**: For each hidden layer, the previous layer’s output is used as input for the current layer.
- **Output Layer**: The final output is computed using the softmax activation.

### `backpropagation(inputs, Y)`
This method implements the backpropagation algorithm to compute the gradients of the weights and biases. It works by:
1. Calculating the error at the output layer.
2. Propagating the error backward through each hidden layer.
3. Calculating the gradients of the weights and biases at each layer.

### `update_parameters(dW, dB, alpha)`
This method updates the weights and biases using the gradients calculated by backpropagation. It applies gradient descent with a learning rate `alpha` to adjust the parameters.

## Activation Functions

### ReLU Activation Function
The ReLU (Rectified Linear Unit) function is used for all hidden layers. It is defined as:
```python
def relu(x):
    return np.maximum(0, x)
