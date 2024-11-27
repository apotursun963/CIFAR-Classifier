# Neural Netwrok Model Architecture

## Overview
This document provides an overview of the custom neural network model implemented from scracth.
The model consists of an input layer, multiple hidden layers, and an output layer, with each layer having specific parameters. The architecture leverages the ReLu activation function for hidden layers and the softmax activation for the output layer.

## Model Class: `NeuralNet`
The neural network is implemented as a Python class `NeuralNet` with methods to initialize the netwrok, perform feedforward computation, backpropagation, and update parameters.

### Initialization
The netwrok is initialized with the following parameters:
- **Input Unit**: The number of input features.
- **Hidden Units**: A list specifying the number of neurons in each hidden layer.
- **Output Unit**: The number of neurons in the output layer.

### Key Components
- **Weights**: Initialized using the `He` initialization method, which scale the weights by the square root of 2 divided by the sum of the input and output units.
- **Biases**: Initialized to zero for all layers.

## Layer-wise Architecture
The model consists of:
- **Input Layer**: Accepts input features.
- **Hidden Layers**: Multiple hidden layers where each layer apllies the ReLu activation function.
- **Output Layer**: Applies the softmax activation function to produce the final output.

### Initialization of Parameters
The weights and biases for each layer are initialized as follows:
1. **Input to First Hidden Layer**:
   - Weights are initialized with random values, sclaed by `sqrt(2 / (input_units + first_hidden_units))`.
   - Biases are initialized to zero.
2. **Hidden Layers**:
   - Weights are initialized similarly to the input-hidden layer connection, using the size of adjacent layers.
   - Biases are initialized to zero.
3. **Last Hidden Layer to Output Layer**:
   - Weights and biases are initialized in the same way as above.

## Methods

### `initialize_parameters()`
This method initializes the weights and biases for all layers in the netwrok, based on the size of the input, hidden, and output layers. The weights are initialized using a random normal distribution, and biases are set to zero.

### `feedforward()`




