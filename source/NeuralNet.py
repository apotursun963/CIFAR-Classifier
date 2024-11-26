
from utils import relu, softmax, relu_derivative, accuracy, cross_entropy_loss
from DataLoader import Cifar10DataLoader
import numpy as np


# 1. parametreleri tanımla (w, b)
# 2. İleri besleme
# 3. geri yayılım
# 4. parametre güncelleme

# vs codda ve colab ve derfterde yazdığın bütün derin öğrenme notları toparla
# ve bunları gücel bir şekilde tr, en olmak üzere githuba at
# deep learning notes olsun yada farklı güzel olsun hem özet geçmiş olursun
# hemde ingilizcen gelişir


class NeuralNet:
    def __init__(self, input_unit, hidden_units, output_unit):
        self.hidden_lyrs = len(hidden_units)
        self.input_unit = input_unit
        self.hidden_units = hidden_units
        self.output_unit = output_unit
        self.initialize_parameters()

    def initialize_parameters(self):
        self.weights = []; self.biases = []

        # Initializing weights and biases for the input to first hidden layer
        self.weights.append(
            np.random.randn(self.input_unit, self.hidden_units[0]) * np.sqrt(2 / (self.input_unit + self.hidden_units[0]))
        )
        self.biases.append(np.zeros((1, self.hidden_units[0]))) 

        # Initializing weights and biases for hidden layers
        for i in range(self.hidden_lyrs -1):
            self.weights.append(
                np.random.randn(self.hidden_units[i], self.hidden_units[i+1]) * np.sqrt(2 / (self.hidden_units[i] + self.hidden_units[i+1]))
            ) 
            self.biases.append(np.zeros((1, self.hidden_units[i+1]))) 

        # Initializing weights and biases for the last hidden layer to output layer
        self.weights.append(
            np.random.randn(self.hidden_units[len(self.hidden_units) -1], self.output_unit) * np.sqrt(2 / (self.hidden_units[-1] + self.output_unit))
        ) 
        self.biases.append(np.zeros((1, self.output_unit)))

    def feedforward(self, X):
        self.lyr_outputs = []

        # first hidden layer output
        first_output = relu(np.dot(X, self.weights[0]) + self.biases[0])   
        self.lyr_outputs.append(first_output)

        # Subsequent hidden layers output
        for i in range(self.hidden_lyrs -1):
            output = relu(np.dot(self.lyr_outputs[i], self.weights[i+1]) + self.biases[i+1])   
            self.lyr_outputs.append(output)

        # final output layer
        self.final_output = softmax(np.dot(self.lyr_outputs[-1], self.weights[-1]) + self.biases[-1])   
        return (self.final_output)

    def backpropagation(self, inputs, Y):
        m = Y.shape[0]
        error_list = []
        dW = []; dB = []

        # Computing error and gradients at the output layer
        error_list.append(self.final_output - Y)                            
        dW.append((1/m) * np.dot(self.lyr_outputs[-1].T, error_list[0]))  
        dB.append((1/m) * np.sum(error_list[0], axis=0))

        # Computing error and gradients for hidden layers
        for i in range(self.hidden_lyrs):
            error_list.append(
                np.dot(error_list[-1], self.weights[len(self.weights) -i -1].T) * relu_derivative(self.lyr_outputs[len(self.lyr_outputs) -i -1])
            )
            dW.append((1/m) * np.dot(inputs.T if i == (self.hidden_lyrs - 1) else self.lyr_outputs[len(self.lyr_outputs) -i -2].T, error_list[-1]))
            dB.append((1/m) * np.sum(error_list[-1], axis=0))
        return (dW[::-1], dB[::-1])

    def update_parameters(self, dW, dB, alpha):
        for idx, (w, b, dw, db) in enumerate(zip(self.weights,self.biases, dW, dB)):
            self.weights[idx] = w - alpha * dw
            self.biases[idx] = b - alpha * db
        return (self.weights, self.biases)

    #  test dosyasında evulate evulate dosyasında olmalı
    def evulate(self):
        pass


