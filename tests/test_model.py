import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sources.DataLoader import Cifar10DataLoader
from sources.utils import relu, softmax, accuracy, cross_entropy_loss
import matplotlib.pyplot as plt
import numpy as np
import random


class TestNeuralNet:
    def __init__(self, num_layers, x_test, y_test, make_test=False) -> None:
        self.num_layers = num_layers
        self.x_test = x_test
        self.y_test = y_test
        self.make_test = make_test
        self.classes = [
            "Airplane", "Automobile", "Bird", "Cat", "Deer", 
            "Dog", "Frog", "Horse", "Ship", "Truck"
        ]

    def load_parameters(self):
        self.weight_lst = []; self.bias_lst = []

        for idx in range(self.num_layers - 1):
            self.weight_lst.append(np.load(f"output\\checkpoints\\weights\\W{idx + 1}.npy"))
            self.bias_lst.append(np.load(f"output\\checkpoints\\biases\\B{idx + 1}.npy"))
        return (self.weight_lst, self.bias_lst)

    def evulate(self):
        if not self.make_test:
            raise ValueError("you have to set the make_test=True to execute testing evulation")

        W, B = self.load_parameters()
    
        self.layer_outputs = [relu(np.dot(self.x_test, W[0]) + B[0])]
    
        for i in range(len(self.weight_lst) - 2):
            self.layer_outputs.append(relu(np.dot(self.layer_outputs[i], W[i + 1]) + B[i + 1]))

        self.prediction = softmax(np.dot(self.layer_outputs[-1], W[-1]) + B[-1])
        return (self.prediction)

    def compute_acc_loss(self):
        acc = accuracy(self.y_test, self.prediction)
        loss = cross_entropy_loss(self.y_test, self.prediction)
        print(f"Test loss: {loss:.4f} | Test accuracy: %{acc * 100:.4f}")
        
    def visualize_results(self, num_imgs=10):
        plt.figure(figsize=(8, 6))
        for idx in range(num_imgs):
            index = random.randint(0, len(self.x_test) - 1)
            plt.subplot(2, 5, idx + 1)
            plt.imshow(self.x_test[index].reshape(32, 32, 3))
            plt.title(f"true: {self.classes[np.argmax(self.y_test[index])]}\npred: {self.classes[np.argmax([self.prediction[index]])]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

data_loader = Cifar10DataLoader(load_data=True)
data_loader.load(load="test")
data_loader.process_data(data="test")


testing = TestNeuralNet(
    num_layers=5, 
    x_test=data_loader.test_imgs, 
    y_test=data_loader.test_labels,
    make_test=True
)

testing.evulate()

testing.visualize_results()

testing.compute_acc_loss()
