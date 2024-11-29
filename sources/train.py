from utils import cross_entropy_loss, accuracy
from DataLoader import Cifar10DataLoader
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
import numpy as np
import time
import os

class TrainNeuralNet:
    def __init__(self, Model):
        self.model = Model            # Initializing the model
    
    def train(self, x_train, y_train, epoch, learning_rate, batch_size): 
        self.loss_list = []
        self.accuracy_list = []

        start = time.time()
        for i in range(1, epoch +1):
            for batch in range(0, len(x_train), batch_size):
                x_batch = x_train[batch: batch + batch_size]
                y_batch = y_train[batch: batch + batch_size]
                
                self.model.feedforward(x_train)
                dW, dB = self.model.backpropagation(x_train, y_train)
                self.model.update_parameters(dW, dB, learning_rate)
            
            predictions_epoch = self.model.feedforward(x_train)
            loss = cross_entropy_loss(y_train, predictions_epoch)
            self.loss_list.append(loss)
            acc = accuracy(y_train, predictions_epoch)
            self.accuracy_list.append(acc)

            if i % 10 == 0 or i == epoch:
                print(f"Epoch {i} - Loss: {loss:.3f} - Accuracy: {acc * 100:.2f}")
        end = time.time()
        print(f"Training duration of model: {(end - start) / 60:.2f} minute")

    def plot_acc_loss(self):
        _, axs = plt.subplots(2, 1, figsize=(8,6))
        # accuracy
        axs[0].plot(self.accuracy_list, label="Accuracy", color="b")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()
        axs[0].grid(True)
        # loss
        axs[1].plot(self.loss_list, label="Loss", color="r")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def save_parameters(self):
        root_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), ".")), "output", "chekpoints")

        weights_path = os.path.join(root_path, "weights")
        biases_path = os.path.join(root_path, "biases")
        os.makedirs(weights_path, exist_ok=True)
        os.makedirs(biases_path, exist_ok=True)

        for idx, (weight, bias) in enumerate(zip(self.model.weights, self.model.biases)):
            np.save(os.path.join(weights_path, f"W{idx + 1}.npy"), weight)
            np.save(os.path.join(biases_path, f"B{idx + 1}.npy"), bias)
        print("Model Parameters (Weights, Biases) Successfully Saved.")


# Initializing the CIFAR-10 data loader
data_loader = Cifar10DataLoader(load_data=True)
data_loader.load(load="train")
data_loader.process_data(data="train")
data_loader.visualize_images(num_imgs=10, dataset="train")

# Creating an nn from the NeuralNet class
model = NeuralNet(
    input_unit=3072,
    hidden_units=[256, 512, 256],
    output_unit=10
)

# Initializing TrainNeuralNet class to train the model
trainer = TrainNeuralNet(model)

# Training the model for 7500 epochs, learning rate is 0.01
trainer.train(
    data_loader.train_imgs,
    data_loader.train_labels,
    epoch=200,
    learning_rate=0.01,
    batch_size=64
)

# We draw the accuracies and losses in train
trainer.plot_acc_loss()

# Saving the parameters of the model
trainer.save_parameters()
