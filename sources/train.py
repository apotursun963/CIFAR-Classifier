from utils import cross_entropy_loss, accuracy
from DataLoader import Cifar10DataLoader
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
import numpy as np
import time
import os

class TrainNeuralNet: 
    def __init__(self, Model) -> None:
        self.model = Model

    def train(self, x_train, y_train, epoch, learning_rate, batch_size): 
        self.loss_list = []
        self.accuracy_list = []

        start = time.time()
        for i in range(1, epoch +1):
            batch_loss = []
            batch_acc = []

            for j in range(0, len(x_train), batch_size):
                x_batch = x_train[j: j + batch_size]
                y_batch = y_train[j: j + batch_size]

                predictions = self.model.feedforward(x_batch)
                dW, dB = self.model.backpropagation(x_batch, y_batch)
                self.model.update_parameters(dW, dB, learning_rate)

                batch_loss.append(cross_entropy_loss(y_batch, predictions))
                batch_acc.append(accuracy(y_batch, predictions))

            epoch_loss = np.mean(batch_loss)
            epoch_acc = np.mean(batch_acc)

            self.loss_list.append(epoch_loss)
            self.accuracy_list.append(epoch_acc)

            if (i % 1 == 0):
                print(f"Epoch: {i} | Loss: {epoch_loss:.4f} | Accuracy: %{epoch_acc * 100:.2f}")
        end = time.time()
        print(f"Training duration of model: {(end - start) / 60}")

    def plot_acc_loss(self):
        _, axs = plt.subplots(2, 1, figsize=(8,6))
        # accuracy
        axs[0].plot(self.accuracy_list, label="Accuracy", color="b")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()
        axs[0].grid(True)
        # loss
        axs[0].plot(self.loss_list, label="Loss", color="r")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)
        
        plt.tight_layout()
        plt.show()

    def save_parameters(self):
        for idx, (weight, bias) in enumerate(zip(self.model.weights, self.model.biases)):
            np.save(os.path.join(f"output\\checkpoints\\weights\\W{idx + 1}.npy"), weight)
            np.save(os.path.join(f"output\\checkpoints\\biases\\B{idx + 1}.npy"), bias)
        print("Model Parameters (Weights, Biases) Successfully Saved.")


data_loader = Cifar10DataLoader(load_data=True)
data_loader.load(load="train")
data_loader.process_data(data="train")
data_loader.visualize_images(num_imgs=10, dataset="train")

model = NeuralNet(
    input_unit=3072,
    hidden_units=[256, 512, 256],
    output_unit=10
)

trainer = TrainNeuralNet(model)

trainer.train(
    data_loader.train_imgs,
    data_loader.train_labels,
    epoch=1,
    learning_rate=0.01,
    batch_size=64
)

trainer.plot_acc_loss()
trainer.save_parameters()
