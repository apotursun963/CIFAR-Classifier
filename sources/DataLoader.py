from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import random


class Cifar10DataLoader:
    def __init__(self, load_data=False) -> None:
        self.load_data = load_data
        self.train_imgs = None
        self.train_labels = None
        self.test_imgs = None
        self.test_labels = None
        self.classes = [
            "Airplane", "Automobile", "Bird", "Cat", "Deer", 
            "Dog", "Frog", "Horse", "Ship", "Truck"
        ]

    def load(self, load=None):
        if self.load_data == False:
            raise ValueError("please set load_data=True to load the data.")
        
        elif self.load_data and load == "both":
            (self.train_imgs, self.train_labels), (self.test_imgs, self.test_labels) = cifar10.load_data()
            print("Successfully loaded train and test datasets")

        elif self.load_data and load == "train":
            (self.train_imgs, self.train_labels), (_ , _) = cifar10.load_data()
            print("Successfully loaded train dataset")

        elif self.load_data and load == "test":
            (_ , _), (self.test_imgs, self.test_labels) = cifar10.load_data()
            print("Successfully loaded test dataset")

        else:
            raise ValueError("Data type must be 'train', 'test' or 'both'")
    
    def visualize_images(self, num_imgs, dataset=None):
        if dataset == "train" and self.train_imgs is None:
            raise ValueError("Training data is not loaded. Please load training data first.")

        elif dataset == "test" and self.test_imgs is None:
            raise ValueError("Testing data is not loaded. Please load testing data first.")

        plt.figure(figsize=(10, 5))
        for i in range(num_imgs):
            if dataset == "train":
                index = random.randint(0, len(self.train_imgs) - 1)
                image = self.train_imgs[index]
                label = self.train_labels[index]
            else:
                index = random.randint(0, len(self.test_imgs) - 1)
                image = self.test_imgs[index]
                label = self.test_labels[index]
            plt.subplot(2, 5, i + 1)
            plt.imshow(image.reshape(32, 32, 3))
            plt.title(f"{self.classes[np.argmax(label)]}")
            plt.axis("off")
        plt.show()
    
    def process_data(self, data=None):
        if data == "both" and self.load_data:
            raise ValueError("Training data is not loaded. Please load training data first.")

        if data == "both":
            if self.train_imgs is None or self.test_imgs is None:
                raise ValueError("data not loaded please first load the data")
            self.process_single_data(self.train_imgs, self.train_labels)
            self.process_single_data(self.test_imgs, self.test_labels)
            print("Train and Test data pre-processing Complete.")
        
        elif data == "train":
            if self.train_imgs is None:
                raise ValueError("Training data is not loaded. Please load training data first.")
            self.process_single_data(self.train_imgs, self.train_imgs)
            print("Train data pre-processing Complete.")

        elif data == "test":
            if self.test_imgs is None:
                raise ValueError("Testing data is not loaded. Please load testing data first.")
            self.process_single_data(self.test_imgs, self.test_labels)
            print("Test data pre-processing Complete.")
        else:
            raise ValueError("Please specify the data type as 'train' or 'test'.")
    
    def process_single_data(self, images, labels):
        images = images.reshape(images.shape[0], 32 * 32 * 3) / 255.0
        labels = to_categorical(labels, num_classes=10)
        return (images, labels)

    def data_summary(self):
        if not self.load_data:
            raise ValueError("please set load_data=True to load the data.")

        elif self.train_imgs is not None and self.train_labels is not None:
            print(f"Shape of Training Datasets: {self.train_imgs.shape}")
        
        elif self.test_imgs is not None and self.test_labels is not None:
            print(f"Shape of Testing Datasets: {self.test_imgs.shape}")

    def split_data(self):   ## to validation
        pass


dataloader = Cifar10DataLoader(load_data=True)
dataloader.load(load="test")
dataloader.process_data(data="test")
dataloader.data_summary()
