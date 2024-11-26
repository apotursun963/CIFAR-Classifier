
from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.utils import to_categorical
import matplotlib.pyplot as plt
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
            plt.imshow(image)
            plt.title(f"{self.classes[int(label[0])]}")
            plt.axis("off")
        plt.show()
    
    def process_data(self, data=None):
        if data == "train" and self.load_data:
            if self.train_imgs is None or self.train_labels is None:
                raise ValueError("Training data is not loaded. Please load training data first.")
            self.train_imgs = self.train_imgs.reshape(self.train_imgs.shape[0], 32 * 32 * 3) / 255.0
            self.train_labels = to_categorical(self.train_labels, num_classes=10)
            print("Train data pre-processing Complete.")

        elif data == "test" and self.load_data:
            if self.test_imgs is None or self.test_labels is None:
                raise ValueError("Testing data is not loaded. Please load testing data first.")
            self.test_imgs = self.test_imgs.reshape(self.test_imgs.shape[0],  32 * 32 * 3) / 255.0
            self.test_labels = to_categorical(self.test_labels, num_classes=10)
            print("Test data pre-processing Complete.")

        else:
            raise ValueError("Please specify the data type as 'train' or 'test'.")
    def data_summary(self):
        pass

    def split_data(self):   ## to validation
        pass



# data_loader = Cifar10DataLoader(load_data=True)

# data_loader.load(load="both")

# data_loader.visualize_images(num_imgs=10, dataset="test")

# data_loader.process_data(data="train")

