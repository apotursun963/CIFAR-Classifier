# Cifar-10 Classification with Neural Network

## Projects Overview
This projects performs classification on the **CIFAR-10 dataset** using an artificial neural network (ANN) model.
Our model is trained using keras and Tensorflow, with the goal of correctly classifying each image in this dataset into its respective class.

This project includes the following steps:
1. Loading and preprocessing the CIFAR-10 dataset
2. Creating and training the artificial neural network model
3. Testing the model and calculating accuracy
4. Plotting accuracy and loss graphs during the tranining process
5. Saving the model and storing parameters for future use

## How the projects works

### 1. Loading the Dataset
In this project, the **CIFAR-10** dataset is loaded using the `keras.dataset` module. This dataset contains 32x32 color images in 10 different classes. These imaegs are **preprocessed** and the dataset split into **training** and **test** sets.
For more information, you can access the **CIFAR-10 dataset description** [cifar-10](docs/dataset_description.md).

### 2. Artificial Neural Network Model
The model is built using the **artificial neural network** (ANN) architecture. The model includes the following components:
- **Input layer**: Each image is of size 32x32x3 (3072 pixels), which is fed into the input layer.
- **Hidden layers**: There are three hidden layers, each containing 512 neurons, using the ReLU activation function.
- **Output layer**: The final layer consists of 10 neurons, and the **softmax** activation function is used to calculate the probability for each class.

For more information about the model architecture, you can access it **here**: [model-architecture](docs/model_architecture.md).

### 3. Training Process
During training, the model makes predictions by comparing with the **true labels** and updates its weights using the **backpropagation** algorithm to improve accuracy. The optimization algorithm used is **Gradient Descent** (GD), with hyperparameters such as learning rate and batch size adjust during the process.

### 4. Testing and Evaluation
Once the model is trained, the accuracy is calculated using the test dataset. Just like during training, predictions are made on the test data, and the model's overall performance is evaluated. Additionally, functions to visualize loss and accuracy graphs during the training process have been included.

### 5. Saving the Model
After training, all **weights** and **biases** are saved. This ensures that the trained model can be loaded and used again later.
