import numpy as np

def softmax(x):
    """
    The softmax function is a function used in multiple classification problems and converts the output values 
    ​​of the model into probability values ​​between [0, 1]. and the sum of the probability values ​​is always 1.
    """
    exps = np.exp(x - np.max(x)) 
    return (exps / np.sum(exps, axis=1, keepdims=True))

def relu(x):
    """
    relu, is a non-linear activation function used in the hidden layers of neural networks. 
    Its fundamental principle is to zero out negative values while allowing positive values to pass through unchanged. 
    This characteristic helps the model learn quickly.
    """
    return (np.maximum(0, x))

def relu_derivative(x):
    """
    The relu derivative represents the gradient of the ReLU activation function. 
    For negative inputs, the derivative is 0, meaning the neurons cannot be updated and may become "dead." 
    For positive inputs, the derivative is 1, allowing the neurons to be updated normally. 
    In summary, the ReLU derivative determines how neurons are updated during the learning process.
    """
    return (np.where(x > 0, 1, 0))

def accuracy(Y, predictions):
    """
    The accuracy is a metric used to evaluate the performance (correctness) of the model. 
    It calculates how accurately the model predicts by comparing the actual labels(values) with the labels predicted by the model.
    Working Method: Actual values, Predicted values, Average calculation
    """
    true = np.argmax(Y, axis=1)
    mdl_prediction = np.argmax(predictions, axis=1)
    return (np.mean(mdl_prediction == true))

def cross_entropy_loss(Y, predictions):
    """
    It is a loss function that measures the difference between the values predicted by the model and the actual values. 
    It also evaluates how poorly or well the model performs. The lower the loss value, the better the model's performance.
    """
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    return (-np.mean(np.sum(Y * np.log(predictions), axis=1)))
