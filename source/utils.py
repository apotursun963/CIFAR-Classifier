
import numpy as np

def relu(x):
    return (np.maximum(0, x))

def relu_derivative(x):
    return (np.where(x > 0, 1, 0))

def softmax(x):
    exps = np.exp(x - np.max(x))
    return (exps / np.sum(exps, axis=1, keepdims=True))

def accuracy(Y, predictions):
    true = np.argmax(Y, axis=1)
    mdl_prediction = np.argmax(predictions, axis=1)
    return (np.mean(mdl_prediction == true))

def cross_entropy_loss(Y, predictions):
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    return (-np.mean(np.sum(Y * np.log(predictions), axis=1)))
