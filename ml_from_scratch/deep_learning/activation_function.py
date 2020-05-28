import numpy as np

# TODO:
#   1. gradient


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.array([elem if elem > 0 else 0 for elem in x])


def leakyReLU(x, alpha=0.001):
    return np.array([elem if elem > 0 else elem * alpha for elem in x])


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
