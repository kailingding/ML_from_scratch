import numpy as np


def mse(y, yhat):
    return np.mean(np.square(y - yhat))


def rmse(y, yhat):
    return np.sqrt(np.mean(np.square(y - yhat)))


def mae(y, yhat):
    return np.mean(np.abs(y - yhat))


def hinge_loss(y, yhat):
    return np.max(0, 1 - yhat * y)


def cross_entropy(y, yhat):
    return -y * np.log(yhat) + (1 - y) * np.log(yhat)
