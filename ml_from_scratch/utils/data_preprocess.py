import numpy as np


def min_max_normalize(lst):
    """
        Normalized list x, in range [0, 1]
    """
    maximum = max(lst)
    minimum = min(lst)
    toreturn = []
    for i in range(len(lst)):
        toreturn.append((lst[i] - minimum) / (maximum - minimum))
    return toreturn

def z_standardize(X_inp):
    """
        Z-score Standardization.
        Standardize the feature matrix, and store the standarize rule.
    """
    toreturn = X_inp.copy()
    for i in range(X_inp.shape[1]):
        std = np.std(X_inp[:, i])  # ------ Find the standard deviation of the feature
        mean = np.mean(X_inp[:, i])  # ------ Find the mean value of the feature
        temp = []
        for j in np.array(X_inp[:, i]):
            temp += [(j - mean) / std]
        toreturn[:, i] = temp
    return toreturn


def sigmoid(x):
    """
        Sigmoid Function
    """
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    print(1)
    print(__package__)
