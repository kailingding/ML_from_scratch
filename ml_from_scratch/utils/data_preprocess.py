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


def z_standardize(lst, mean=True, std=True):
    """
        Z-score Standardization.
    """
    mean = 0
    std = 1
    if mean:
        mean = np.mean(lst)
    if std:
        std = np.std(lst)

    return [(x - mean) / std for x in lst]


def sigmoid(x):
    """ 
        Sigmoid Function
    """
    return 1 / (1 + np.exp(-x))
