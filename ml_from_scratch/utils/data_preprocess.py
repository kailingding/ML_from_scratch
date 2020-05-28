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

def z_standardize(arr, mean=True, std=True):
    """
        Z-score Standardization.
        Standardize the feature matrix, and store the standarize rule.
    """
    _arr = np.copy(arr)
    arr_mean = 0
    arr_std = 1
    if mean:
        arr_mean = np.mean(_arr)
    if std:
        arr_std = np.std(_arr)
    standardized_arr = (_arr - arr_mean) / arr_std
    return standardized_arr
    #
    # toreturn = X_inp.copy()
    # for i in range(X_inp.shape[1]):
    #     std = np.std(X_inp[:, i])  # ------ Find the standard deviation of the feature
    #     mean = np.mean(X_inp[:, i])  # ------ Find the mean value of the feature
    #     temp = []
    #     for j in np.array(X_inp[:, i]):
    #         temp += [(j - mean) / std]
    #     toreturn[:, i] = temp
    # return toreturn


def sigmoid(x):
    """
        Sigmoid Function
    """
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    print(1)
    print(__package__)
