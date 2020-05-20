import numpy as np
import pandas as pd


class Naive_Bayes():
    """

    Naive Bayes classifer

    Attributes:
        prior: P(Y)
        likelihood: P(X_j | Y)
    """

    def __init__(self):
        """
            Initialization
        """
        self.classes = []

    def fit(self, X_train, y_train):
        """
            The fit function fits the Naive Bayes model based on the training data.
            Here, we assume that all the features are **discrete** features.

            X_train is a matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.

            y_train contains the corresponding labels. 
        """

        # calculate prior
        self.prior = dict()
        separated = [[x for x, t in zip(X_train, y_train) if t == c] for c in np.unique(y_train)]
        for x, c in zip(separated, np.unique(y_train)):
            self.classes.append(c)
            self.prior[f'Y = {c}'] = np.log(len(x) / X_train.shape[0])

        # calculate likelihood
        self.likelihood = dict()
        for x, c in zip(separated, np.unique(y_train)):
            # laplace smoothing
            count = np.array(x).sum(axis=0) + 1
            for i, cnt in enumerate(count):
                self.likelihood[f'X{i} | Y = {c}'] = np.log((cnt) / np.array(count).T.sum(axis=0))

    def ind_predict(self, x: list):
        """
            Predict the most likely class label of one test instance based on its feature vector x.
        """
        posterior = []
        for _key, prior_prob in self.prior.items():
            prob = prior_prob + np.array([self.likelihood[f'X{j} | {_key}'] * x[j]
                                          for j in range(len(x))])[np.newaxis:].sum(axis=0)
            # append prob to posterior list
            posterior.append(prob)
        # pick label with highest probabilty
        return self.classes[np.argmax(posterior)]

    def predict(self, X):
        """
            X is a matrix or 2-D numpy array, represnting testing instances.
            Each testing instance is a feature vector.

            Return the predictions of all instances in a list.
        """
        ret = [self.ind_predict(x) for x in X]
        return ret


if __name__ == "__main__":
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
    col = ['class_name', 'left_weight', 'left_distance',
           'right_weight', 'right_distance']
    data = pd.read_csv(url, delimiter=',', names=col)

    X = np.array(data.iloc[:, 1:])
    y = data.class_name
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=88)
    clf = Naive_Bayes()
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
