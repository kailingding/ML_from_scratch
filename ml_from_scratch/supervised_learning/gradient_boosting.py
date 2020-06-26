import numpy as np
from decision_tree import DecisionTree


class GradientBoosting:
    def __init__(self):
        pass

    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            # compute gradient
            gradient = self.loss.gradient(y, y_pred)
            # train DT
            self.trees[i].fit(X, gradient)
            # predict
            update = self.trees[i].predict(X)
            # update y_pred with learning rate
            y_pred -= np.multiply(self.learning_rate, update)

    def predict(self, X):
        y_pred = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update
