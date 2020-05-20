import numpy as np


class Linear_Regression():
    def __init__(self, alpha=1e-10, num_iter=10000, early_stop=1e-50, intercept=True, init_weight=None):
        """
        attributes:
            alpha: Learning Rate, default 1e-10
            num_iter: Number of Iterations to update coefficient with training data
            early_stop: Constant control early_stop.
            intercept: Bool, If we are going to fit a intercept, default True.
            init_weight: Matrix (n x 1)
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight

    def fit(self, X_train, y_train):
        """
            Save the datasets in the model, and perform gradient descent.

            Parameter:
                X_train: Matrix or 2-D array. Input feature matrix.
                Y_train: Matrix or 2-D array. Input target value.

        """

        self.X = np.mat(X_train)
        self.y = np.mat(y_train)

        if self.intercept:
            # add constant ones for bias weight
            ones = np.ones(len(self.X)).reshape(-1, 1)
            self.X = np.hstack([ones, self.X])

        # initialize weight with uniform from [-1, 1]
        self.coef = np.random.uniform(-1, 1, self.X.shape[1])
        self.gradient_descent()  # call gradient descent to finddd the best params

    def gradient(self):
        """
            Helper function to calculate the gradient respect to coefficient.
        """
        y_pred = self.X.dot(self.coef)
        self.grad_coef = np.array(-(self.y - y_pred).dot(self.X)).flatten()

    def gradient_descent(self):
        """
            Training function
        """

        self.loss = []

        for i in range(self.num_iter):

            self.gradient()  # update gradient

            previous_y_hat = self.X.dot(self.coef)  # previous y_hat

            temp_coef = self.coef - self.alpha * self.grad_coef  # currenty_hat

            pre_error = np.mean(0.5 * np.square(self.y - previous_y_hat))  # previous error

            current_error = np.mean(0.5 * np.square(self.y - self.X.dot(temp_coef)))  # current error

            # This is the early stop, don't modify fllowing three lines.
            if (abs(pre_error - current_error) < self.early_stop) | (
                    abs(abs(pre_error - current_error) / pre_error) < self.early_stop):
                self.coef = temp_coef
                return self

            if current_error <= pre_error:
                # increase learning rate if the error diff is positive
                self.alpha *= 1.3
                self.coef = temp_coef
            else:
                # decrease otherwise
                self.alpha *= 0.9

            self.loss.append(current_error)

            if i % 10000 == 0:
                print('Iteration: ' + str(i))
                print('Coef: ' + str(self.coef))
                print('Loss: ' + str(current_error))
        return self

    def predict(self, X):
        """
            X is a matrix or 2-D numpy array, represnting testing instances.
            Each testing instance is a feature vector.

            Parameter:
            X: Matrix, array or list. Input feature point.

            Return:
                ret: prediction of given data matrix
        """
        X = np.mat(X)
        if self.intercept:
            ones = np.ones(X.shape[0]).reshape(-1, 1)
            X = np.hstack([ones, X])
        ret = [np.array(x).dot(self.coef)[0] for x in X]
        return ret


if __name__ == '__main__':
    X = np.array(np.mat(np.arange(1, 1000, 5)).T)
    y = np.array((30 * X)).flatten() + 20
    linear_reg = Linear_Regression(alpha=1, num_iter=100000, init_weight=np.mat([15, 25]).T)
    linear_reg.fit(X, y)
    print(linear_reg.predict(X))
