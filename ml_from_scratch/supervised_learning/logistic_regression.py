import numpy as np
import pandas as pd

from ml_from_scratch.utils.data_preprocess import sigmoid, z_standardize

class Logistic_Regression:
    def __init__(self, alpha, num_iter, early_stop=0, standardized=True):
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.standardized = standardized
        self.theta, self.b = 0, 0
        self.X, self.y = None, Nbone

    def fit(self, X_train, y_train):
        """
            Save the datasets in our model, and do normalization to y_train

            Parameter:
                X_train: Matrix or 2-D array. Input feature matrix.
                Y_train: Matrix or 2-D array. Input target value.
        """

        self.X = X_train  # features
        self.y = y_train  # label

        count = 0
        uni = np.unique(y_train)
        for y in y_train:
            if y == min(uni):
                self.y[count] = -1
            else:
                self.y[count] = 1
            count += 1

        self.n, self.m = X_train.shape
        self.theta = np.random.normal(-1 / self.m ** .5, 1 / self.m ** .5, size=self.m)
        self.b = 0

        # train model (find best coef)
        _, _ = self.gradient_descent_logistic(self.alpha, self.num_iter,
                self.early_stop, self.standardized)

    def gradient(self, X_inp, y_inp, theta, b):
        """
            Calculate the grandient of Weight and Bias, given sigmoid_yhat, true label, and data

            Parameter:
                X_inp: Matrix or 2-D array. Input feature matrix.
                y_inp: Matrix or 2-D array. Input target vpalue.
                theta: Matrix or 1-D array. Weight matrix.
                b: int. Bias.

            Return:
                grad_theta: gradient with respect to theta
                grad_b: gradient with respect to b
        """
        # update gradient
        y_pred = sigmoid(np.dot(X_inp, theta) + b)

        grad_b = (-1 / self.n) * (y_inp - y_pred).mean()
        grad_theta = (-1 / self.n) * np.dot(X_inp.T, (y_inp - y_pred))

        return grad_theta, grad_b

    def gradient_descent_logistic(
            self, alpha, num_pass, early_stop=0, standardized=True
    ):
        """
            Logistic Regression with gradient descent method

            Parameter:
                alpha: (Hyper Parameter) Learning rate.
                num_pass: Number of iteration
                early_stop: (Hyper Parameter) Least improvement error allowed before stop.
                            If improvement is less than the given value, then terminate the function
                            and store the coefficents.
                            default = 0.
                standardized: bool, determine if we standardize the feature matrix.

            Return:
                self.theta: theta after training
                self.b: b after training
        """

        if standardized:
            self.X = np.array([z_standardize(x) for x in self.X.T]).T
            self.standardized = True

        for i in range(num_pass):
            # update gradients
            grad_theta, grad_b = self.gradient(self.X, self.y, self.theta, self.b)
            temp_theta = self.theta - alpha * grad_theta
            temp_b = self.b - alpha * grad_b

            # early stopping
            previous_y_hat = sigmoid(np.dot(self.X, self.theta) + self.b)
            temp_y_hat = sigmoid(np.dot(self.X, temp_theta) + temp_b)
            pre_error = -(np.dot(self.y, np.log(previous_y_hat)) +
                          np.dot((1 - self.y), np.log(previous_y_hat)))
            temp_error = -(np.dot(self.y, np.log(temp_y_hat)) +
                           np.dot((1 - self.y), np.log(temp_y_hat)))

            if (abs(pre_error - temp_error) < early_stop) | (
                    abs(abs(pre_error - temp_error) / pre_error) < early_stop
            ):
                return temp_theta, temp_b

            self.theta = temp_theta
            self.b = temp_b

        return self.theta, self.b

    def predict_ind(self, x: list):
        # calculate probability (you can use the sigmoid function)
        p = sigmoid(np.dot(x, self.theta) + self.b)

        return p

    def predict(self, X):
        """
            Parameter:
            x: Matrix, array or list. Input feature point.

            Return:
                p: prediction of given data matrix
        """
        # Use predict_ind to generate the prediction list
        if self.standardized:
            X = np.array([z_standardize(x) for x in X.T]).T

        ret = [self.predict_ind(x) for x in X]

        return ret


if __name__ == "__main__":
    url_Wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # names = ['f_acid', 'v_acid', 'c_acid', 'sugar', 'chlorides', 'f_SO2', 't_SO2', 'density', 'ph',
    #           'sulphates', 'alcohol', 'quality']
    wine = pd.read_csv(url_Wine, delimiter=";")

    wine5 = wine.loc[wine.quality == 5]
    wine6 = wine.loc[wine.quality == 6]
    wineall = pd.concat([wine5, wine6])
    X = np.array(wineall.iloc[:, :10])
    Y = np.array(wineall.quality)

    count1 = 0
    for y in Y:
        if y == 5:
            Y[count1] = -1
        else:
            Y[count1] = 1
        count1 += 1

    logit = Logistic_Regression()
    logit.fit(X, Y)

    g = logit.gradient_descent_logistic(0.01, 10000)
    w, b = g

    hat = logit.predict(X)

    count = 0
    for i in range(len(hat)):
        if hat[i] < 0.5:
            if Y[i] == -1:
                count += 1
        else:
            if Y[i] == 1:
                count += 1
    print(count)
