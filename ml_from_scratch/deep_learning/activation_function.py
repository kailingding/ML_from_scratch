import numpy as np


class Sigmoid:
	def __call__(self, x):
		return 1 / (1 + np.exp(-x))

	def gradient(self, x):
		return self.__call__(x) * (1 - self.__call__(x))


class ReLU:
    def __call__(self, x):
        return np.array([elem if elem > 0 else 0 for elem in x])

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class LeakyReLU:
	def __init__(self, alpha=0.001):
		self.alpha = alpha

    def __call__(self, x):
        return np.array([elem if elem > 0 else elem * self.alpha for elem in x])

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)


class Tanh:
	def __call__(self, x):
		return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

	def gradient(self, x):
		return 1 - np.square(self.__call__(x))


class Softmax:
	def  __call__(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def gradient(self, x)
		return self.__call__(x) * (1 - self.__call__(x))
