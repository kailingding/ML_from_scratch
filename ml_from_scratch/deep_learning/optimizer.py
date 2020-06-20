import numpy


class SGD:
	'''
		SGD pytorch code snippet:
			train_loader = DataLoader(dataset=dataset, batch_size=1)
			for epoch in range(n_epochs):
				for x_batch, y_batch in train_loader:
					train(...)
					validate(...)

		Mini-batch GD:
			# X.shape = (100, 4)
			train_loader = DataLoader(dataset=dataset, batch_size=5) # 5 instances
			for epoch in range(n_epochs):
				for x_batch, y_batch in train_loader:
					train(...)
					validate(...)

	'''
    def __init__(self, lr=0.01, momentum=0):
        self.lr = lr
        self.momentum = momentum
        self.updated_w = None

    def step(self, params, grad_wrt_w):
        if self.updated_w is None:
            self.updated_w = np.zeros(np.shape(params))

        # use momentun if set
        self.updated_w = self.momentum * self.updated_w + (1 - self.momentum) * grad_wrt_w

        # gradient descent
        return params - self.lr * self.updated_w



class Adagrad:
	def __init__(self):
		pass

	def step(self):
		pass


class Adam:
	def __init__(self):
		pass

	def step(self):
		pass
