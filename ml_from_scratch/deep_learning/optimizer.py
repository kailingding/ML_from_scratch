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

    def step(self, w, grad_wrt_w):
        if self.updated_w is None:
            self.updated_w = np.zeros(np.shape(w))

        # use momentun if set
        self.updated_w = self.momentum * self.updated_w + (1 - self.momentum) * grad_wrt_w

        # gradient descent
        return w - self.lr * self.updated_w



class NesterovAcceleratedGradient:
	def __init__(self, lr=0.01, momentum):
		pass

	def step(self):
		pass



class Adagrad:
	def __init__(self, lr=0.01):
		self.lr = lr
		self.grad_cache = None # sum of squares of gradients
		self.epsilon = 1e-7

	def step(self, w, grad_wrt_w):
		# if not initialized
		if not self.grad_cache:
			self.grad_cache = np.zeros(np.shape(w))

		# Add the square of the gradient of the loss function at w
		self.grad_cache += np.square(grad_wrt_w)

		# Adaptive gradient with higher learning rate for sparse data
		w -= self.lr * grad_wrt_w / (np.sqrt(self.grad_cache) + self.epsilon)
		return w


class RMSProp:
	def __init__(self, lr=0.01, rho=0.99):
		self.lr = lr
		self.grad_cache = None # sum of squares of gradients
		self.epsilon = 1e-7
		self.rho = rho

	def step(self):
		# if not initialized
		if not self.grad_cache:
			self.grad_cache = np.zeros(np.shape(w))

		# decay with only a small portion of sum of squares of gradients
		self.grad_cache =  self.rho * self.grad_cache + (1 - self.rho) * np.square(grad_wrt_w)

		# Adaptive gradient with higher learning rate for sparse data
		w -= self.lr * grad_wrt_w / (np.sqrt(self.grad_cache) + self.epsilon)

		return w

class Adam:
	def __init__(self, lr=0.01, momentum=0, rho=0.99):
		self.lr = lr
		self.beta1 = momentum
		self.beta2 = rho
		self.m, self.v = None, None
		self.epsilon = 1e-7  # smoothing factor


	def step(self, w, grad_wrt_w):
		if not self.m:
			self.m = np.zeros(np.shape(w))
			self.v = np.zeros(np.shape(w))
		# momentum-like step
		self.m = self.beta1 * grad_wrt_w + (1 - self.beta1) * grad_wrt_w
		# RMSProp-like step
		self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad_wrt_w)
		w = w - self.lr * self.m / (np.sqrt(self.v) + self.epsilon)
		return w
