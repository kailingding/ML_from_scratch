import numpy as np
import copy


class Layer(object):
    def __init__(self):
        pass

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass


class Linear(Layer):
    def __init__(self, input_channel, output_channel, optimizer, lr=0.01):
        self.lr = lr
        # Xavier weight Initialization
        self.w = np.random.normal(loc=0., scale=np.sqrt(2 / (
            input_channel + output_channel)), size=(input_channel, output_channel))
        self.b = np.zeros(output_channel)
        self.layer_input = None
        self.w_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

    def forward_pass(self, X):
        self.layer_input = X
        return X.dot(self.w) + self.b

    def backward_pass(self, grads_output):

        # compute gradients w.r.t layer weights and biases
        new_grad_w = np.dot(self.X.T, grads_output)
        new_grad_b = np.sum(grads_output, axis=0, keepdims=True)

        # update weights with optimizer
        self.w = self.w_optimizer.step(self.w, new_grad_w)
        self.b = self.b_optimizer.step(self.b, new_grad_b)

        # Return accumulated gradient for next layer
        # chain rule: df/dx = df/dy * dy/dx
        accum_grad = np.dot(self.w.T, grads_output)

        return accum_grad


class Conv2d(Layer):
    def __init__(self, input_channel, output_channel):
        pass

    def forward_pass(self, X):
        pass

    def backward_pass(self):
        pass


class RNN(Layer):
    def __init__(self):
        pass

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass


class LSTM(Layer):
    def __init__(self):
        pass

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass


class BatchNorm1d(Layer):
    def __init__(self, optimizer, momentum=0.1, eps=1e-6):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.training = True
        self.running_mean, self.running_var = None, None

        # initialize gamma and beta
        self.gamma, self.beta = None, None

        # initialize optimizer
        self.gamma_optimizer = copy.copy(optimizer)
        self.beta_optimizer = copy.copy(optimizer)


    def forward_pass(self, X, training=True):
        if not self.running_mean or not self.running_var:
            self.running_mean = np.mean(X)
            self.running_var = np.std(X)

        # batch norm algo:
        #   1) normalize the batch, x_norm = (x - x_mean) / x_std
        #   2) create output X with gamma and beta (should 
        #       become empirical mean and std after training)
        if self.training:
            mean = np.mean(X)
            var = np.var(X)
            self.running_mean = self.momentum * self.running_mean + (
                1 -  self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (
                1 -  self.momentum) * self.var
        else:
            mean = self.running_mean
            var = self.running_var

        X_norm = (X - mean) / np.sqrt(var + self.eps)
        output_X = self.gamma * X_norm + self.beta

        return output_X

    def backward_pass(self, accum_grad):
        if self.training:
            grad_gamma = np.sum(accum_grad * X_norm, axis=0)
            grad_beta = np.sum(accum_grad, axis=0)

            self.gamma = self.gamma_optimizer.step(self.gamma, grad_gamma)
            self.beta = self.beta_optimizer.step(self.beta, grad_beta)

        # TODO
        return accum_grad


class BatchNorm2d(Layer):
    def __init__(self, input_channel, output_channel):
        pass

    def forward_pass(self, X):
        pass

    def backward_pass(self):
        pass
