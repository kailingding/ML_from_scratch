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
    def __init__(self, input_channel, output_channel):
        pass

    def forward_pass(self, X):
        pass

    def backward_pass(self):
        pass


class BatchNorm2d(Layer):
    def __init__(self, input_channel, output_channel):
        pass

    def forward_pass(self, X):
        pass

    def backward_pass(self):
        pass
