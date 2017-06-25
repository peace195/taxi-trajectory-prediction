import random
import numpy as np
from Data import Data
import theano
from ClusterCoordinates import ClusterCoordinates
import theano.tensor as T
import math as M

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        cluster = ClusterCoordinates("cluster_coordinates.csv")
        coordinates = cluster.getCoordinates()
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        zero = np.ndarray(shape=(sizes[-1], 1))
        zero.fill(0.)
        self.biases[-1] = zero
        self.weghts = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.weghts[-1] = coordinates

    def feedforward(self, a):
        a = relu(np.dot(self.weghts[0], a) + self.biases[0])
        a = softmax(np.dot(self.weghts[1], a) + self.biases[1])
        a = np.dot(self.weghts[2], a) + self.biases[2]
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        n = len(training_data)
        for iter in xrange(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[k: k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch", iter
                self.evaluate(test_data)

    def update_mini_batch(self, mini_batch, eta):
        der_sum_b = [np.zeros(b.shape) for b in self.biases]
        der_sum_w = [np.zeros(w.shape) for w in self.weghts]
        for x, y in mini_batch:
            delta_b, delta_w = self.backpropagation(np.transpose(x), np.transpose(y))
            der_sum_w = [dsb + db for dsb, db in zip(der_sum_b, delta_b)]
            nabla_w = [dsw + dw for dsw, dw in zip(der_sum_w, delta_w)]
        self.biases = [b - (eta/len(mini_batch)*dsb) for b, dsb in zip(self.biases, der_sum_b)]
        self.weghts = [w - (eta/len(mini_batch)*dsw) for w, dsw in zip(self.weghts, der_sum_w)]
    def backpropagation(self, x, y):
        der_sum_b = [np.zeros(b.shape) for b in self.biases]
        der_sum_w = [np.zeros(w.shape) for w in self.weghts]
        activation = x
        activations = [x]
        zs = []
        # forwardpropagation
        type = 0

        for b, w in zip(self.biases, self.weghts):
            z = np.dot(w, activation) + b
            zs.append(z)
            if type == 0: activation = relu(z)
            else:
                if type == 1: activation = softmax(z)
                else:
                    activation = z
            type += 1
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y)
        type = 0
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            if type == 0: sp = softmax_prime(z)
            else:
                sp = z
            type += 1
            tmp = np.dot(np.transpose(self.weghts[-l+1]), delta)
            delta = multi(tmp, sp)
            der_sum_b[-l] = delta
            der_sum_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        return (der_sum_b, der_sum_w)

    def cost_derivative(self, output_activations, y):
        derivative = np.zeros(y.shape)
        derivative_x = erdist_der_lat(y[0], y[1], output_activations[0], output_activations[1])
        derivative_y = erdist_der_lon(y[0], y[1], output_activations[0], output_activations[1])
        derivative[0] = derivative_x
        derivative[1] = derivative_y
        return derivative

    def evaluate(self, test_data):
        for x in test_data:
            print self.feedforward(x)

def relu(z):
    for i in xrange(len(z)):
        if z[i] < 0:
            z[i] = 0
    return z

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0)

def softmax_prime(z):
    z1 = softmax(z)
    z2 = 1 - softmax(z)
    for i in xrange(len(z1)):
        z1[i] *= z2[i]
    return z1

def erdist(x, y, a, b):
    x = x * M.pi
    y = y * M.pi
    a = a * M.pi
    b = b * M.pi
    xx = (b - y) * M.cos((a + x) / 2)
    yy = (a - x)
    return M.sqrt(xx**2 + yy**2) * M.pi

def erdist_der_lat(x, y, a, b):
    xx = (y * M.pi - b * M.pi) * M.sin((a + x) * M.pi) / 2
    yy = 2 * (a - x) * M.pi
    return M.pi ** 2 * (xx + yy) * M.pi / (2 * erdist(x, y, a, b))
def erdist_der_lon(x, y, a, b):
    xx = (b - y) * M.pi
    yy = M.cos(((a + x) * M.pi) / 2) ** 2
    return M.pi ** 2 * (xx + yy) * M.pi / (2 * erdist(x, y, a, b))

def multi(x, y):
    z = np.zeros(x.shape)
    for i in xrange(len(x)):
        z[i] = x[i] * y[i]
    return z
def MLP():
    # Training Data
    data    = Data('../train.csv.zip', 'train.csv', 5, 10, 10, 10, 10, 10, 10, 10, 2500)
    outputs = data.outputs()
    inputs  = data.inputs()
    training_data = zip(inputs, outputs)
    # Test Data
    data_test = Data('../test.csv.zip', 'test.csv', 5, 10, 10, 10, 10, 10, 10, 10, 250)
    #outputs_test = data_test.outputs()
    inputs_test = data_test.inputs()
    test_data = inputs_test
    #Training Model
    net = Network([90, 500, 3388, 2])
    net.SGD(training_data, 1, 200, 0.1, test_data=test_data)

if __name__ == '__main__':
    MLP()