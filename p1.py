import numpy as np
from math import exp


def sigmoid(x):
    return 1.0/(1+exp(-x))


def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_d(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_d(self.output), self.weights2.T) * sigmoid_d(self.layer1)))


        self.weights1 += d_weights1
        self.weights2 += d_weights2
