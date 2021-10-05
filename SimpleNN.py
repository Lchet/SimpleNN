# -*- coding:utf-8 -*-
# @Filename:    SimpleNN.py
# Created on:   30/09/21 13:02
# @Author:      Luc


# We will first build a neural network with only one input layer (consisting of three neurons) and
# one output layer.
# 1. Construct a function returning a sigmoid function. (Reminder: s (x) = 1 )
# 1+e−x
# 2. Construct a function returning the derivative of a sigmoid function.
# 3. Build an array of three weights (3x1 array – think why these dimensions!) and initialize
# their value randomly. (It is good practice to use weights with mean = 0)
# 4. Create a loop, iterating 1000 times (equal to the desired number of learning steps).
# For each iteration, calculate the difference between the network prediction and the real
# value of y.
# Multiply that difference with the sigmoid derivative - use the dot product of this number
# with the input layer in order to update your weights for the next iteration.
# 5. Run this network, using the following input and output data sets:
# Input dataset (4x3 matrix):
# [ [ 0 , 0 , 1 ],
# [ 0 , 1 , 1 ],
# [ 1 , 0 , 1 ],
# [ 1 , 1 , 1 ] ]
# Output dataset (4x1 matrix):
# [[ 0 ],
# [ 0 ],
# [ 1 ],
# [ 1 ]]
# 6. For a slightly harder problem: We will add another layer to the network - a hidden layer.
# The hidden layer will also have three neurons in it. The output layer will still have only
# one neuron. When using a “for loop” in order to update the weights of the network, make
# sure to update the weights of both layers.
# 7. Run the new network on these new input and output data sets:
# Input d
import matplotlib.pyplot as plt
import numpy as np



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_tag(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)


def loss(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()


class SimpleNN():
    """
    This class implement a simple Neural Network with the same activation function
    across all the layers
    """

    def __init__(self, layers=[3, 1], l_r=0.1, iterations=1000, activation_f=sigmoid):
        """
        :param layers: len(layers) specifies the number of layer and each item represents number of
                        neurons in the index layer.
        :param l_r: learning rate
        :param iterations:
        """
        self.l_r = l_r
        self.layers = layers
        self.n_layers = len(self.layers)
        self.epochs = iterations
        self.sigma = activation_f
        self.costs = []
        self.sample_size = None
        self.weights = []
        self.activations = []
        self.biases = []
        self._init_weights_and_biases()

    def fit(self, X, y):

        for i in range(self.epochs):
            y_hat = self._forward_propagation(X.T)
            self.costs.append(loss(y.ravel(), y_hat.ravel()))
            self.gradient_descent(X, y)
            # print(f"Epoch {i} Loss: {self.costs[i]}\n Epoch {i+1} complete\n")



    def predict(self, X):
        return self._forward_propagation(X.T)

    def _init_weights_and_biases(self):
        self.biases = [np.random.normal(size=(n_bias_in_layer, 1))
                       for n_bias_in_layer in self.layers[1:]]
        self.weights = [np.random.normal(size=(rows, cols))
                        for cols, rows in zip(self.layers[:-1], self.layers[1:])]

    def _forward_propagation(self, x):
        """
        feadd forward for one sample of the data set and return output of the network.
        :param x:
        :return:    Prediction - output of the network
        """
        # x is a single sample of shape(n,1)
        for b, W in zip(self.biases, self.weights):
            x = sigmoid((W@x) + b)
        return x

    def _back_propagation(self, x, y):
        """
        run forward for each a single sample x to get output and then
        do back propagation and return a grad_b, grad_w: gradients for the cost function.
        grad_b and grad_w are lists of reals and matrices respectively

        :param x: a sample of the training set
        :param y:  label for x
        :return:
        """

        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        # forward propagation
        # store activations of each layer in list
        activation = np.expand_dims(x, axis=1)
        activations = [activation]
        for b, W in zip(self.biases, self.weights):
            activation = sigmoid((W @ activation) + b)
            activations.append(activation)

        # backward pass
        cost_derivative = activations[-1] - y
        diagonal = sigmoid_tag(activation[-1])
        delta_L = cost_derivative * diagonal
        grad_b[-1] = delta_L
        grad_w[-1] = delta_L @ activations[-2].T

        # loop backward by using index -l starting from the second element from the end
        # because we have already computed layer L
        delta_l = delta_L
        for l in range(2, self.n_layers):
            diagonal = sigmoid_tag(activations[-l])
            # the * is an element wise multiplication with the sigma derivative (sigma_tag)
            # this vector can also be put in a diagonal matrix
            delta_l = self.weights[-l + 1].T @ delta_l * diagonal
            grad_b[-l] = delta_l
            grad_w[-l] = delta_l @ activations[-l - 1].T

        return grad_b, grad_w

    def gradient_descent(self, X, y):
        """
        gradient descent of the network.
        :param X:   training data
        :param y: labels
        :return:
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for sample, label in zip(X, y):
            delta_b, delta_w = self._back_propagation(sample, label)
            grad_b = [g_b + d_b for g_b, d_b in zip(grad_b, delta_b)]
            grad_w = [g_w + d_w for g_w, d_w in zip(grad_w, delta_w)]
        # updates weights
        self.weights = [w - (self.l_r / len(X)) * g_w for w, g_w in zip(self.weights, grad_w)]
        self.biases = [b - (self.l_r / len(X)) * g_b for b, g_b in zip(self.biases, grad_b)]



def exercise_1():
    X = np.array(([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]))
    y = np.array(([[0], [0], [1], [1]]))
    nn = SimpleNN(layers=[3, 1])
    nn.fit(X, y)
    y_hat = nn.predict(X)
    print(f"Loss for exercise 1: {loss(y, y_hat)}")
    plt.plot(range(nn.epochs), nn.costs)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Exercise 1")
    plt.show(block=True)


def exercise_2():
    X = np.array(([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]))
    y = np.array(([[0], [1], [1], [0]]))
    nn = SimpleNN(layers=[3, 3, 1])
    nn.fit(X, y)
    y_hat = nn.predict(X)
    print(f"Loss for exercise 2: {loss(y, y_hat)}")
    plt.plot(range(nn.epochs), nn.costs)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Exercise 1")
    plt.show(block=True)


if __name__ == '__main__':
    exercise_1()
    exercise_2()
    pass
