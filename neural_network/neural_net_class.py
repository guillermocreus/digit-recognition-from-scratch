import numpy as np
import sys
sys.path.insert(0, '../Import_data')

# ______IMPORT DATA_________

from train_test_data import all_train_test_data
data_train, data_test, label_train, label_v, label_test = all_train_test_data()
M = len(data_train)

# __________________________


# __________ FUNCIONES ______________

factor_hidden = 1e-3
factor_start = 1e-4
factor_softmax = 1e-2


def relu_hidden(x):
    var = x * factor_hidden
    if (var > 0):
        return var
    else:
        return var * 1e-3


def relu_start(x):
    var = x * factor_start
    if (var > 0):
        return var
    else:
        return var * 1e-3


def relu_hidden_d(y):
    if (y > 0):
        return factor_hidden
    else:
        return factor_hidden * 1e-3


def relu_start_d(y):
    if (y > 0):
        return factor_start
    else:
        return factor_start * 1e-3


def square(x):
    return x * x


def divide(x):
    return 1 / x


def exponencial(x):
    var = x * factor_softmax
    if (var > 20):
        return np.exp(20)
    elif (var < -20):
        return np.exp(-20)
    return np.exp(var)


# _______________________________________


# _____________ VECTORIZAR ______________


relu_start_v = np.vectorize(relu_start)
relu_start_d_v = np.vectorize(relu_start_d)

relu_start_v = np.vectorize(relu_start)
relu_sart_d_v = np.vectorize(relu_start_d)

relu_hidden_v = np.vectorize(relu_hidden)
relu_hidden_d_v = np.vectorize(relu_hidden_d)

square_v = np.vectorize(square)
divide_v = np.vectorize(divide)
exp_v = np.vectorize(exponencial)

# _______________________________________


# _______________ SOFTMAX _______________

def softmax(a):
    aux = exp_v(a)
    suma = aux.dot(np.ones(10))
    Z = (aux.T * divide_v(suma)).T
    return Z

# _______________________________________


class NeuralNet:
    def __init__(self, L, layer_dimensions, activation_functions,
                 derivation_functions):  # size(layer_dimensions) == L+1
        self.layers = np.empty((L,), dtype=object)  # data + L layers
        self.L = L  # number of layers
        self.X = np.empty((L,), dtype=object)
        self.A = np.empty((L,), dtype=object)
        self.Ekj = np.empty((layer_dimensions[0], layer_dimensions[-1]))
        for l in range(L):
            self.layers[l] = Layer(layer_dimensions[l:l+2],
                                   activation_functions[l])  # slice

    def print_precision(self, data_train, label_train, data_test, label_test):
        self.feed_forward(data_test)
        Z_gorro = self.X[-1]
        aciertos = 0
        for k in range(len(Z_gorro)):
            if (int(np.argmax(Z_gorro[k])) == int(label_test[k])):
                aciertos += 1

        porcentaje_aciertos = 100 * aciertos / len(data_test)
        print("El porcentaje de aciertos del TEST es del: ",
              str(porcentaje_aciertos), "%\n")

        self.feed_forward(data_train)
        Z_gorro = self.X[-1]
        aciertos = 0
        for k in range(len(Z_gorro)):
            if (int(np.argmax(Z_gorro[k])) == int(label_train[k])):
                aciertos += 1

        porcentaje_aciertos = 100 * aciertos / len(data_train)
        print("El porcentaje de aciertos del TRAIN es del: ",
              str(porcentaje_aciertos), "%\n")

    def obtain_Ekj(self):
        self.Ekj = -np.multiply(label_v, np.log(self.X[-1]))

    def calculate_error(self):
        self.obtain_Ekj()
        Ek = self.Ekj.dot(np.ones(10))
        return float(np.ones(M).dot(Ek))

    def feed_forward(self, data):
        L = self.L
        self.A[0] = data_train.dot(self.layers[0].weights)
        self.X[0] = self.layers[0].activation_function(self.A[0])
        for l in range(1, L):
            self.A[l] = self.X[l-1].dot(self.layers[l].weights)
            if (l < L-1):
                self.X[l] = self.layers[l].activation_function(self.A[l])
            else:
                self.X[l] = softmax(self.A[l])

    def obtain_deltas(self):
        L = self.L
        delta = np.empty((L,), dtype=object)
        for l in range(L):
            if (l < L-1):
                next_layer = self.layers[l+1]
                current_layer = self.layers[l]
                suma = next_layer.delta.dot(next_layer.weights.T)
                suma = suma.dot(np.ones(next_layer.J))
                delta[l] = np.multiply
                (current_layer.activation_function(self.A[l]), suma)

            else:
                current_layer = self.layers[l]
                delta[l] = (self.X[l] - label_v) * factor_softmax

    def back_propagate(self):
        self.obtain_deltas()
        self.obtain_grad()

    def update_weights(self, learning_rate):
        L = self.L
        for l in range(L):
            layer = self.layers[l]
            layer.weights -= learning_rate * layer.grad

    def gradient_descent(self, learning_rate, nIter, data_train,
                         label_train, data_test, label_test):
        cont = 0
        self.feed_forward(data_train)
        while (cont < nIter):
            cont += 1
            self.back_propagate()
            self.update_weights(learning_rate)
            self.feed_forward(data_train)

            if (cont % 50 == 0):
                self.print_precision(data_train, data_test)
                self.calculate_error()

    def obtain_grad(self):
        L = self.L
        for l in range(L):
            delta = self.layers[l].delta
            if (l > 0):
                self.layers[l].grad = self.X[l-1].T.dot(delta)
            else:
                self.layers[l].grad = data_train.T.dot(delta)


class Layer:
    def __init__(self, dimensions, activation_function):
        self.I = dimensions[0]
        self.J = dimensions[1]
        correction = np.sqrt(2 / dimensions[0])
        self.weights = correction * np.random.rand(self.I, self.J)
        self.activation_function = activation_function
        self.grad = np.empty((self.I, self.J))
        self.delta = np.empty((M, self.J))


def main():
    L = 4
    layer_dimensions = (L + 1)*[20]
    layer_dimensions[0] = 28 * 28 + 1
    layer_dimensions[-1] = 10

    activation_functions = L*[relu_hidden_v]
    activation_functions[0] = relu_start_v
    activation_functions[-1] = softmax

    derivation_functions = L*[relu_hidden_d_v]
    derivation_functions[0] = relu_start_d_v
    derivation_functions[-1] = softmax

    NeuralNetwork = NeuralNet(L, layer_dimensions,
                              activation_functions, derivation_functions)
    NeuralNetwork.gradient_descent(0.1, 4000, data_train, label_train,
                                   data_test, label_test)


main()
