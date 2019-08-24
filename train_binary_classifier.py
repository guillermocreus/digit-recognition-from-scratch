import numpy as np
import time

N0 = 28 * 28 + 1  # dimension layer 0
M = 0;  # Se actualiza posteriormente

def sigmoid(x):
    return 1.0/(1 + np.exp(-x/30))    


def sigmoid_d(y):
    return y*(1-y)*1/30


def square(x):
    return x*x


sigmoid_v = np.vectorize(sigmoid)
sigmoid_d_v = np.vectorize(sigmoid_d)
square_v = np.vectorize(square)


def rel_error(new_error, old_error):
    return abs(new_error - old_error) / new_error


def obtain_y(X, weights):
    return sigmoid_v(X.dot(weights))


def obtain_Ek(X, Y, separador, d0, d1):  # Ek Matriz, asumiendo label vector fila de dimension M
    M = len(X)
    Ek = np.zeros(M)
    for k in range(M):
        zk = 0
        if (k >= separador):
            zk = 1
        Ek[k] = (Y[k] - zk)
    return Ek


def calculate_error(Ek, X):
    M = len(X)
    aux = square_v(Ek)
    return 0.5 * float(np.ones(M).dot(aux))


def grad(X, Y, Ek):
    M = len(X)
    grad_weights = np.zeros(N0)
    Y_d = sigmoid_d_v(Y)
    for i in range(N0):
        aux = 0
        for k in range(M):
            aux += Ek[k]*X[k][i]*Y_d[k]
        grad_weights[i] = aux
    return grad_weights


# Entrena el clasificador binario de labels d0, d1 y devuelve los pesos 
# Supongo que el output correcto es d1 --> output = 1
def train_classifier(X, separador, d0, d1):
    M = len(X)
    weights = np.random.rand(N0)  # [i, t]
    Y = obtain_y(X, weights)
    Ek = obtain_Ek(X, Y, separador, d0, d1)
    # eps = 1e-5
    n_iteraciones = 500
    cont = 0
    learning_rate = 0.01
    old_error = np.inf
    new_error = calculate_error(Ek, X)

    while (cont < n_iteraciones):
        cont += 1
        start = time.time()
        weights -= learning_rate * grad(X, Y, Ek)
        Y = obtain_y(X, weights)
        Ek = obtain_Ek(X, Y, separador, d0, d1)
        old_error = new_error
        new_error = calculate_error(Ek, X)
        end = time.time()
        print(old_error, new_error)
        print('Training ' + d0 + ' vs ' d1 + ': -->', 'rel Error = ' + str(rel_error(new_error, old_error)) + ',', 'elapsed time = '+str(end-start)+',', cont, '\n')
        print(Y)
        
    return weights

def test_data(binary_nets, foto):
    v = 10 * [0]
    for digits in binary_nets:
        result = float(sigmoid(weights.dot(foto)))
        if result >= .5:
            v[digits[1]] += 1
        else:
            v[digits[0]] += 1
    return np.argmax(v)
