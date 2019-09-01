# Empieza a las 14:00 01/09/2019
import numpy as np
import time
from numpy import linalg as LA
import sys
sys.path.insert(0, '../Import_data')
from train_test_data import all_train_test_data



# ________ IMPORT DATA ______________

X, fotos_test, label_v, label_test = all_train_test_data()
M = len(X)

# ___________________________________


N0 = 28 * 28 + 1  # dimension layer 0
N1 = 20  # dimension layer 1


def sigmoid(x):
    return 1.0/(1 + np.exp(-x/30))
    if (x < -150): return 0
    elif (x > 150): return 1

def sigmoid_d(y):
    return y*(1-y) * (1/30)
    
def relu_mod(x):
    if (x > 0): return x/100
    else: return x/10000
    
def relu_d(y):
    if (y > 0): return 1/100
    else: return 1/10000

def square(x):
    return x*x


sigmoid_v = np.vectorize(sigmoid)
sigmoid_d_v = np.vectorize(sigmoid_d)
relu_mod_v = np.vectorize(relu_mod)
square_v = np.vectorize(square)


def rel_error(new_error, old_error):
    return abs(new_error - old_error) / new_error


def obtain_y(X, weights_capa1):
    return relu_mod_v(X.dot(weights_capa1))


def obtain_z(Y, weights_capa2):
    return sigmoid_v(Y.dot(weights_capa2))


def obtain_Ekt(label_v, Z):
    return Z - label_v


def calculate_error(Ekt):
    aux = square_v(Ekt).dot(np.ones(10))
    return 0.5 * float(np.ones(M).dot(aux))


def predict(weights_capa1, weights_capa2, foto):
    y0 = obtain_y(foto, weights_capa1)
    z0 = obtain_z(y0, weights_capa2)
    return int(np.argmax(z0))

def print_precision(weights_capa1, weights_capa2):
    aciertos = 0
    for k0 in range(len(fotos_test)):
        if (predict(weights_capa1, weights_capa2, fotos_test[k0]) == int(label_test[k0])): aciertos += 1
    porcentaje_aciertos = 100 * aciertos / len(fotos_test)
    print("El porcentaje de aciertos es del: ", str(porcentaje_aciertos), "%\n")


def obtain_delta_capa2(Z, Ekt):
    return np.multiply(Ekt, sigmoid_d_v(Z))


def grad_capa2(Y, delta_capa2):
    grad_weights_capa2 = np.zeros((N1, 10))
    for j in range(N1):
        for t in range(10):
            grad_weights_capa2[j, t] = delta_capa2[:, t].dot(Y[:, j])
    return grad_weights_capa2


def obtain_delta_capa1(Y, Ekt, weights_capa2, delta_capa2):
    delta_capa1 = np.zeros((M, N1))
    for k in range(M):
        for j in range(N1):
            delta_capa1[k, j] = relu_d(float(Y[k, j])) * delta_capa2[k, :].dot(weights_capa2[j, :])
    return delta_capa1


def grad_capa1(X, delta_capa1):
    grad_weights_capa1 = np.zeros((N0, N1))
    for i in range(N0):
        for j in range(N1):
            grad_weights_capa1[i, j] = delta_capa1[:, j].dot(X[:, i])
    return grad_weights_capa1


def main():
    np.random.seed(2)
    weights_capa1 = np.sqrt(2 / N0) * np.random.rand(N0, N1)  # [i, j]
    weights_capa2 = np.sqrt(2 / N1) * np.random.rand(N1, 10)  # [j, t]
    Y = obtain_y(X, weights_capa1)
    Y[:, -1] = 1
    Z = obtain_z(Y, weights_capa2)
    Ekt = obtain_Ekt(label_v, Z)
    eps = 1e-4
    n_iteraciones = 800
    cont = 0
    learning_rate = 0.1

    while (cont < n_iteraciones):
        cont += 1
        if (cont % 10 == 0): print_precision(weights_capa1, weights_capa2)

        start = time.time()
        delta_capa2 = obtain_delta_capa2(Z, Ekt)
        delta_capa1 = obtain_delta_capa1(Y, Ekt, weights_capa2, delta_capa2)
        weights_capa2 -= learning_rate * grad_capa2(Y, delta_capa2)
        weights_capa1 -= learning_rate * grad_capa1(X, delta_capa1)

        Y = obtain_y(X, weights_capa1)
        Y[:, -1] = 1
        Z = obtain_z(Y, weights_capa2)

        Ekt = obtain_Ekt(label_v, Z)
        end = time.time()

        print("Iteracion: ", str(cont), "   Time elapsed: ", str(end - start), "\n")
        if (cont % 10 == 0):
            error = calculate_error(Ekt)
            print("Error: ", error, "\n")

main()
