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
    var = x * 1e-7
    return 1.0/(1 + np.exp(-var))
    if (var < -5): return 0.000001
    elif (var > 5): return 0.999999

def sigmoid_d(y):
    return y*(1-y) * 1e-7
    
def relu_mod(x):
    var = x * 1e-4
    if (var > 0): 
        return var
    else: 
        return var * 1e-3
    
def relu_d(y):
    if (y > 0): 
        return 1e-4
    else: 
        return 1e-7

def square(x):
    return x*x


sigmoid_v = np.vectorize(sigmoid)
sigmoid_d_v = np.vectorize(sigmoid_d)
relu_mod_v = np.vectorize(relu_mod)
relu_d_v = np.vectorize(relu_d)
square_v = np.vectorize(square)


def rel_error(new_error, old_error):
    return abs(new_error - old_error) / new_error


def obtain_y(X, weights_capa1):
    return relu_mod_v(X.dot(weights_capa1))


def obtain_z(Y, weights_capa2):
    aux = sigmoid_v(Y.dot(weights_capa2))
    cont = 0
    for k in range(M):
        for t in range(10):
            if (aux[k][t] < 5*1e-2 or (1 - aux[k][t]) < 5*1e-2):
                cont += 1
    print("% Neuronas saturadas (Z):")
    print(100.0 * cont / (10 * M))
    return aux

def obtain_Ekt(label_v, Z):
    for i in range(10):
        print(label_v[i], Z[i])
    return np.multiply(label_v, np.log(Z))


def calculate_error(Ekt):
    Ek = Ekt.dot(np.ones(10))
    return - float(np.ones(M).dot(Ek))


def predict(weights_capa1, weights_capa2, foto):
    y0 = relu_mod_v(foto.dot(weights_capa1))
    z0 = sigmoid_v(y0.dot(weights_capa2))
    return int(np.argmax(z0))

def print_precision(weights_capa1, weights_capa2):
    aciertos = 0
    for k0 in range(len(fotos_test)):
        if (predict(weights_capa1, weights_capa2, fotos_test[k0]) == int(label_test[k0])): 
            aciertos += 1
    porcentaje_aciertos = 100 * aciertos / len(fotos_test)
    print("El porcentaje de aciertos es del: ", str(porcentaje_aciertos), "%\n")


def obtain_delta_capa2(Z, label_v):
    return np.multiply(label_v, np.ones(Z.shape) - Z)


def grad_capa2(Y, delta_capa2):
    return Y.T.dot(delta_capa2)


def obtain_delta_capa1(X, Y, weights_capa1, weights_capa2, delta_capa2):
	return np.multiply(relu_d_v(X.dot(weights_capa1)), delta_capa2.dot(weights_capa2.T))


def grad_capa1(X, delta_capa1):
	return X.T.dot(delta_capa1)


def main():
    np.random.seed(24)
    weights_capa1 = np.sqrt(2 / N0) * np.random.rand(N0, N1)  # [i, j]
    weights_capa2 = np.sqrt(2 / N1) * np.random.rand(N1, 10)  # [j, t]
    Y = obtain_y(X, weights_capa1)
    Y[:, -1] = 1
    Z = obtain_z(Y, weights_capa2)
    eps = 1e-4
    n_iteraciones = 800
    cont = 0
    learning_rate = 0.05

    while (cont < n_iteraciones):
        cont += 1
        start = time.time()

        delta_capa2 = obtain_delta_capa2(Z, label_v)
        delta_capa1 = obtain_delta_capa1(X, Y, weights_capa1, weights_capa2, delta_capa2)

        update_capa2 = -grad_capa2(Y, delta_capa2)
        update_capa1 = -grad_capa1(X, delta_capa1)

        weights_capa2 += update_capa2 
        weights_capa1 += update_capa1


        Y = obtain_y(X, weights_capa1)
        Y[:, -1] = 1
        Z = obtain_z(Y, weights_capa2)

        end = time.time()

        print("Iteracion: ", cont, "   Time elapsed: ", str(end - start), "\n")

        if (cont % 10 == 0): 
            print_precision(weights_capa1, weights_capa2)
            Ekt = obtain_Ekt(label_v, Z)
            error = calculate_error(Ekt)
            print("Error: ", error, "\n")


main()

