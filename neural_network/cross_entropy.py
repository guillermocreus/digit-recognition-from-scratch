import numpy as np
import time
from numpy import linalg as LA
import sys
sys.path.insert(0, '../Import_data')
from train_test_data import all_train_test_data



# ________ IMPORT DATA ______________

X, fotos_test, label_train, label_v, label_test = all_train_test_data()
M = len(X)

N0 = 28 * 28 + 1  # dimension layer 0
N1 = 20  # dimension layer 1
# ___________________________________



# __________ FUNCIONES ______________

def sigmoid(x):
    var = x * 1e-2
    return 1.0/(1 + np.exp(-var))
    if (var < -5): return 0.000001
    elif (var > 5): return 0.999999

def sigmoid_d(y):
    return y*(1-y) * 1e-2
    
def relu_mod(x):
    var = x * 1e-2
    if (var > 0): 
        return var
    else: 
        return var * 1e-3
    
def relu_d(y):
    if (y > 0): 
        return 1e-2
    else: 
        return 1e-5

def square(x):
    return x * x

def divide(x):
    return 1 / x

def exponencial(x, factor = 1e-3):
    var = x * factor
    if (var > 20):
        return np.exp(20)
    elif (var < -20):
        return np.exp(-20)
    return np.exp(var)


sigmoid_v = np.vectorize(sigmoid)
sigmoid_d_v = np.vectorize(sigmoid_d)
relu_mod_v = np.vectorize(relu_mod)
relu_d_v = np.vectorize(relu_d)
square_v = np.vectorize(square)
divide_v = np.vectorize(divide)
exp_v = np.vectorize(exponencial)

# _______________________________________


def softmax(Y, weights_capa2):
    bkt = Y.dot(weights_capa2)
    print(bkt[0,0], bkt[0,1], bkt[22, 3])
    aux = exp_v(bkt)
    suma = aux.dot(np.ones(10))
    Z = (aux.T * divide_v(suma)).T
    print(Z[0])
    print(Z[1])
    return Z


def rel_error(new_error, old_error):
    return abs(new_error - old_error) / new_error


def obtain_y(X, weights_capa1):
    return relu_mod_v(X.dot(weights_capa1))


def obtain_z(Y, weights_capa2):
    return softmax(Y, weights_capa2)


def obtain_Ekt(label_v, Z):
    return -np.multiply(label_v, np.log(Z))


def calculate_error(Ekt):
    Ek = Ekt.dot(np.ones(10))
    return float(np.ones(M).dot(Ek))


def predict(weights_capa1, weights_capa2, foto):
    y0 = relu_mod_v(foto.dot(weights_capa1))
    z0 = exp_v(y0.dot(weights_capa2))
    suma = float(z0.dot(np.ones(10)))
    z0 /= suma
    return int(np.argmax(z0))


def print_precision(weights_capa1, weights_capa2):
    aciertos = 0
    for k0 in range(len(fotos_test)):
        if (predict(weights_capa1, weights_capa2, fotos_test[k0]) == int(label_test[k0])): 
            aciertos += 1
    porcentaje_aciertos = 100 * aciertos / len(fotos_test)
    print("El porcentaje de aciertos es del: ", str(porcentaje_aciertos), "%\n")


def obtain_delta_capa2(Z, label_v, factor = 1e-3):
    sum_without_t0 = np.ones(Z.shape)
    sums = Z.dot(np.ones(10))
    sum_without_t0 = (sum_without_t0.T * sums).T
    sum_without_t0 = sum_without_t0 - Z
    left_in = np.multiply(label_v, sum_without_t0)

    right_in = np.ones(Z.shape)
    sums_label = label_v.dot(np.ones(10))
    right_in = (right_in.T * sums_label).T
    right_in -= label_v
    right_in = np.multiply(right_in, Z)
    
    return factor * ((left_in - right_in).T * divide_v(sums)).T


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
    learning_rate = 0.1

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
            print("Y = ", Y)
            for k in range(15):
                print("Z[" + str(k) + "] = ", Z[k], " , label = ", label_train[k])
            print_precision(weights_capa1, weights_capa2)
            Ekt = obtain_Ekt(label_v, Z)
            error = calculate_error(Ekt)
            print("Error: ", error, "\n")


main()

