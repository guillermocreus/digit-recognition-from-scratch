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

factor_capa1 = 1e-1
factor_capa2 = 1e-3

def sigmoid(x):
    var = x * 1e-2
    return 1.0/(1 + np.exp(-var))
    if (var < -5): return 0.000001
    elif (var > 5): return 0.999999

def sigmoid_d(y):
    return y*(1-y) * 1e-2
    
def relu_mod(x):
    var = x * factor_capa1
    if (var > 0):
        return var
    else: 
        return var * 1e-3
    
def relu_d(y):
    if (y > 0):
        return factor_capa1
    else:
        return factor_capa1 * 1e-3

def square(x):
    return x * x

def divide(x):
    return 1 / x

def exponencial(x):
    var = x * factor_capa2
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
    aux = exp_v(bkt)
    suma = aux.dot(np.ones(10))
    Z = (aux.T * divide_v(suma)).T
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
    y0 = relu_mod_v(fotos_test.dot(weights_capa1))
    z0 = exp_v(y0.dot(weights_capa2))
    suma = z0.dot(np.ones(10))
    z0 = (z0.T * divide_v(suma)).T

    aciertos = 0
    for k0 in range(len(fotos_test)):
        if (int(np.argmax(z0[k0]) == int(label_test[k0]))): 
            aciertos += 1
    porcentaje_aciertos = 100 * aciertos / len(fotos_test)
    print("El porcentaje de aciertos del TEST es del: ", str(porcentaje_aciertos), "%\n")


def print_precision_train(weights_capa1, weights_capa2):
    y0 = relu_mod_v(X.dot(weights_capa1))
    z0 = exp_v(y0.dot(weights_capa2))
    suma = z0.dot(np.ones(10))
    z0 = (z0.T * divide_v(suma)).T

    aciertos = 0
    for k0 in range(len(X)):
        if (int(np.argmax(z0[k0])) == int(label_train[k0])): 
            aciertos += 1
    porcentaje_aciertos = 100 * aciertos / len(X)
    print("El porcentaje de aciertos del TRAIN es del: ", str(porcentaje_aciertos), "%\n")


def obtain_delta_capa2(Z, label_v):
    return (Z - label_v) * factor_capa2


def grad_capa2(Y, delta_capa2):
    return Y.T.dot(delta_capa2)


def obtain_delta_capa1(X, Y, weights_capa1, weights_capa2, delta_capa2):
    return np.multiply(relu_d_v(X.dot(weights_capa1)), delta_capa2.dot(weights_capa2.T))


def grad_capa1(X, delta_capa1):
    return X.T.dot(delta_capa1)


def dynamic_learning_rate(cont):
    return 0.2 * (1 + np.exp(-5 * cont / 1000))


def main():
    np.random.seed(24)
    weights_capa1 = np.sqrt(2 / N0) * np.random.rand(N0, N1)  # [i, j]
    weights_capa2 = np.sqrt(2 / N1) * np.random.rand(N1, 10)  # [j, t]
    Y = obtain_y(X, weights_capa1)
    Y[:, -1] = 1
    Z = obtain_z(Y, weights_capa2)
    eps = 1e-4
    n_iteraciones = 4000
    cont = 0
    error_antiguo = 5e7


    while (cont < n_iteraciones):
        learning_rate = dynamic_learning_rate(cont)
        cont += 1
        start = time.time()

        delta_capa2 = obtain_delta_capa2(Z, label_v)
        delta_capa1 = obtain_delta_capa1(X, Y, weights_capa1, weights_capa2, delta_capa2)

        update_capa2 = -learning_rate * grad_capa2(Y, delta_capa2)
        update_capa1 = -learning_rate * grad_capa1(X, delta_capa1)

        weights_capa2 += update_capa2 
        weights_capa1 += update_capa1


        Y = obtain_y(X, weights_capa1)
        Y[:, -1] = 1
        
        Z = obtain_z(Y, weights_capa2)

        end = time.time()

        print("Iteracion: ", cont, "   Time elapsed: ", str(end - start), "\n")

        if (cont % 10 == 0): 
            print("\n ESTADO DEL ENTRENAMIENTO:")
            print("\n________________________________\n")

            print_precision(weights_capa1, weights_capa2)
            print_precision_train(weights_capa1, weights_capa2)


            Ekt = obtain_Ekt(label_v, Z)
            error = calculate_error(Ekt)
            delta_error = error_antiguo - error
            error_antiguo = error
            print(" _____________________________________________________")
            print("|   Error: ", error, "   Delta Error: ", delta_error, "|")
            print(" _____________________________________________________")
            print()


main()

