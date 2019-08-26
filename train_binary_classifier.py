import numpy as np
import time
from numpy.linalg import inv
import sys
from itertools import combinations
sys.path.insert(0, 'Import_data')
from train_test_data import train_test_data

A_train, A_test, label_train, label_test, v_train, v_test = train_test_data()

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


def hessian(X, Y, Ek):
    M = len(X)
    Y_d = sigmoid_d_v(Y)
    H = np.zeros((785, 785))
    for k in range(M):
        for i in range(785):
            for j in range(785):
                if (i <= j):
                    H[i, j] += Y_d[k] * X[k][i] * X[k][j] * (Y_d[k] + Ek[k] * (1 - 2 * Y[k]))

        print("Acabo foto " + str(k))

    for i in range(785):
        for j in range(785):
            if (i > j):
                H[i, j] = H[j, i]

    return H

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
    eps = 1e-5
    n_iteraciones = 3
    cont = 0
    learning_rate = 0.01
    old_error = np.inf
    new_error = calculate_error(Ek, X)

    while (cont < n_iteraciones): #and rel_error(new_error, old_error) > eps):
        print("inside")
        cont += 1
        start = time.time()
        gradiente = grad(X, Y, Ek)
        Hessian = hessian(X, Y, Ek)
        print("hessiana acabada")
        weights -= learning_rate * inv(Hessian).dot(gradiente)
        #weights -= learning_rate * gradiente
        Y = obtain_y(X, weights)
        Ek = obtain_Ek(X, Y, separador, d0, d1)
        old_error = new_error
        new_error = calculate_error(Ek, X)
        end = time.time()
        print(old_error, new_error)
        print('Training ' + str(d0) + ' vs ' + str(d1) + ': -->', 'rel Error = ' + str(rel_error(new_error, old_error)) + ',', 'elapsed time = '+str(end-start)+',', cont, '\n')
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
    

def train_binary_nets():
    iterator = combinations('0123456789', 2)  # iterador con (0,1), (0,2), ... , (8,9)
    binary_nets = {}  # diccionario con todos los classificadores binarios

    for digits in iterator:

        digits = tuple(map(int,digits)) #pasar a integer
        d0 = digits[0]
        d1 = digits[1]
        X1 = A_train[d0]
        X2 = A_train[d1]
        X = np.concatenate((X1, X2), axis = 0)
        separador = len(X1)
        weights = train_classifier(X, separador, d0, d1)
        
        name = 'saved_weights/weights' + str(d0) + 'vs' + str(d1) + '.csv'
        np.savetxt(name, weights, delimiter=",")
        binary_nets[digits] = weights

    
    return binary_nets
    
def main ():
    binary_nets = train_binary_nets()   
    print("End training")
     
    N_aciertos = 0
    N_test = len(label_test)
    precision = N_aciertos/N_test
    for i in range(10):
        print(i)
        for foto in A_test[i]:
            prediction = test_data(binary_nets, foto)
            if (prediction == i): N_aciertos += 1
        print("Precision en digito ", i, " : ", precision) 
    print("Precision Final: ", precision)


main()
