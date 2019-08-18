import numpy as np
from math import exp

import time

# IMPORT DATA
csv = np.genfromtxt('data/train.csv', delimiter=",")
label = csv[1:-2, 0]
data_sin_bias = csv[1:-2, 1:]
M = len(data_sin_bias)  # dimension de data
bias = np.ones((M, 1))
X = np.append(data_sin_bias, bias, axis=1)

N0 = 28 * 28 + 1  # dimension layer 0
N1 = 20  # dimension layer 1


def sigmoid(x):
    return 1.0/(1 + exp(-x))


def sigmoid_d(y):
    return y*(1-y)


def square(x):
    return x*x


sigmoid_v = np.vectorize(sigmoid)
sigmoid_d_v = np.vectorize(sigmoid_d)
square_v = np.vectorize(square)


def rel_error(new_error, old_error):
    return abs(new_error - old_error) / new_error


def obtain_y(X, weights_capa1):
    return sigmoid_v(X.dot(weights_capa1))


def obtain_z(Y, weights_capa2):
    return sigmoid_v(Y.dot(weights_capa2))


def obtain_Ekt(label, Z, weights_capa1, weights_capa2):  # Ekt Matriz, asumiendo label vector fila de dimension M
    Ekt = np.zeros((M, 10))
    for k in range(M):
        for t in range(10):
            zkt = 0
            if (int(label[k]) == t):
                zkt = 1  # lo he cmabiado esto!!! creo q asi esta bien...
            Ekt[k, t] = (Z[k, t] - zkt)

    return Ekt


def calculate_error(Ekt):
    return 0.5 * float(np.ones(M).dot(square_v(Ekt)).dot(np.ones(10).T)) 


def obtain_delta_capa2(Z, Ekt):
	delta_capa2 = np.zeros((M,10))
	for k in range(M):
		for t in range(10):
		    delta_capa2[k,t] = Ekt[k,t] * sigmoid_d(float(Z[k,t]))
	return delta_capa2


def grad_capa2(Y, delta_capa2):
    grad_weights_capa2 = np.zeros((N1, 10))
    for j in range(N1):
        for t in range(10):
            grad = 0
            for k in range(M):
                grad += delta_capa2[k,t]*Y[k,j]
            grad_weights_capa2[j, t] = grad
    return grad_weights_capa2


def obtain_delta_capa1(Y, Ekt, weights_capa2, delta_capa2):
    delta_capa1 = np.zeros((M,N1))
    for k in range(M):
        for j in range(N1):
            delta = 0
            for t in range(10):
                delta += delta_capa2[k,t]*weights_capa2[j,t] #podriamos ponerlo como dot product
            delta_capa1[k,j] = sigmoid_d(float(Y[k,j]))*delta
    return delta_capa1


def grad_capa1(X, delta_capa1):
    grad_weights_capa1 = np.zeros((N0, N1))
    for i in range(N0):
        for j in range(N1):
            grad = 0
            for k in range(M):
                grad += delta_capa1[k,j]*X[k,i]
            grad_weights_capa1[i, j] = grad
    grad_weights_capa1[:, -1] = 0  # Forzar ceros en gradiente para el bias CREO QUE ESTA MAL, ESTOY PENSANDO EN W transpuesta CREO _ REVISAR TODAS LAS W
    return grad_weights_capa1


def main():    
    weights_capa1 = np.zeros((N0, N1))
    weights_capa2 = np.zeros((N1, 10))
    Y = obtain_y(X, weights_capa1)
    Z = obtain_z(Y, weights_capa2)
    Ekt = obtain_Ekt(label, Z, weights_capa1, weights_capa2)
    eps = 1e-3
    n_iteraciones = 5000
    cont = 0
    learning_rate = 0.1

    old_error = np.inf
    new_error = calculate_error(Ekt)
    
    while (rel_error(new_error, old_error) > eps and cont < n_iteraciones):
        cont += 1
        print("buenas")
        start = time.time()
        delta_capa2 = obtain_delta_capa2(Z, Ekt)
        delta_capa1 = obtain_delta_capa1(Y, Ekt, weights_capa2, delta_capa2)
        new_weights_capa2 = weights_capa2 - learning_rate * grad_capa2(Y, delta_capa2)
        new_weights_capa1 = weights_capa1 - learning_rate * grad_capa1(X, delta_capa1)
        weights_capa1 = new_weights_capa1.copy()
        weights_capa2 = new_weights_capa2.copy()
        Y = obtain_y(X, weights_capa1)
        Z = obtain_z(Y, weights_capa2)
        Ekt = obtain_Ekt(label, Z, weights_capa1, weights_capa2)
        old_error = new_error
        new_error = calculate_error(Ekt)
        end = time.time()
        print('rel Error = ' + str(rel_error(new_error, old_error))+',', 'elapsed time = '+str(end-start)+',', cont)	
        
    print(weights_capa2[:,0])
	


main()
