import numpy as np
from np import dot
from math import exp

N0 = 28 * 28 + 1  # dimension layer 0
N1 = 20  # dimension layer 1
M = len(csv)  # dimension de data


def sigmoid(x):
    return 1.0/(1 + exp(-x))


def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))


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
	    if (label[k] == t) zkt = 1  # lo he cmabiado esto!!! creo q asi esta bien...
	    Ekt[k, t] = (Z[k, t] - zkt)
    return Ekt

def calculate_error(Ekt):
	return 0.5 * float(np.ones(M).dot(square_v(Ekt)).dot(np.ones(10).T))
	
"""
def obtain_Ekt(label, Z, weights_capa1, weights_capa2): #Ekt Matriz, asumiendo label vector fila de dimension M
	Ekt = Z.copy()
	for k in range(M):
		Ekt[k,label[k]] += -1
    return Ekt
"""   

    
def grad_capa2_jt(Y, weights_capa1, weights_capa2, j0, t0):
    derivada_jt = 0
    for k in range(M):
        derivada_jt += E[k,t0] * sigmoid_d(float(Y[k].dot(weights_capa2.T[t0].T))) * Y[k,j0]
        return derivada_jt

def grad_capa2(Y, weights_capa1, weights_capa2):
	grad_weights_capa2 = np.zeros((N1, 10))
	for j in range(N1):
		for t in range(10):
			grad_weights_capa2[j,t] = grad_capa2_jt(Y, weights_capa1, weights_capa2, j, t)
	return grad_weights_capa2
          
def grad_capa1_ij(X, Y, weights_capa1, weights_capa2, i0, j0):
    M = len(data)
    derivada_ij = 0
    for k in range(M):
        for t in range(10):
            derivada_ij += E[k,t] * sigmoid_d(float(Y[k].dot(weights_capa2.T[t0].T))) * weights_capa2[j0,t] * sigmoid_d(float(X[k].dot(weights_capa1.T[j0].T))) * X[k,i0]
    return derivada_ij

def grad_capa1(X, Y, weights_capa1, weights_capa2):
	grad_weights_capa1 = np.zeros((N1, 10))
	for i in range(N1):
		for j in range(10):
			grad_weights_capa1[i,j] = grad_capa1_ij(X, Y, weights_capa1, weights_capa2, i, j)
	return grad_weights_capa1

def main():    
    csv = np.genfromtxt('data/train.csv', delimiter=",")
    label = csv[1:, 0]
    data = csv[1:, 1:]
    weights_capa1 = np.zeros((N0, N1))
    weights_capa2 = np.zeros((N1, 10))
    Y = obtain_y(X, weights_capa1)
    Z = obtain_z(Y, weights_capa2)
    Ekt = obtain_Ekt(label, Z, weights_capa1, weights_capa2)

main()

epsilon = 10e-6
learning_rate = 0.01

old_error = np.inf
new_error = calculate_error(Ekt)

while rel_error > eps:
	new_weights_capa1 = weights_capa1 - learning_rate * grad_capa1(X, Y, weights_capa1, weights_capa2)
	new_weights_capa2 = weights_capa2 - learning_rate * grad_capa2(Y, weights_capa1, weights_capa2)
	
	weights_capa1 = new_weights_capa1.copy()
	weights_capa2 = new_weights_capa2.copy()
    Y = obtain_y(X, weights_capa1)
    Z = obtain_z(Y, weights_capa2)
    Ekt = obtain_Ekt(label, Z, weights_capa1, weights_capa2)
    old_error = new_error
    new_error = calculate_error(Ekt)
