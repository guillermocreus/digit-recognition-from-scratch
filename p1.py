import numpy as np
from np import dot
from math import exp

N1 = 20

def sigmoid(x):
    return 1.0/(1+exp(-x))


def sigmoid_d(x):
    return sigmoid(x)*(1-sigmoid(x))

def grad_capa1_ij(data, weights_capa1, weights_capa2, i0, j0):
    M = len(data)
    result = 0
    for k in range(M):
        for t in range(10):
            result += -2 * E[k][t] * sigmoid_d(z_barra[k][t]) * (weights_capa2[j0][t]) * sigmoid_d(data[k].dot(weights_capa1[i])) * data[j0]
    return result

def grad_capa2_jt(data, weights_capa1, weights_capa2, j0, to):
    M = len(data)
    result = 0
    for k in range(M):
        result += -2 * E[k][t0] * sigmoid_d(z_barra[k][t0]) * y[j0] 
    

def grad_capa1(data, weights_capa1, N1):
    M = len(data)
    grad_capa1 = [[]]
    for i in range(28*28 + 1):
        for j in range(N1):
            grad_capa1[i][j] = grad_capa1_ij(data, weights_capa1, i, j)
        
