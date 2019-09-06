import numpy as np
import time
import sys
from math import sqrt
from itertools import combinations
sys.path.insert(0, '../Import_data')
from train_test_data import all_train_test_data

fotos_train, fotos_test, label_train, label_test = all_train_test_data()

M = len(fotos_train)
N0 = 28 * 28 + 1  # dimension layer 0
N1 = 20  # dimension layer 1


def relu_mod(x):
    if (x > 0): return x/100
    else: return x/10000
    
def relu_d(y):
    if (y > 0): return 1/100
    else: return 1/10000


def sigmoid(x):
    return 1.0/(1 + np.exp(-x/30))
    if (x < -50):
        return 0
    elif (x > 50):
        return 1
    

def sigmoid_d(y):
    return y*(1-y)*1/30


def square(x):
    return x*x


sigmoid_v = np.vectorize(sigmoid)
sigmoid_d_v = np.vectorize(sigmoid_d)
square_v = np.vectorize(square)
relu_mod_v = np.vectorize(relu_mod)


def rel_error(new_error, old_error):
    return abs(new_error - old_error) / new_error


def obtain_y(X, weights_capa1):
    res = relu_mod_v(X.dot(weights_capa1))
    cont = 0
    for k in range(M):
        for j in range(N1):
            if (res[k][j] < 5*1e-2 or (1 - res[k][j]) < 5*1e-2):
                cont += 1
    #print("% Neuronas saturadas (Y):")
    #print(100.0 * cont / (N1 * M))
    return res


def obtain_z(Y, weights_capa2):
    res = sigmoid_v(Y.dot(weights_capa2))
    cont = 0
    for k in range(M):
        for t in range(10):
            if (res[k][t] < 5*1e-2 or (1 - res[k][t]) < 5*1e-2):
                cont += 1
    print("% Neuronas saturadas (Z):")
    print(100.0 * cont / (10 * M))
    return res


def obtain_Ekt(label, Z):  # Ekt Matriz, asumiendo label vector fila de dimension M
    Ekt = np.zeros((M, 10))
    for k in range(M):
        for t in range(10):
            zkt = 0
            if (int(label[k]) == t):
                zkt = 1
            Ekt[k, t] = (Z[k, t] - zkt)
    return Ekt


def calculate_error(Ekt):
    aux = square_v(Ekt).dot(np.ones(10))
    return 0.5 * float(np.ones(M).dot(aux))


def obtain_delta_capa2(Z, Ekt):
    delta_capa2 = np.zeros((M, 10))
    for k in range(M):
        for t in range(10):
            delta_capa2[k, t] = Ekt[k, t] * sigmoid_d(float(Z[k, t]))
    return delta_capa2


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


def find_alpha(weights_capa1, weights_capa2, dir_weights1, dir_weights2, grad_capa1, grad_capa2, label, X):
	scalar_prod = 0
	norm2 = 0
	for i in range(len(dir_weights1)):
		norm2 += square_v(dir_weights1[i]).dot(np.ones(len(dir_weights1[i])))
		scalar_prod += dir_weights1[i].dot(grad_capa1[i])

	for i in range(len(dir_weights2)):
		norm2 += square_v(dir_weights2[i]).dot(np.ones(len(dir_weights2[i])))
		scalar_prod += dir_weights2[i].dot(grad_capa2[i])

	norm2 = sqrt(norm2)
	m = scalar_prod / norm2
	print("m deberia ser negativa", m)
	c = 0.5
	t = -c * m

	Y_aux = obtain_y(X, weights_capa1)
	Z_aux = obtain_z(Y_aux, weights_capa2)
	Ekt_aux = obtain_Ekt(label, Z_aux)
	error = calculate_error(Ekt_aux)

	# Normalizo la direccion de descenso
	dir_weights1 /= norm2
	dir_weights2 /= norm2

	alpha = 0.2
	error_nuevo = 2 * error + 1
	new_weights1 = weights_capa1.copy()
	new_weights2 = weights_capa2.copy()

	while((error - error_nuevo) < alpha * t):
		alpha /= 2
		
		new_weights1 = weights_capa1.copy() + alpha*dir_weights1.copy()
		new_weights2 = weights_capa2.copy() + alpha*dir_weights2.copy()

		Y_aux = obtain_y(X, new_weights1)
		Z_aux = obtain_z(Y_aux, new_weights2)

		Ekt_aux = obtain_Ekt(label, Z_aux)
		error_nuevo = calculate_error(Ekt_aux)
		print(alpha, error_nuevo, error)
		print()

	print("THE alpha", alpha)
	return alpha

def obtain_beta(gradiente_capa1, gradiente_capa2, old_grad_capa1, old_grad_capa2):
    num = 0
    den = 0
    for i in range(len(gradiente_capa1)):
        num += gradiente_capa1[i].dot(gradiente_capa1[i])
        den += old_grad_capa1[i].dot(old_grad_capa1[i])
    
    for i in range(len(gradiente_capa2)):
        num += gradiente_capa2[i].dot(gradiente_capa2[i])
        den += old_grad_capa2[i].dot(old_grad_capa2[i])
        
    return num/den


def main():
	X = fotos_train
	weights_capa1 =  np.random.rand(N0, N1)  # [i, j]
	weights_capa2 =  np.random.rand(N1, 10)  # [j, t]
	Y = obtain_y(X, weights_capa1)
	Y[:, -1] = 1
	Z = obtain_z(Y, weights_capa2)
	Ekt = obtain_Ekt(label_train, Z)
	eps = 1e-4
	n_iteraciones = 100
	cont = 0
	learning_rate = 0.01
	old_error = np.inf
	new_error = calculate_error(Ekt)

	
	s_weights2 = 0
	s_weights1 = 0

	old_grad_capa2 = np.zeros((N1, 10))
	old_grad_capa1 = np.zeros((N0, N1))

	gradiente_capa2 = np.zeros((N1, 10))
	gradiente_capa1 = np.zeros((N0, N1))

	while (rel_error(new_error, old_error) > eps and cont < n_iteraciones):
		cont += 1

		if (cont % 10 == 0):
			for i in range(15) : 
				print('Z : ', Z[i, :], ' , prediction : ', np.argmax(Z[i,:]), ' , label : ', label_train[i])

		start = time.time()
		delta_capa2 = obtain_delta_capa2(Z, Ekt)
		delta_capa1 = obtain_delta_capa1(Y, Ekt, weights_capa2, delta_capa2)

		if (cont == 1):
			gradiente_capa2 = grad_capa2(Y, delta_capa2)
			gradiente_capa1 = grad_capa1(X, delta_capa1)

			s_weights2 = -gradiente_capa2
			s_weights1 = -gradiente_capa1

		else:
			old_grad_capa2 = gradiente_capa2.copy()
			old_grad_capa1 = gradiente_capa1.copy()

			gradiente_capa2 = grad_capa2(Y, delta_capa2)
			gradiente_capa1 = grad_capa1(X, delta_capa1)

			beta = obtain_beta(gradiente_capa1, gradiente_capa2, old_grad_capa1, old_grad_capa2)

			s_weights2 = -gradiente_capa2 + beta*s_weights2
			s_weights1 = -gradiente_capa1 + beta*s_weights1

		dir_weights2 = s_weights2
		dir_weights1 = s_weights1

		alpha = 0.1
		if (cont >= 2):
			alpha = find_alpha(weights_capa1, weights_capa2, dir_weights1, dir_weights2, gradiente_capa1, gradiente_capa2, label_train, X)

		weights_capa2 += alpha * dir_weights2
		weights_capa1 += alpha * dir_weights1

		Y = obtain_y(X, weights_capa1)
		Y[:, -1] = 1
		Z = obtain_z(Y, weights_capa2)
		Ekt = obtain_Ekt(label_train, Z)

		old_error = new_error
		new_error = calculate_error(Ekt)
		elapsed_time = time.time() - start

		print("Old error: ", old_error, "	New error: ", new_error)
		print('rel Error = ' + str(rel_error(new_error, old_error)) + ',', 'elapsed time = '+str(elapsed_time) + ', #Iteraciones: ', cont)
		print()

main()
