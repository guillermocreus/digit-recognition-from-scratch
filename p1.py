import numpy as np
import time
from labeled_data import import_labeled_data

# IMPORT DATA
A = import_labeled_data()
print(A[2][0])

csv = np.genfromtxt('data/train.csv', delimiter=",")
label = csv[1:30, 0]
data_sin_bias = csv[1:30, 1:]
data_sin_bias /= 783
M = len(data_sin_bias)  # dimension de data
bias = np.ones((M, 1))
X = np.append(data_sin_bias, bias, axis=1)

N0 = 28 * 28 + 1  # dimension layer 0
N1 = 20  # dimension layer 1


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


def rel_error(new_error, old_error):
    return abs(new_error - old_error) / new_error


def obtain_y(X, weights_capa1):
    res = sigmoid_v(X.dot(weights_capa1))
    cont = 0
    for k in range(M):
        for j in range(N1):
            if (res[k][j] < 5*1e-2 or (1 - res[k][j]) < 5*1e-2):
                cont += 1
    print("% Neuronas saturadas (Y):")
    print(100.0 * cont / (N1 * M))
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


def obtain_Ekt(label, Z, weights_capa1, weights_capa2):  # Ekt Matriz, asumiendo label vector fila de dimension M
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
            delta_capa1[k, j] = sigmoid_d(float(Y[k, j])) * delta_capa2[k, :].dot(weights_capa2[j, :])
    return delta_capa1


def grad_capa1(X, delta_capa1):
    grad_weights_capa1 = np.zeros((N0, N1))
    for i in range(N0):
        for j in range(N1):
            grad_weights_capa1[i, j] = delta_capa1[:, j].dot(X[:, i])
    # grad_weights_capa1[:, -1] = 0  # Forzar ceros en gradiente para el bias CREO QUE ESTA MAL, ESTOY PENSANDO EN W transpuesta CREO _ REVISAR TODAS LAS W
    return grad_weights_capa1


def main():
    weights_capa1 =  np.random.rand(N0, N1)  # [i, j]
    weights_capa2 =  np.random.rand(N1, 10)  # [j, t]
    Y = obtain_y(X, weights_capa1)
    Y[:, -1] = 1
    Z = obtain_z(Y, weights_capa2)
    Ekt = obtain_Ekt(label, Z, weights_capa1, weights_capa2)
    eps = 1e-4
    n_iteraciones = 0
    cont = 0
    learning_rate = 0.01
    print(label[1])
    print(label[2])
    old_error = np.inf
    new_error = calculate_error(Ekt)
    
    while (rel_error(new_error, old_error) > eps and cont < n_iteraciones):
        # while (cont < n_iteraciones):
        cont += 1
        print("buenas")
        start = time.time()
        delta_capa2 = obtain_delta_capa2(Z, Ekt)
        delta_capa1 = obtain_delta_capa1(Y, Ekt, weights_capa2, delta_capa2)
        weights_capa2 -= learning_rate * grad_capa2(Y, delta_capa2)
        weights_capa1 -= learning_rate * grad_capa1(X, delta_capa1)
        Y = obtain_y(X, weights_capa1)
        Y[:, -1] = 1
        Z = obtain_z(Y, weights_capa2)
        Ekt = obtain_Ekt(label, Z, weights_capa1, weights_capa2)
        old_error = new_error
        new_error = calculate_error(Ekt)
        end = time.time()
        print("stop")
        print(old_error, new_error)
        print('rel Error = ' + str(rel_error(new_error, old_error)) + ',', 'elapsed time = '+str(end-start)+',', cont)
        print()
    print(Z[1,:])
    print(Z[2,:])

main()
