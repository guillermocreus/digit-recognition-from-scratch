import numpy as np
import time
from labeled_data import import_labeled_data

# IMPORT DATA
A, v = import_labeled_data()

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


def obtain_Ek(Y, X, r0, s0):  # Ek Matriz, asumiendo label vector fila de dimension M
    M = len(X)
    Ek = np.zeros(M)
    for k in range(M):
        zk = 0
        if (k >= v[r0]):
            zk = 1
        Ek[k] = (Y[k] - zk)
    #print(Ek)
    return Ek


def calculate_error(Ek, X):
    M = len(X)
    aux = square_v(Ek)
    return 0.5 * float(np.ones(M).dot(aux))


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


# Entrena el clasificador binario de labels r0, s0 y devuelve los pesos 
# Supongo que el output correcto es s0 --> output = 1
def train_classifier(r0, s0):
    X = np.concatenate((A[r0], A[s0]), axis = 0) # v[r0] y v[s0] dan el numero de elementos existentes de ese label
    M = len(X)
    weights =  np.random.rand(N0)  # [i, t]
    Y = obtain_y(X, weights)
    Ek = obtain_Ek(Y, X, r0, s0)
    eps = 1e-5
    n_iteraciones = 500
    cont = 0
    learning_rate = 0.01
    old_error = np.inf
    new_error = calculate_error(Ek, X)

    
    csv = np.genfromtxt('data/test.csv', delimiter=",")
    foto1 = np.append(csv[1, :]/783, np.ones(1), axis=0)
    # while (rel_error(new_error, old_error) > eps and cont < n_iteraciones):
    while (cont < n_iteraciones):
        if (cont % 10 == 0): 
            print("THE END")
            print("Prediction", float(sigmoid(weights.dot(foto1))), float(weights.dot(foto1)))
        cont += 1
        print("buenas")
        start = time.time()
        weights -= learning_rate * grad(X, Y, Ek)
        Y = obtain_y(X, weights)
        Ek = obtain_Ek(Y, X, r0, s0)
        old_error = new_error
        new_error = calculate_error(Ek, X)
        end = time.time()
        print("stop")
        print(old_error, new_error)
        print('rel Error = ' + str(rel_error(new_error, old_error)) + ',', 'elapsed time = '+str(end-start)+',', cont)
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
    





def main():
    iterator = combinations('0123456789', 2) #iterador con (0,1), (0,2), ... , (8,9)
    binary_nets = {} #diccionario con todos los classificadores binarios
    for digits in iterator:
        digits = tuple(map(int,digits)) #pasar a integer
        d0 = digits[0]
        d1 = digits[1]
        W = train_classifier(d0,d1)
        d[digits] = W

    
    

    
	
train_classifier(0, 2)
