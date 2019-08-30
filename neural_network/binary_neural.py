# ___________ IMPORTS ______________________
import numpy as np
import time, sys
from itertools import combinations
from train_binary_classifier import train_classifier
sys.path.insert(0, '../Import_data')
from train_test_data import train_test_data
# __________________________________________

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


class Neural_Network_Binary:
    def __init__(self, data, label, digits): #digits sera tupla por ej (2,3)  y la neural net mapea 2 --> 1 y 3 --> 0
        self.M = len(label)
        self.digits = digits
        self.label = label
        self.X = data
        self.N0 = len(self.X[0])
        self.W = np.random.rand(self.N0, 1)
        
    def feed_forward(self):
        self.Z = sigmoid_v((self.X).dot(self.W))
        
    def obtain_Ekt(self):  # Ekt Matriz, asumiendo label vector fila de dimension M
        Ekt = np.zeros((self.M, 1))
        for k in range(self.M):
            target = 0
            if (self.label[k] == self.digits[0]): 
                target = 1
            Ekt = self.Z[k] - target
        return Ekt
    
    def obtain_delta(self, Ekt):
        delta = np.zeros((self.M, 1))
        for k in range(self.M):
            for t in range(1):
                delta[k, t] = Ekt[k, t] * sigmoid_d(float(Z[k, t]))
        return delta
        
    def obtain_gradient(self, delta):
        gradient = np.zeros((self.N0,1))
        for j in range(self.N0):
            for t in range(1):
                gradient = delta[:, t].dot(self.X[:, j])
                
    def calculate_error(self, Ekt):
        aux = square_v(Ekt).dot(np.ones(1))
        return 0.5 * float(np.ones(self.M).dot(aux))
        
        
def gradient_descend(NeuralNet, n_iteraciones, eps, learning_rate):
    cont = 0
    NeuralNet.feed_forward() #Inicializa toda la red (Calcula Z)
    Ekt = NeuralNet.obtain_Ekt()
    old_error = np.inf
    new_error = calculate_error(NeuralNet, Ekt)
    
    while (rel_error(new_error, old_error) > eps and cont < n_iteraciones): 
        cont += 1
        gradient = obtain_gradient(NeuralNet, obtain_delta(NeuralNet, Ekt))
        NeuralNet.W = NeuralNet.W - learning_rate * gradient
        NeuralNet.feed_forward() #Inicializa toda la red (Calcula Z)
        Ekt = NeuralNet.obtain_Ekt()
        old_error = new_error
        new_error = calculate_error(NeuralNet, Ekt)
    
    return NeuralNet    

'''

def main():
    data, vM = import_labeled_data()
    X = np.concatenate((data[1], data[1]), axis = 0) # v[r0] y v[s0] dan el numero de elementos existentes de ese label
    label = vM[0]*[0] + vM[1]*[1]
    NNB = Neural_Network_Binary(X, label, (0,1))
    NNB = gradient_descend(NNB, 5000, 1e-4, 0.01)
    print(NNB.W)

'''

def main():
    A_train, A_test, label_train, label_test, v_train, v_test = train_test_data()
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
        binary_nets[digits] = train_classifier(X, separador, d0, d1)
    
    return binary_nets


main()
