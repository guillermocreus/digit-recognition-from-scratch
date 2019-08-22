import numpy as np
import time
from labeled_data import import_labeled_data
from itertools import combinations



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
        self.X = np.append(data, np.ones((self.M, 1)), axis=1)
        self.N0 = len(self.X)
        self.W = np.random.rand(self.N0, 1)
        
    def feed_forward(self):
        self.Z = sigmoid_v((self.X).dot(self.W))
        
    def obtain_Ekt(self):  # Ekt Matriz, asumiendo label vector fila de dimension M
        Ekt = np.zeros((M, 1))
        for k in range(M):
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
                gradient = delta[:, t].dot(X[:, j])
                
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


def main():
    data, vM = import_labeled_data()
    iterator = combinations('0123456789', 2) #iterador con (0,1), (0,2), ... , (8,9)
    binary_nets = {} #diccionario con todos los classificadores binarios
    for digits in iterator:
        d0 = int(digits[0])
        d1 = int(digits[1])
        X1 = A[d0]
        X2 = A[d1]
        X = np.concatenate((X1, X2), axis = 0) # v[r0] y v[s0] dan el numero de elementos existentes de ese label
        label = vM[d0]*[d0] + vM[d1]*[d1]
        NNB = Neural_Network_Binary(X, label, digits)
        NNB = gradient_descend(NNB, 5000, 1e-4, 0.01)
        binary_nets[digits] = NNB
    return binary_nets
        
        

