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
        self.Z = (self.X).dot(self.W)
        
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
        self.W = self.W - learning_rate * gradient
        NeuralNet.feed_forward() #Inicializa toda la red (Calcula Z)
        Ekt = NeuralNet.obtain_Ekt()
        old_error = new_error
        new_error = calculate_error(NeuralNet, Ekt)
    
    return NeuralNet    

'''

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
