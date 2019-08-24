import sys
import numpy as np
from train_binary_classifier import train_classifier
sys.path.insert(0, 'Import_data')
from train_test_data import train_test_data

A1, A2, l1, l2, v1, v2 = train_test_data()
d0 = 0
d1 = 1
X1 = A1[d0]
X2 = A1[d1]
X = np.concatenate((X1, X2), axis=0)
separador = len(X1)
weights = train_classifier(X, separador, d0, d1)

for i in range(len(weights)):
    print(weights[i])
