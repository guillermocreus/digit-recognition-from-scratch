import numpy as np

csv = np.genfromtxt('data/train.csv', delimiter=",")
label = csv[1:, 0]
data = csv[1:, 1:]
