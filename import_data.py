import numpy as np

csv = np.genfromtxt('data/train.csv', delimiter=",")
label = csv[1:, 0]
data = csv[1:, 1:]

M = len(data)
data = np.append(data, np.ones((M, 1)), axis=1) #added bias node
