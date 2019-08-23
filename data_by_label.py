import numpy as np

csv = np.genfromtxt('data/train.csv', delimiter=",")
label = csv[1:, 0]
data_sin_bias = csv[1:, 1:]
data_sin_bias /= 783
result = [[]]

for i in range(9):
	result.append([])

for k in range(len(label)):
	result[int(label[k])].append(data_sin_bias)

for i in range(10):
	file_name = "label" + str(i) + ".csv"
	np.savetxt(file_name, result[i], delimiter= ",")