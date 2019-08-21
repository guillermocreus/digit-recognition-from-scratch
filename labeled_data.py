import numpy as np

def import_labeled_data():
	csv = np.genfromtxt('data/train.csv', delimiter=",")
	label = csv[1:, 0]
	fotos = csv[1:, 1:]
	fotos /= 783
	M = len(fotos)  # dimension de data
	v = 10 * [0]
	for k in range(M):
		v[int(label[k])] += 1

	A = np.empty((10,),dtype=object)
	for i in range(10):
		A[i] = np.empty((v[i], 784))

	cont = 10 * [0]
	for k in range(M):
		i = int(label[k])
		A[i][cont[i]] = fotos[k]
		cont[i] += 1

	del fotos  # libero memoria de las fotos (ya no se usaran)

	return A