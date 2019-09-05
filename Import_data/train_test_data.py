import numpy as np

def vector_label(label):
	M = len(label)
	label_v = np.zeros((M, 10))
	for k in range(M):
		label_v[k, int(label[k])] = 1

	return label_v

def shuffle(fotos, label):
        assert len(fotos) == len(label)
        vM_train = 10 * [0]
        vM_test = 10 * [0]
        M = len(fotos)
        N_train = int(0.75 * M)
        fotos_shuffled = np.empty(fotos.shape, dtype=fotos.dtype)
        label_shuffled = np.empty(label.shape, dtype=label.dtype)
        np.random.seed(2)
        perm = np.random.permutation(M)
        for i in range(M):
                fotos_shuffled[i] = fotos[perm[i]]
                label_shuffled[i] = label[perm[i]]
                if (i < N_train):
                        vM_train[int(label_shuffled[i])] += 1
                else:
                        vM_test[int(label_shuffled[i])] += 1

        return fotos_shuffled, label_shuffled, vM_train, vM_test

def train_test_data():
	csv = np.genfromtxt('../data/train.csv', delimiter=",")
	label = csv[1:, 0]
	fotos_sin_bias = csv[1:, 1:]
	fotos_sin_bias /= 783
	M = len(fotos_sin_bias)  #data dimension
	bias = np.ones((M, 1))
	fotos = np.append(fotos_sin_bias, bias, axis=1) #added bias node
	

	fotos_shuffled, label_shuffled, vM_train, vM_test = shuffle(fotos, label)
	N_train = int(0.75 * M)
	N_test = M - N_train
	
	fotos_train = fotos_shuffled[:N_train,:]
	label_train = label_shuffled[:N_train]

	fotos_test = fotos_shuffled[N_train:,:]
	label_test = label_shuffled[N_train:]

	A_train = np.empty((10,),dtype=object) #array donde componente i sera una matriz de data de digito i
	A_test = np.empty((10,),dtype=object) #array donde componente i sera una matriz de data de digito i
	
	for i in range(10):
		A_train[i] = np.empty((vM_train[i], 785))
		A_test[i] = np.empty((vM_test[i], 785))

	
	cont_train = 10 * [0]
	for k in range(N_train):
		i = int(label_train[k])
		A_train[i][cont_train[i]] = fotos_train[k]
		cont_train[i] += 1

	cont_test = 10 * [0]
	for k in range(N_test):
		i = int(label_test[k])
		A_test[i][cont_test[i]] = fotos_test[k]
		cont_test[i] += 1

	del fotos  # libero memoria de las fotos (ya no se usaran)

	return A_train, A_test, label_train, label_test, vM_train, vM_test

def all_train_test_data():
	csv = np.genfromtxt('../data/train.csv', delimiter=",")
	label = csv[1:, 0]
	fotos_sin_bias = csv[1:, 1:]
	fotos_sin_bias /= 783
	M = len(fotos_sin_bias)  #data dimension
	bias = np.ones((M, 1))
	fotos = np.append(fotos_sin_bias, bias, axis=1) #added bias node
	
	fotos_shuffled, label_shuffled, vM_train, vM_test = shuffle(fotos, label)
	N_train = int(0.75 * M)
	N_test = M - N_train
	
	fotos_train = fotos_shuffled[:N_train,:]
	label_train = label_shuffled[:N_train]

	fotos_test = fotos_shuffled[N_train:,:]
	label_test = label_shuffled[N_train:]

	label_v = vector_label(label_train)

	del fotos  # libero memoria de las fotos (ya no se usaran)

	return fotos_train, fotos_test, label_train, label_v, label_test
