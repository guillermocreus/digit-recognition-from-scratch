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


import threading
import queue

queue = queue.Queue()

def MyThread1():
    A1, A2, l1, l2, v1, v2 = train_test_data()
    queue.put([A1, A2, l1, l2, v1, v2])
def MyThread2():
    A, B, c, d, e, f = train_test_data()
    queue.put([A, B, c, d, e, f])

t1 = threading.Thread(target=MyThread1, args=[])
t2 = threading.Thread(target=MyThread2, args=[])

t1.start()
t2.start()
t1.join()
t2.join()

result1 = queue.get()
result2 = queue.get()