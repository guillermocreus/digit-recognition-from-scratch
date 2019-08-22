import numpy as np

csv = np.genfromtxt('data/test.csv', delimiter=",")
foto1 = csv[1, :]
with open('numero.txt', 'w') as f:
	for i in range(28):
		v = foto1[28*i : 28*(i+1)]
		for j in range(len(v)):
			num = int(v[j])
			if v[j] > 0:
				f.write(str(num) + ' ')
			else :  
				f.write( '000' + ' ')
		f.write('\n')
		
		
	
