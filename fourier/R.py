import numpy as np

###

Q_vals = np.load('/home/velni/phd/w/tfm/py/sample/Q.npy')
dQ_vals = np.load('/home/velni/phd/w/tfm/py/sample/dQ.npy')

vol = dQ_vals[0]*32

sigma = sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

###

R_vals_l = []

for Q_index in range(len(Q_vals)):
	R = np.zeros((3,3))
	for ap in range(3):
		for bp in range(3):
			R[ap][bp] = 0.5*np.trace(np.dot(sigma[ap],np.dot(Q_vals[Q_index],np.dot(sigma[bp],np.linalg.inv(Q_vals[Q_index])))))
	R_vals_l.append(R)

R_vals = np.array(R_vals_l)

###

a1 = 2
b1 = 2

R1 = R_vals[:,a1,b1]

a2 = 2
b2 = 2
R2 = R_vals[:,a2,b2]

RR = R1*R2

res = sum(RR)*dQ_vals[0]

print(res)

