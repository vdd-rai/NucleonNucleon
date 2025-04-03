import numpy as np

###

I = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

Q = np.load('/home/velni/phd/w/tfm/py/sample/Q.npy')
dQ = np.load('/home/velni/phd/w/tfm/py/sample/dQ.npy')

### exact calculation

R = [np.zeros((3,3),dtype="complex_") for idx in range(len(Q))]

for idx in range(len(Q)):
	for a in range(3):
		for b in range(3):
			R[idx][a,b] = (1/2)*np.trace( np.dot(sigma[a],np.dot(Q[idx],np.dot(sigma[b],np.linalg.inv(Q[idx])))) )

H = [I+R[idx] for idx in range(len(R))]

### coefficients

def O_zero(H,idx):
	H_list = [h[*idx] for h in H]

	vol = dQ[0]*32

	return (1/vol)*sum(H_list)*dQ[0]

def O_one(H,idx,idx2):
	H_list = [h[*idx] for h in H]

	A = 0
	for step in range(len(H_list)):
		arg = H_list[step]*R[step][idx2[0],idx2[1]]*dQ[0]
	A = A+arg

	vol = 0
	for step in range(len(R)):
		vol = vol + R[step][idx2[0],idx2[1]]*R[step][idx2[0],idx2[1]]*dQ[0]*32

	return A/vol


### check

Q_idx = 13
idx = [0,0] # ij
idx2 = [0,0] # ab

print('--- H=I+R')
print(H[Q_idx][*idx])
print('--- I')
print(f'Exact: {I[*idx]}')
print(f'Computed: {O_zero(H,idx)}')
print('--- R')
print(f'Exact: {R[Q_idx][*idx]}')
print(f'Computed: {O_one(H,idx,idx2)*R[Q_idx][*idx2]}')
