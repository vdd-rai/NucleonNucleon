import numpy as np

from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

#

I = np.array([[1.,0j],[0j,1.]])
sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

r_vals = np.load('/home2/victor.diaz/nnp/sample/r.npy')
Q_vals = np.load('/home2/victor.diaz/nnp/sample/Q.npy')

### y grid

y1 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y1.npy')
y2 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y2.npy')
y3 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y3.npy')

l1 = len(y1)
l2 = len(y2)
l3 = len(y3[0])

idx_list = []
for i in range(2,l1-2):
	for j in range(2,l2-2):
		for k in range(2,l3-2):
			idx_list.append([i,j,k])

### product approximation for derivatives

# d1

def d1U(r_index,Q_index,mod,hD):
	
	U_vals = np.load(f'/home2/victor.diaz/nnp/prod/data/{mod}/U_S_r={r_index}_Q={Q_index}.npy')
	d1_vals = np.zeros((l1,l2,l3,4))
	
	for idx in idx_list:
		p1 = U_vals[idx[0]+1,idx[1],idx[2]]
		p2 = U_vals[idx[0]+2,idx[1],idx[2]]
		m1 = U_vals[idx[0]-1,idx[1],idx[2]]
		m2 = U_vals[idx[0]-2,idx[1],idx[2]]
		
		vals = (-p2+8.0*p1-8.0*m1+m2)/(12*hD)
		d1_vals[*idx] = vals

	return d1_vals

# d2

def d2U(r_index,Q_index,mod,hD):
	
	U_vals = np.load(f'/home2/victor.diaz/nnp/prod/data/{mod}/U_S_r={r_index}_Q={Q_index}.npy')
	d2_vals = np.zeros((l1,l2,l3,4))
	
	for idx in idx_list:
		p1 = U_vals[idx[0],idx[1]+1,idx[2]]
		p2 = U_vals[idx[0],idx[1]+2,idx[2]]
		m1 = U_vals[idx[0],idx[1]-1,idx[2]]
		m2 = U_vals[idx[0],idx[1]-2,idx[2]]
		
		vals = (-p2+8.0*p1-8.0*m1+m2)/(12*hD)
		d2_vals[*idx] = vals

	return d2_vals

# d3

def d3U(r_index,Q_index,mod,hD):
	
	U_vals = np.load(f'/home2/victor.diaz/nnp/prod/data/{mod}/U_S_r={r_index}_Q={Q_index}.npy')
	d3_vals = np.zeros((l1,l2,l3,4))
	
	for idx in idx_list:
		p1 = U_vals[idx[0],idx[1],idx[2]+1]
		p2 = U_vals[idx[0],idx[1],idx[2]+2]
		m1 = U_vals[idx[0],idx[1],idx[2]-1]
		m2 = U_vals[idx[0],idx[1],idx[2]-2]
		
		vals = (-p2+8.0*p1-8.0*m1+m2)/(12*hD)
		d3_vals[*idx] = vals

	return d3_vals

### output

hD = 0.2

import time

start_time = time.time()

point_list = []
for i in range(len(r_vals)):
    for j in range(len(Q_vals)):
        point_list.append([i,j])

for i,point in enumerate(point_list):
    #np.save(f'/home2/victor.diaz/nnp/deriv/data/0/d1/d1_U_r={point[0]}_Q={point[1]}', d1U(point[0],point[1],0,hD))
    #np.save(f'/home2/victor.diaz/nnp/deriv/data/0/d2/d2_U_r={point[0]}_Q={point[1]}', d2U(point[0],point[1],0,hD))
    np.save(f'/home2/victor.diaz/nnp/deriv/data/0/d3/d3_U_r={point[0]}_Q={point[1]}', d3U(point[0],point[1],0,hD))
