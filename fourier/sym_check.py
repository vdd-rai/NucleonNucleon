#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 11:05:10 2025

@author: velni
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from m_terms import g1,g2,g3,g4,g5,g6,g7

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

### y grid

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3)):
            idx_list.append([i,j,k])

#%% metric

def g_ij(A,B,c6,r_index,Q_index,mod): # scalar value, 9x9
    
    # data
    d1 = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d1/d1_U_r={r_index}_Q={Q_index}.npy')
    d2 = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d2/d2_U_r={r_index}_Q={Q_index}.npy')
    d3 = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d3/d3_U_r={r_index}_Q={Q_index}.npy')

    DA = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/D{A}/D{A}_U_r={r_index}_Q={Q_index}.npy')
    DB = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/D{B}/D{B}_U_r={r_index}_Q={Q_index}.npy')

    g = 0
    
    if mod == 0:
        for idx in idx_list:
            G1 = g1.g1(idx,DA,DB,d1,d2,d3)
            G2 = g2.g2(idx,DA,DB,d1,d2,d3)
            G3 = g3.g3(idx,DA,DB,d1,d2,d3)
            
            g = g + (2*(G1 + G2 - G3))
    
    if mod == 6:
        for idx in idx_list:
            G1 = g1.g1(idx,DA,DB,d1,d2,d3)
            G2 = g2.g2(idx,DA,DB,d1,d2,d3)
            G3 = g3.g3(idx,DA,DB,d1,d2,d3)
            G4 = g4.g4(idx,DA,DB,d1,d2,d3)
            G5 = g5.g5(idx,DA,DB,d1,d2,d3)
            G6 = g6.g6(idx,DA,DB,d1,d2,d3)
            G7 = g7.g7(idx,DA,DB,d1,d2,d3)
            
            g = g + (2*(G1 + G2 - G3 + c6*G4 - c6*G5 + (c6/2.)*G6 - (c6/2.)*G7) )
    
    return g*(0.2**3)

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')
Q_vals = np.load('/home/velni/phd/w/tfm/py/sample/Q.npy')

point_list = []
for i in range(len(r_vals)):
    for j in range(len(Q_vals)):
        point_list.append([i,j])

m = 0.526
c6 = 1.

#%% -A^1_0;12 = A^1_0;21

import time

start_time = time.time()

R = [13,14,15]

G1 = np.zeros((3,len(Q_vals)))
for r_idx,r in enumerate(R):
    for Q_index in range(len(Q_vals)):
        G1[r_idx][Q_index] = g_ij(1,5,c6,r,Q_index,0)

G2 = np.zeros((3,len(Q_vals)))
for r_idx,r in enumerate(R):
    for Q_index in range(len(Q_vals)):
        G2[r_idx][Q_index] = g_ij(2,4,c6,r,Q_index,0)
        
### Fourier

dQ_vals = np.load('/home/velni/phd/w/tfm/py/sample/dQ.npy')

vol = dQ_vals[0]*32

def A_0(G,r_index,mod):
    
    # Initialize value
    A = 0
    
    # Integrate over Q
    A = sum(G[r_index])*dQ_vals[0]
    
    return A/vol

###

X = 1.731 + 0.1*np.arange(R[0],R[-1]+1,1)
Y1 = [A_0(G1,r,0) for r in [0,1,2]] 
Y2 = [-A_0(G2,r,0) for r in [0,1,2]] 

plt.xlabel(r'$r$')
plt.ylabel(r'Coefficient')

plt.plot(X,Y1,'r--',label=r'$A^1_{0;12}$')
plt.plot(X,Y2,'b--',label=r'$-A^1_{0;21}$')
plt.legend()
plt.grid(True)
plt.show()

print()
print()
print("--- runtime : %s seconds ---" % (time.time() - start_time))

#%% A^1_0;11 = A^1_0;22 = 0

import time

start_time = time.time()

R = [13,14,15]

G1 = np.zeros((3,len(Q_vals)))
for r_idx,r in enumerate(R):
    for Q_index in range(len(Q_vals)):
        G1[r_idx][Q_index] = g_ij(1,4,c6,r,Q_index,0)

G2 = np.zeros((3,len(Q_vals)))
for r_idx,r in enumerate(R):
    for Q_index in range(len(Q_vals)):
        G2[r_idx][Q_index] = g_ij(2,5,c6,r,Q_index,0)
        
### Fourier

dQ_vals = np.load('/home/velni/phd/w/tfm/py/sample/dQ.npy')

vol = dQ_vals[0]*32

def A_0(G,r_index,mod):
    
    # Initialize value
    A = 0
    
    # Integrate over Q
    A = sum(G[r_index])*dQ_vals[0]
    
    return A/vol

###

X = 1.731 + 0.1*np.arange(R[0],R[-1]+1,1)
Y1 = [A_0(G1,r,0) for r in [0,1,2]] 
Y2 = [A_0(G2,r,0) for r in [0,1,2]] 

plt.xlabel(r'$r$')
plt.ylabel(r'Coefficient')

plt.plot(X,Y1,'r-',label=r'$A^1_{0;11}$')
plt.plot(X,Y2,'b-',label=r'$A^1_{0;22}$')
plt.legend()
plt.grid(True)
plt.show()

print()
print()
print("--- runtime : %s seconds ---" % (time.time() - start_time)) # 1486 s

#%% A^1_0;13 = A^1_0;31 = 0

import time

start_time = time.time()

R = [13,14,15]

G1 = np.zeros((3,len(Q_vals)))
for r_idx,r in enumerate(R):
    for Q_index in range(len(Q_vals)):
        G1[r_idx][Q_index] = g_ij(1,6,c6,r,Q_index,0)

G2 = np.zeros((3,len(Q_vals)))
for r_idx,r in enumerate(R):
    for Q_index in range(len(Q_vals)):
        G2[r_idx][Q_index] = g_ij(3,4,c6,r,Q_index,0)
        
### Fourier

dQ_vals = np.load('/home/velni/phd/w/tfm/py/sample/dQ.npy')

vol = dQ_vals[0]*32

def A_0(G,r_index,mod):
    
    # Initialize value
    A = 0
    
    # Integrate over Q
    A = sum(G[r_index])*dQ_vals[0]
    
    return A/vol

###

X = 1.731 + 0.1*np.arange(R[0],R[-1]+1,1)
Y1 = [A_0(G1,r,0) for r in [0,1,2]] 
Y2 = [A_0(G2,r,0) for r in [0,1,2]] 

plt.xlabel(r'$r$')
plt.ylabel(r'Coefficient')

plt.plot(X,Y1,'r-',label=r'$A^1_{0;13}$')
plt.plot(X,Y2,'b-',label=r'$A^1_{0;31}$')
plt.legend()
plt.grid(True)
plt.show()

print()
print()
print("--- runtime : %s seconds ---" % (time.time() - start_time)) # 1486 s



