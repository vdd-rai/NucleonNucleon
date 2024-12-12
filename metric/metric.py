#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:51:45 2024

@author: velni
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from terms import g1,g2,g3,g4,g5,g6,g7

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
    d1 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/{mod}/d1/d1_U_r={r_index}_Q={Q_index}.npy')
    d2 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/{mod}/d2/d2_U_r={r_index}_Q={Q_index}.npy')
    d3 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/{mod}/d3/d3_U_r={r_index}_Q={Q_index}.npy')

    DA = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/{mod}/D{A}/D{A}_U_r={r_index}_Q={Q_index}.npy')
    DB = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/{mod}/D{B}/D{B}_U_r={r_index}_Q=-{Q_index}.npy')

    g = 0
    
    if mod == 0:
        for idx in idx_list:
            G1 = g1.g1(idx,DA,DB,d1,d2,d3)
            G2 = g2.g2(idx,DA,DB,d1,d2,d3)
            G3 = g3.g3(idx,DA,DB,d1,d2,d3)
            
            g = g + 2*(G1 + G2 - G3)
    
    if mod == 6:
        for idx in idx_list:
            G1 = g1.g1(idx,DA,DB,d1,d2,d3)
            G2 = g2.g2(idx,DA,DB,d1,d2,d3)
            G3 = g3.g3(idx,DA,DB,d1,d2,d3)
            G4 = g4.g4(idx,DA,DB,d1,d2,d3)
            G5 = g5.g5(idx,DA,DB,d1,d2,d3)
            G6 = g6.g6(idx,DA,DB,d1,d2,d3)
            G7 = g7.g7(idx,DA,DB,d1,d2,d3)
            
            g = g + 2*(G1 + G2 - G3 + c6*G4 - c6*G5 + (c6/2.)*G6 - (c6/2.)*G7)
    
    return g

#%% output

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')
Q_vals = np.load('/home/velni/Escritorio/TFM/py/sample/Q.npy')

point_list = []
for i in range(len(r_vals)):
    for j in range(len(Q_vals)):
        point_list.append([i,j])

m = 0.526
c6 = 1.

#

import time

start_time = time.time()

g_14 = np.zeros(len(Q_vals))

for r_index in range(50,61):
    for Q_index in range(len(Q_vals)):
        g_14[Q_index] = g_ij(1,4,c6,r_index,Q_index,0)
    np.save(f'/home/velni/Escritorio/TFM/py/metric/data/0/pre/-g_14_r={r_index}', g_14) 

print()
print()
print("--- runtime : %s seconds ---" % (time.time() - start_time)) # 400 seconds
        
"""
g = np.zeros((len(r_vals),len(Q_vals),9,9))

for r_index in range(len(r_vals)):
    for Q_index in range(len(Q_vals)):
        for A in range(9):
            for B in range(9):
                g[r_index][Q_index][A][B] = g_ij_6(A,B,c6,r_index,Q_index)
        np.save(f'/home/velni/Escritorio/TFM/py/metric/data/0/g_r={r_index}_Q={Q_index}', g)
        g = np.zeros((len(r_vals),len(Q_vals),9,9))

Q_index = 0

for r_index in range(len(r_vals)):
    for A in range(9):
        for B in range(9):
            g[r_index][Q_index][A][B] = g_ij_6(A,B,c6,r_index,Q_index)
    np.save(f'/home/velni/Escritorio/TFM/py/metric/data/0/g_r={r_index}_Q={Q_index}', g)
    g = np.zeros((len(r_vals),len(Q_vals),9,9))
"""

#%%

g_l = []

for r_index in range(14):
    g_temp = 0
    for Q_index in range(len(Q_vals)):
        k = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/0/g_12_r={r_index}_Q={Q_index}.npy')
        g_temp = g_temp + k
    g_l.append(g_temp)
    
#%% merge

for r_index in range(61):
    G = []
    g_p = list(np.load(f'/home/velni/Escritorio/TFM/py/metric/data/0/pre/g_14_r={r_index}.npy'))
    g_m = list(np.load(f'/home/velni/Escritorio/TFM/py/metric/data/0/pre/-g_14_r={r_index}.npy'))
    G = g_m + g_p
    np.save(f'/home/velni/Escritorio/TFM/py/metric/data/0/g_14_r={r_index}', G) 

#%%

Q_slice = 23
G = []

for r_index in range(len(r_vals)):
    g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/0/g_14_r={r_index}.npy')
    G.append(g[Q_slice])

plt.plot(np.arange(0,32,1),g,'k-')
plt.show()








