#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:46:23 2024

@author: velni
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

from terms import v1,v2,v3,v6

### y grid

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3)):
            idx_list.append([i,j,k])
            
###

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')
Q_vals = np.load('/home/velni/Escritorio/TFM/py/sample/Q.npy')

#%% Potential
"""
def V_0(r_index):
    
    V_vals = np.zeros(len(Q_vals))
    V = 0
    
    start_time = time.time()

    for Q_index in range(len(Q_vals)):
        d1 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d1/d1_U_r={r_index}_Q={Q_index}.npy')
        d2 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d2/d2_U_r={r_index}_Q={Q_index}.npy')
        d3 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d3/d3_U_r={r_index}_Q={Q_index}.npy')
        
        for idx in idx_list:
            V1 = v1.v1(idx,d1,d2,d3)
            V2 = v2.v2(idx,d1,d2,d3)
            V3 = v3.v3(idx,d1,d2,d3)
        
            V += V1 + 0.5*(V2 + V3)
        
        V_vals[Q_index] = V
        V = 0
        print('')
        print("--- runtime : %s seconds ---" % (time.time() - start_time))
        
    return V_vals
    
#%% output

V_vals = V_0(0)
"""









