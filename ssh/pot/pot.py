#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:46:23 2024

@author: velni
"""

import numpy as np

from terms import v1,v2,v3,v6

### y grid

y1 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y1.npy')
y2 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y2.npy')
y3 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y3.npy')

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3[0])):
            idx_list.append([i,j,k])

### funcs

r_vals = np.load('/home2/victor.diaz/nnp/sample/r.npy')
Q_vals = np.load('/home2/victor.diaz/nnp/sample/Q.npy')

def pot(m,c6,r_index,Q_index,mod):

    # data

    #field = np.load(f'/home/velni/phd/w/tfm/py/prod/data/m/U_S_r={r_index}_Q={Q_index}.npy')
    d1 = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/d1/d1_U_r={r_index}_Q={Q_index}.npy')
    d2 = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/d2/d2_U_r={r_index}_Q={Q_index}.npy')
    d3 = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/d3/d3_U_r={r_index}_Q={Q_index}.npy')

    V = 0

    if mod == 0:
        for idx in idx_list:
            V1 = v1.v1(idx,d1,d2,d3)
            V2 = v2.v2(idx,d1,d2,d3)
            V3 = v3.v3(idx,d1,d2,d3)

            V = V + (V1 + 0.5*(V2 - V3))

    if mod == 1:
       for idx in idx_list:
            V1 = v1.v1(idx,d1,d2,d3)
            V2 = v2.v2(idx,d1,d2,d3)
            V3 = v3.v3(idx,d1,d2,d3)

            V = V + (V1 + 0.5*(V2 - V3) + 2*(m**2)*(field[*idx][0] - 1))

    if mod == 6:
        for idx in idx_list:
            V1 = v1.v1(idx,d1,d2,d3)
            V2 = v2.v2(idx,d1,d2,d3)
            V3 = v3.v3(idx,d1,d2,d3)
            V6 = v6.v6(idx,d1,d2,d3)

            #V = V + ???

    return V*(0.2**3)

### output

point_list = []
for i in range(len(r_vals)):
    for j in range(len(Q_vals)):
        point_list.append([i,j])

m = 0.526
c6 = 1.

# 

V = np.zeros(len(Q_vals))

for r_index in range(61):
    t0 = time.time()
    for Q_index in range(len(Q_vals)):
        V[Q_index] = pot(m,c6,r_index,Q_index,0)
    np.save(f'/home2/victor.diaz/nnp/pot/data/0/V_r={r_index}', V)
