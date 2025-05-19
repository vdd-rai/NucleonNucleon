#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:51:45 2024

@author: velni
"""

import numpy as np

from m_terms import g1,g2,g3,g4,g5,g6,g7

### y grid

y1 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y1.npy')
y2 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y2.npy')
y3 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y3.npy')

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3[0])):
            idx_list.append([i,j,k])

hD = 0.01

# metric

def g_ij(A,B,c6,r_index,Q_index,mod): # scalar value, 9x9

    # data
    d1 = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/d1/d1_U_r={r_index}_Q={Q_index}.npy')
    d2 = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/d2/d2_U_r={r_index}_Q={Q_index}.npy')
    d3 = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/d3/d3_U_r={r_index}_Q={Q_index}.npy')

    DA = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/D{A}/D{A}_U_r={r_index}_Q={Q_index}.npy')
    DB = np.load(f'/home2/victor.diaz/nnp/deriv/data/{mod}/D{B}/D{B}_U_r={r_index}_Q={Q_index}.npy')

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

    return g*(0.2*0.2*0.2)

# output

r_vals = np.load('/home2/victor.diaz/nnp/sample/r.npy')
Q_vals = np.load('/home2/victor.diaz/nnp/sample/Q.npy')

point_list = []
for i in range(len(r_vals)):
    for j in range(len(Q_vals)):
        point_list.append([i,j])

m = 0.526
c6 = 1.

#

g = np.zeros(len(Q_vals))

for r_index in range(0,61):
    for Q_index in range(len(Q_vals)):
        g[Q_index] = g_ij(3,3,c6,r_index,Q_index,0)
    np.save(f'/home2/victor.diaz/nnp/metric/data/0/g_11_r={r_index}', g)
    g = np.zeros(len(Q_vals))
