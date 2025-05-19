#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:46:23 2024

@author: velni
"""

import numpy as np

### y grid

y1 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y1.npy')
y2 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y2.npy')
y3 = np.load('/home2/victor.diaz/nnp/sample/reg_50x50x100/y3.npy')

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3[0])):
            idx_list.append([i,j,k])

###

r_vals = np.load('/home2/victor.diaz/nnp/sample/r.npy')
Q_vals = np.load('/home2/victor.diaz/nnp/sample/Q.npy')
dQ_vals = np.load('/home2/victor.diaz/nnp/sample/dQ.npy')

### R matrices

sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

R = [np.zeros((3,3),dtype=complex) for idx in range(len(Q_vals))]

for idx in range(len(Q_vals)):
    for a in range(3):
        for b in range(3):
            R[idx][a,b] = (1/2)*np.trace( np.dot(sigma[a],np.dot(Q_vals[idx],np.dot(sigma[b],np.linalg.inv(Q_vals[idx])))) )

### Coefficients

def A_0(alpha,i,j,r_index,mod):

    # Initialize value
    A = 0

    # Integrate over Q
    g = np.load(f'/home2/victor.diaz/nnp/metric/data/{mod}/g_{i}{j + 3*alpha}_r={r_index}.npy')

    for Q_index in range(len(Q_vals)):
        arg = g[Q_index]*dQ_vals[0]
        A = A + arg

    vol = dQ_vals[0]*32

    return A/vol

def A_ab(alpha,a,b,i,j,r_index,mod):

    # Integrate over Q
    g = np.load(f'/home2/victor.diaz/nnp/metric/data/{mod}/g_{i}{j + 3*alpha}_r={r_index}.npy')
    A = 0
    for Q_index in range(len(g)):
        arg = g[Q_index]*R[Q_index][a-1][b-1]*dQ_vals[0]
        A = A + arg

    vol = 0
    for step in range(len(R)):
        vol = vol + R[step][a-1][b-1]*R[step][a-1][b-1]*dQ_vals[0]

    return A/vol

#

def B_0(alpha,beta,i,j,L,r_index,mod):

    # Initialize value
    B = 0
    delta2 = np.identity(2)
    delta3 = np.identity(3)

    # Integrate over Q
    g = np.load(f'/home2/victor.diaz/nnp/metric/data/{mod}/g_{i+3*alpha}{j+3*beta}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
         B = B + (g[Q_index] - L*delta2[alpha-1][beta-1]*delta3[i-1][j-1])*dQ_vals[0]

    vol = dQ_vals[0]*32

    return B/vol

def B_ab(alpha,beta,a,b,i,j,r_index,mod):

    # Integrate over Q
    g = np.load(f'/home2/victor.diaz/nnp/metric/data/{mod}/g_{i+3*alpha}{j+3*beta}_r={r_index}.npy')
    B = 0
    for Q_index in range(len(g)):
        arg = g[Q_index]*R[Q_index][a-1][b-1]*dQ_vals[0]
        B = B + arg

    vol = 0
    for step in range(len(R)):
        vol = vol + R[step][a-1][b-1]*R[step][a-1][b-1]*dQ_vals[0]

    return B/vol

#

def C_0(i,j,M,r_index,mod):

    # Initialize value
    C = 0
    delta3 = np.identity(3)

    # Integrate over Q
    g = np.load(f'/home2/victor.diaz/nnp/metric/data/{mod}/g_{i}{j}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
         C = C + (g[Q_index] - (M/2)*delta3[i-1][j-1])*dQ_vals[Q_index]

    vol = dQ_vals[0]*32

    return C/(2*vol)

def C_ab(a,b,i,j,r_index,mod):

    # Integrate over Q
    g = np.load(f'/home2/victor.diaz/nnp/metric/data/{mod}/g_{i}{j}_r={r_index}.npy')
    C = 0
    for Q_index in range(len(g)):
        arg = g[Q_index]*R[Q_index][a-1][b-1]*dQ_vals[0]
        C = C + arg

    vol = 0
    for step in range(len(R)):
        vol = vol + R[step][a-1][b-1]*R[step][a-1][b-1]*dQ_vals[0]

    return C/(2*vol) # /2 was missing

#

def D_0(r_index,mod):

    # Initialize value
    D = 0

    # Integrate over Q
    V = np.load(f'/home2/victor.diaz/nnp/pot/data/{mod}/V_r={r_index}.npy')

    D = sum(V)
    vol = dQ_vals[0]*32

    return D/vol

def D_ab(a,b,r_index,mod):

    # Integrate over Q
    V = np.load(f'/home2/victor.diaz/nnp/pot/data/{mod}/V_r={r_index}.npy')
    D = 0
    for Q_index in range(len(V)):
        arg = V[Q_index]*R[Q_index][a-1][b-1]*dQ_vals[0]
        D = D + arg

    vol = 0
    for step in range(len(R)):
        vol = vol + R[step][a-1][b-1]*R[step][a-1][b-1]*dQ_vals[0]

    return D/vol # why not /2?

### output

M = 145.85
L = 106.83

fourier = [D_0(r,0) for r in r_vals]
np.save('/home2/victor.diaz/nnp/fourier/data/0/D_0.npy')
