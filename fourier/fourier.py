#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:46:23 2024

@author: velni
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

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

###

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')
Q_vals = np.load('/home/velni/phd/w/tfm/py/sample/Q.npy')
dQ_vals = np.load('/home/velni/phd/w/tfm/py/sample/dQ.npy')

### SU(2) volume # 2*np.pi**2

# import sympy as sp

# x,t,f = sp.symbols("x t f")
# arg = (sp.sin(x)**2)*sp.sin(t)
# vol = sp.integrate(arg, (x, 0, np.pi), (t, 0, np.pi), (f, 0, 2*np.pi))

### R matrices

sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

R = [np.zeros((3,3),dtype=complex) for idx in range(len(Q_vals))]

for idx in range(len(Q_vals)):
    for a in range(3):
        for b in range(3):
            R[idx][a,b] = (1/2)*np.trace( np.dot(sigma[a],np.dot(Q_vals[idx],np.dot(sigma[b],np.linalg.inv(Q_vals[idx])))) )

### Debug

# alpha=1
# i=1
# j=1
# a=2
# b=1
# r_index = 12
# Q_index = 0
# mod = 0
# idx = [25,25,26]

### Coefficients

def A_0(alpha,i,j,r_index,mod):

    # Initialize value
    A = 0

    # Integrate over Q
    g = np.load(f'/home/velni/phd/w/tfm/py/metric/data/{mod}/g_{i}{j + 3*alpha}_r={r_index}.npy')

    A = sum(g)*dQ_vals[0]
    vol = dQ_vals[0]*32

    return A/vol

def A_ab(alpha,a,b,i,j,r_index,mod):

    # Integrate over Q
    g = np.load(f'/home/velni/phd/w/tfm/py/metric/data/{mod}/g_{i}{j + 3*alpha}_r={r_index}.npy')
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
    delta = np.array([[1.,0.],[0.,1.]])

    # Integrate over Q
    g = np.load(f'/home/velni/phd/w/tfm/py/metric/data/{mod}/g_{i+3*alpha}{j+3*beta}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
         B = B + g[Q_index] - L*delta[alpha-1][beta-1]*delta[i-1][j-1]

    vol = dQ_vals[0]*32

    return B/vol

def B_ab(alpha,beta,a,b,i,j,r_index,mod):

    # Integrate over Q
    g = np.load(f'/home/velni/phd/w/tfm/py/metric/data/{mod}/g_{i+3*alpha}{j+3*alpha}_r={r_index}.npy')
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
    delta = np.array([[1.,0.],[0.,1.]])

    # Integrate over Q
    g = np.load(f'/home/velni/phd/w/tfm/py/metric/data/{mod}/g_{i}{j}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
         C = C + g[Q_index] - (M/2)*delta[i-1][j-1]

    vol = dQ_vals[0]*32

    return C/(2*vol)

def C_ab(a,b,i,j,r_index,mod):

    # Integrate over Q
    g = np.load(f'/home/velni/phd/w/tfm/py/metric/data/{mod}/g_{i}{j}_r={r_index}.npy')
    C = 0
    for Q_index in range(len(g)):
        arg = g[Q_index]*R[Q_index][a-1][b-1]*dQ_vals[0]
        C = C + arg

    vol = 0
    for step in range(len(R)):
        vol = vol + R[step][a-1][b-1]*R[step][a-1][b-1]*dQ_vals[0]

    return C/vol

#

def D_0(r_index,mod):

    # Initialize value
    D = 0

    # Integrate over Q
    V = np.load(f'/home/velni/phd/w/tfm/py/pot/data/{mod}/V_r={r_index}.npy')

    D = sum(V)
    vol = dQ_vals[0]*32

    return D/(2*vol)

def D_ab(a,b,r_index,mod):

    # Integrate over Q
    V = np.load(f'/home/velni/phd/w/tfm/py/pot/data/{mod}/V_r={r_index}.npy')
    D = 0
    for Q_index in range(len(V)):
        arg = V[Q_index]*R[Q_index][a-1][b-1]*dQ_vals[0]
        D = D + arg

    vol = 0
    for step in range(len(R)):
        vol = vol + R[step][a-1][b-1]*R[step][a-1][b-1]*dQ_vals[0]

    return D/(2*vol)

"""
def B_0(alpha,beta,i,j,L,r_index,mod):

    # Initialize value
    B = 0
    delta = np.array([[1.,0.],[0.,1.]])

    # Integrate over Q
    g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_{i + 3*alpha}{j + 3*beta}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
        B_s = g[Q_index] - L*delta[alpha-1][beta-1]*delta[i-1][j-1]
        B += B_s
        B_s = 0

    return B/vol

def B_ab(alpha,beta,i,j,a,b,r_index,mod):

    # Initialize values
    B = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # Integrate over Q
    g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_{i + 3*alpha}{j + 3*beta}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
        R = np.zeros((3,3))
        for ap in range(3):
            for bp in range(3):
                R[ap][bp] = 0.5*np.trace(np.dot(sigma[ap],np.dot(Q_vals[Q_index],np.dot(sigma[bp],np.linalg.inv(Q_vals[Q_index])))))
        B_s = g[Q_index]*np.linalg.inv(R)[a-1][b-1]
        B += B_s
        B_s = 0

    return B

#

def C_0(i,j,M,r_index,mod):

    # Initialize value
    C = 0
    delta = np.array([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.],[0.,0.,1.,0.,0.,],[0.,0.,0.,1.,0.],[0.,0.,0.,0.,1.]])

    # Integrate over Q
    g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_{i}{j}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
        C_s = g[Q_index] - 0.5*M*delta[i-1][j-1]
        C += C_s
        C_s = 0

    return C/(2*vol)

def C_ab(i,j,a,b,M,r_index,mod):

    # Initialize values
    C = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # Integrate over Q
    g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_{i}{j}_r={r_index}.npy')
    for Q_index in range(len(Q_vals)):
        R = np.zeros((3,3))
        for ap in range(3):
            for bp in range(3):
                R[ap][bp] = 0.5*np.trace(np.dot(sigma[ap],np.dot(Q_vals[Q_index],np.dot(sigma[bp],np.linalg.inv(Q_vals[Q_index])))))
        C_s = g[Q_index]*np.linalg.inv(R)[a-1][b-1]
        C += C_s
        C_s = 0

    return C

#

def D_0(r_index,mod):

    # Initialize value
    D = 0

    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        D_s = pot(r_index,Q_index,mod,0,0)
        D += D_s
        D_s = 0

    return D/(2*vol)

def D_ab(r_index,mod):

    # Initialize values
    D = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        R = np.zeros((3,3))
        for ap in range(3):
            for bp in range(3):
                R[ap][bp] = 0.5*np.trace(np.dot(sigma[ap],np.dot(Q_vals[Q_index],np.dot(sigma[bp],np.linalg.inv(Q_vals[Q_index])))))
        D_s = pot(r_index,Q_index,mod,0,0)*np.linalg.inv(R)[a-1][b-1]
        D += D_s
        D_s = 0

    return D/2.
"""


### plot

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

r_vals = np.arange(10,50,1)

X = 1.731 + 0.1*r_vals
Y = [-A_0(1,2,1,r,0) for r in r_vals]

plt.plot(X,Y,'k-',label=r'$-\mathcal{A}^{1}_{0;12}$')

#

plt.xlabel(r'$r$')
plt.ylabel(r'Coefficient')

plt.legend()
plt.show()















