#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:46:23 2024

@author: velni
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

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
    
### SU(2) volume # 2*np.pi**2

import sympy as sp

x,t,f = sp.symbols("x t f")
arg = (sp.sin(x)**2)*sp.sin(t)
vol = sp.integrate(arg, (x, 0, np.pi), (t, 0, np.pi), (f, 0, 2*np.pi))

### Debug

alpha=1
i=1
j=1
a=2
b=1
r_index = 12
Q_index = 0
mod = 0
idx = [25,25,26]

### Coefficients

def A_0(alpha,i,j,r_index,mod):
    
    # Initialize value
    A = 0
    
    # Integrate over Q
    g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_{i}{j + 3*alpha}_r={r_index}.npy')
    A = np.sum(g)
    
    return A/vol

def A_ab(alpha,i,j,a,b,r_index,mod):
    
    # Initialize values
    A = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]
             
    # Integrate over Q
    g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_{i}{j + 3*alpha}_r={r_index}.npy')
    for Q_index in range(len(g)):
        A_s = 0
        R = np.zeros((3,3))
        for ap in range(3):
            for bp in range(3):
                R[ap][bp] = 0.5*np.trace(np.dot(sigma[ap],np.dot(Q_vals[Q_index],np.dot(sigma[bp],np.linalg.inv(Q_vals[Q_index])))))
        A_s = g[Q_index]*np.linalg.inv(R)[a-1][b-1]
        A = A + A_s
    
    return A

#

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
    delta = np.array([[1.,0.],[0.,1.]])
    
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
"""
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






#%% A^1_12;11

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

R = np.arange(0,61,1)

X = 1.731 + 0.1*np.arange(0,61,1)
Y = [A_ab(1,1,1,2,1,r,0) for r in R] 

plt.title(r'$A^{1}_{12;11}$')
plt.plot(X,Y,'k-')
plt.show()

#%% A^1_0;12

R = np.arange(0,61,1)

X = 1.731 + 0.1*np.arange(0,61,1)
Y = [A_0(1,1,2,r,0) for r in R] 

plt.plot(X,Y,'k-')
plt.show()















