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

### Potential

def V(r_index,Q_index,mod,m,c6):
    
    # Skyrme
    if mod == 0:
        d1 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d1/d1_U_r={r_index}_Q={Q_index}.npy')
        d2 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d2/d2_U_r={r_index}_Q={Q_index}.npy')
        d3 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d3/d3_U_r={r_index}_Q={Q_index}.npy')
        
        v1 =
        v2 = 
        v3 = 
        
        V = v1 + 0.5*(v2 + v3) 
    
    # Massive
    if mod == 1:
        d1 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/m/d1/d1_U_r={r_index}_Q={Q_index}.npy')
        d2 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/m/d2/d2_U_r={r_index}_Q={Q_index}.npy')
        d3 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/m/d3/d3_U_r={r_index}_Q={Q_index}.npy')
    
        v1 =
        v2 = 
        v3 = 
        v4 = 
        
        V = v1 + 0.5*(v2 + v3) + 2*(m**2)*v4
        
    # Generalized
    if mod == 6:
        d1 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/6/d1/d1_U_r={r_index}_Q={Q_index}.npy')
        d2 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/6/d2/d2_U_r={r_index}_Q={Q_index}.npy')
        d3 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/6/d3/d3_U_r={r_index}_Q={Q_index}.npy')
        
        v1 =
        v2 = 
        v3 = 
        v4 = 
        v5 = v1**3
        v6 = 
        v7 = v1*v3
        
        V = v1 + 0.5*(v2 + v3) + 2*(m**2)*v4 - 0.5*c6*( v5 - 2*v6 + 3*v7 )
        
    return V
    
    
    
### Coefficients

def A_0(alphap,ip,jp,r_index,mod):
    
    # Reasign indices
    i = ip-1
    j = jp-1
    alpha = alphap-1
    
    # Initialize value
    A = 0
    
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_r={r_index}_Q={Q_index}.npy')
        A_s = g[r_index][Q_index][i][j + 3*alpha] 
        A += A_s
        A_s = 0
    
    # Normalization
    V = (4./3)*np.pi
    
    return A/V

def A_ab(alphap,ip,jp,ap,bp,r_index,mod):
    
    # Reasign indices
    i = ip-1
    j = jp-1
    alpha = alphap-1
    a = ap-1
    b = bp-1
    
    # Initialize values
    A = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]
             
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_r={r_index}_Q={Q_index}.npy')
        R_ab = 0.5*np.trace(np.dot(sigma[a],np.dot(Q_vals[Q_index],np.dot(sigma[b],np.linalg.inv(Q_vals[Q_index])))))
        A_s = g[r_index][Q_index][i][j + 3*alpha]*(1./R_ab)
        A += A_s
        A_s = 0
    
    return A

#

def B_0(alphap,betap,ip,jp,L,r_index,mod):
    
    # Reasign indices
    i = ip-1
    j = jp-1
    alpha = alphap-1
    beta = betap-1
    
    # Initialize value
    B = 0
    delta = np.array([[1.,0.],[0.,1.]])
    
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_r={r_index}_Q={Q_index}.npy')
        B_s = g[r_index][Q_index][i + 3*alpha][j + 3*beta] - L*delta[alpha][beta]*delta[i][j]
        B += B_s
        B_s = 0
    
    # Normalization
    V = (4./3)*np.pi
    
    return B/V

def B_ab(alphap,betap,ip,jp,ap,bp,r_index,mod):
    
    # Reasign indices
    i = ip-1
    j = jp-1
    alpha = alphap-1
    beta = betap-1
    a = ap-1
    b = bp-1
    
    # Initialize values
    B = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]
             
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_r={r_index}_Q={Q_index}.npy')
        R_ab = 0.5*np.trace(np.dot(sigma[a],np.dot(Q_vals[Q_index],np.dot(sigma[b],np.linalg.inv(Q_vals[Q_index])))))
        B_s = g[r_index][Q_index][i + 3*alpha][j + 3*beta]*(1./R_ab)
        B += B_s
        B_s = 0
    
    return B

# 

def C_0(ip,jp,M,r_index,mod):
    
    # Reasign indices
    i = ip-1
    j = jp-1
    
    # Initialize value
    C = 0
    delta = np.array([[1.,0.],[0.,1.]])
    
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_r={r_index}_Q={Q_index}.npy')
        C_s = g[r_index][Q_index][i][j] - 0.5*M*delta[i][j]
        C += C_s
        C_s = 0
    
    # Normalization
    V = (4./3)*np.pi
    
    return C/(2*V)

def C_ab(ip,jp,ap,bp,M,r_index,mod):
    
    # Reasign indices
    i = ip-1
    j = jp-1
    a = ap-1
    b = bp-1
    
    # Initialize values
    C = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]
             
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        g = np.load(f'/home/velni/Escritorio/TFM/py/metric/data/{mod}/g_r={r_index}_Q={Q_index}.npy')
        R_ab = 0.5*np.trace(np.dot(sigma[a],np.dot(Q_vals[Q_index],np.dot(sigma[b],np.linalg.inv(Q_vals[Q_index])))))
        C_s = g[r_index][Q_index][i][j]*(1./R_ab)
        C += C_s
        C_s = 0
    
    return C

# 

def D_0(r_index,mod):
    
    # Initialize value
    D = 0
        
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        D_s = V(r_index,Q_index,mod,0,0)
        D += D_s
        D_s = 0
    
    # Normalization
    V = (4./3)*np.pi
    
    return D/(2*V)

def D_ab(r_index,mod):
    
    # Initialize values
    D = 0
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]
             
    # Integrate over Q
    for Q_index in range(len(Q_vals)):
        R_ab = 0.5*np.trace(np.dot(sigma[a],np.dot(Q_vals[Q_index],np.dot(sigma[b],np.linalg.inv(Q_vals[Q_index])))))
        D_s = V(r_index,Q_index,mod,0,0)*(1./R_ab)
        D += D_s
        D_s = 0
    
    return D/2.




















