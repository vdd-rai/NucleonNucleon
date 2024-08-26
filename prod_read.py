#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 11:35:26 2024

@author: velni
"""

import numpy as np
import random as rd
from numba import njit

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

### r,Q sampling

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')
Q_vals = np.load('/home/velni/Escritorio/TFM/py/sample/Q.npy')

### data of f(r) interpolation

f0_plus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/f0_plus.npy')
f0_minus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/f0_minus.npy')

fm_plus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/fm_plus.npy')
fm_minus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/fm_minus.npy')

f6_plus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/f6_plus.npy')
f6_minus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/f6_minus.npy')

### product approximation

def U_S(coord,r_index,Q1,Q2,model):
    """ generates product approximation at point y=(y_1,y_2,y_3) for given r,Q and model
    """
    """ y=list, r=float, Q=np.array, model=str ('0', 'm', '6')
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    # f(r) interpolation
    if model == 0:
        f_p = f0_plus[r_index][coord[0]][coord[1]][coord[2]]
        f_m = f0_minus[r_index][coord[0]][coord[1]][coord[2]]
    if model == 1:
        f_p = fm_plus[r_index][coord[0]][coord[1]][coord[2]]
        f_m = fm_minus[r_index][coord[0]][coord[1]][coord[2]]
    if model == 2:
        f_p = f6_plus[r_index][coord[0]][coord[1]][coord[2]]
        f_m = f6_minus[r_index][coord[0]][coord[1]][coord[2]]

    # U(1)
    dirs1 = [np.dot(np.dot(Q1,s),np.linalg.inv(Q1)) for s in sigma]

    Z1 = pos_m[0]*dirs1[0] + pos_m[1]*dirs1[1] + pos_m[2]*dirs1[2]

    dirs1_param = [ 0.5*np.trace(Z1), 0.5*np.trace(sigma[0].dot(Z1)), 0.5*np.trace(sigma[1].dot(Z1)), 0.5*np.trace(sigma[2].dot(Z1)) ]

    phi1 = np.array( [np.cos(f_m)+dirs1_param[0], np.sin(f_m)*(1./ar_m)*dirs1_param[1], np.sin(f_m)*(1./ar_m)*dirs1_param[2], np.sin(f_m)*(1./ar_m)*dirs1_param[3]] )
    
    # U(2)
    dirs2 = [np.dot(np.dot(Q2,s),np.linalg.inv(Q2)) for s in sigma]

    Z2 = pos_p[0]*dirs2[0] + pos_p[1]*dirs2[1] + pos_p[2]*dirs2[2]

    dirs2_param = [ 0.5*np.trace(Z2), 0.5*np.trace(sigma[0].dot(Z2)), 0.5*np.trace(sigma[1].dot(Z2)), 0.5*np.trace(sigma[2].dot(Z2)) ]

    phi2 = np.array( [np.cos(f_p)+dirs2_param[0], np.sin(f_p)*(1./ar_p)*dirs2_param[1], np.sin(f_p)*(1./ar_p)*dirs2_param[2], np.sin(f_p)*(1./ar_p)*dirs2_param[3]] )
    
    # U_S
    U_S = np.array([phi1[0]*phi2[0] - phi1[1]*phi2[1] - phi1[2]*phi2[2] - phi1[3]*phi2[3],
            phi1[0]*phi2[1] + phi1[1]*phi2[0],
            phi1[0]*phi2[2] + phi1[2]*phi2[0],
            phi1[0]*phi2[3] + phi1[3]*phi2[0]])

    # normalization
    C0 = (phi1[0]*phi2[0] - phi1[1]*phi2[1] - phi1[2]*phi2[2] - phi1[3]*phi2[3])**2
    Ck = (phi1[0]*phi2[1] + phi1[1]*phi2[0])**2 + (phi1[0]*phi2[2] + phi1[2]*phi2[0])**2 + (phi1[0]*phi2[3] + phi1[3]*phi2[0])**2
    
    N = np.sqrt(C0 + Ck)
    
    # output
    return np.array((1./N)*U_S)

### y grid

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3)):
            idx_list.append([i,j,k])
   
#%% test

r = r_vals[17]
Q = Q_vals[17]

I = np.array([[1.,0j],[0j,1.]])

import time

start_time = time.time()

U_test = U_S(idx_list[24089],17,I,Q,0)
print('U_S({}) = {}' .format(idx_list[24089],U_test))
print('')
print('|U| = {}' .format(np.linalg.norm(U_test)))

print()
print()
print("--- runtime : %s seconds ---" % (time.time() - start_time))

#%%% output single

I = np.array([[1.,0j],[0j,1.]])

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3)):
            idx_list.append([i,j,k])

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')
Q_vals = np.load('/home/velni/Escritorio/TFM/py/sample/Q.npy')

def U_S_eval(r_index,Q1,Q2,model):
    U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        for coord in range(4):
            U_vals[idx[0]][idx[1]][idx[2]][coord] = U_S(idx,r_index,Q1,Q2,model)[coord]
    return U_vals

r_index = 12
Q_index = 4

U = U_S_eval(r_index,I,Q_vals[Q_index],1)

plt.imshow(U[:,25,:,0],cmap='gray')
plt.show()


#%% output grid

I = np.array([[1.,0j],[0j,1.]])

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3)):
            idx_list.append([i,j,k])

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')
Q_vals = np.load('/home/velni/Escritorio/TFM/py/sample/Q.npy')

def U_S_eval(r_index,Q1,Q2,model):
    U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        for coord in range(4):
            U_vals[idx[0]][idx[1]][idx[2]][coord] = U_S(idx,r_index,Q1,Q2,model)[coord]
    return U_vals

Q_index = 0

for r_idx,r in enumerate(r_vals):
    np.save(f'/home/velni/Escritorio/TFM/py/prod/data/m/U_S_r={r_idx}_Q=0',U_S_eval(r_idx,I,Q_vals[Q_index],1))
