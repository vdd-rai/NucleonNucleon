#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:34:42 2024

@author: velni
"""

import numpy as np
from numba import njit

### read f(r) values

# plus / minus : x sign
# p / m : hD sign

f_plus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/f0_plus.npy')
f_minus = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/no_deriv/f0_minus.npy')

I = np.array([[1.,0j],[0j,1.]])
sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

hD = 0.5

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')
Q_vals = np.load('/home/velni/Escritorio/TFM/py/sample/Q.npy')

### y grid

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3)):
            idx_list.append([i,j,k])

### product approximation for derivatives

# D4

@njit
def D4U_p(coord,r_index,Q1p,Q2,hD):
    """ Compute product approximation in +hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]])/2.
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = f_plus[r_index][coord[0]][coord[1]][coord[2]]
    f_m = f_minus[r_index][coord[0]][coord[1]][coord[2]]

    Q1 = np.dot(Q1p, np.exp(-1j*0.5*hD*sigma[0]))

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
    return (1./N)*U_S

@njit
def D4U_m(coord,r_index,Q1p,Q2,hD):
    """ Compute product approximation in -hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]])/2.
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = f_plus[r_index][coord[0]][coord[1]][coord[2]]
    f_m = f_minus[r_index][coord[0]][coord[1]][coord[2]]
    
    Q1 = np.dot(Q1p, np.exp(1j*0.5*hD*sigma[0]))

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
    return (1./N)*U_S

def D4_U(r_index,Q1p,Q2,hD):
    D4U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        D4U_vals[idx[0]][idx[1]][idx[2]] = (1./(2*hD))*( D4U_p(idx,r_index,Q1p,Q2,hD) - D4U_m(idx,r_index,Q1p,Q2,hD) )
    return D4U_vals

# D5

@njit
def D5U_p(coord,r_index,Q1p,Q2,hD):
    """ Compute product approximation in +hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]])/2.
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = f_plus[r_index][coord[0]][coord[1]][coord[2]]
    f_m = f_minus[r_index][coord[0]][coord[1]][coord[2]]
    
    # Q1 = np.dot(Q1p, np.exp(-1j*0.5*hD*sigma[1]))
    
    x = -1j*0.5*hD*sigma[1]
    ex = I + x + 0.5*np.dot(x,x) + (1./6)*np.dot(x,np.dot(x,x)) + (1./24)*np.dot(x,np.dot(x,np.dot(x,x)))
    Q1 = np.dot(I,ex)

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
    return (1./N)*U_S

@njit
def D5U_m(coord,r_index,Q1p,Q2,hD):
    """ Compute product approximation in -hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]])/2.
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = f_plus[r_index][coord[0]][coord[1]][coord[2]]
    f_m = f_minus[r_index][coord[0]][coord[1]][coord[2]]
    
    # Q1 = np.dot(Q1p, np.exp(-1j*0.5*hD*sigma[1]))
    
    x = 1j*0.5*hD*sigma[1]
    ex = I + x + 0.5*np.dot(x,x) + (1./6)*np.dot(x,np.dot(x,x)) + (1./24)*np.dot(x,np.dot(x,np.dot(x,x)))
    Q1 = np.dot(I,ex)

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
    return (1./N)*U_S

def D5_U(r_index,Q1p,Q2,hD):
    D5U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        D5U_vals[idx[0]][idx[1]][idx[2]] = (1./(2*hD))*( D5U_p(idx,r_index,Q1p,Q2,hD) - D5U_m(idx,r_index,Q1p,Q2,hD) )
    return D5U_vals

# D6

@njit
def D6U_p(coord,r_index,Q1p,Q2,hD):
    """ Compute product approximation in +hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]])/2.
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = f_plus[r_index][coord[0]][coord[1]][coord[2]]
    f_m = f_minus[r_index][coord[0]][coord[1]][coord[2]]
    
    # Q1 = np.dot(Q1p, np.exp(-1j*0.5*hD*sigma[2]))
    
    x = -1j*0.5*hD*sigma[2]
    ex = I + x + 0.5*np.dot(x,x) + (1./6)*np.dot(x,np.dot(x,x)) + (1./24)*np.dot(x,np.dot(x,np.dot(x,x)))
    Q1 = np.dot(I,ex)

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
    return (1./N)*U_S

@njit
def D6U_m(coord,r_index,Q1p,Q2,hD):
    """ Compute product approximation in -hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]])/2.
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = f_plus[r_index][coord[0]][coord[1]][coord[2]]
    f_m = f_minus[r_index][coord[0]][coord[1]][coord[2]]
    
    # Q1 = np.dot(Q1p, np.exp(1j*0.5*hD*sigma[2]))
    
    x = 1j*0.5*hD*sigma[2]
    ex = I + x + 0.5*np.dot(x,x) + (1./6)*np.dot(x,np.dot(x,x)) + (1./24)*np.dot(x,np.dot(x,np.dot(x,x)))
    Q1 = np.dot(I,ex)

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
    return (1./N)*U_S

def D6_U(r_index,Q1p,Q2,hD):
    D6U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        D6U_vals[idx[0]][idx[1]][idx[2]] = (1./(2*hD))*( D6U_p(idx,r_index,Q1p,Q2,hD) - D6U_m(idx,r_index,Q1p,Q2,hD) )
    return D6U_vals

#%%% output single

import matplotlib.pyplot as plt

hD = 0.5

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

r_index = 18
Q_index = 22

U = D4_U(r_index,I,Q_vals[Q_index],hD)

plt.imshow(U[:,25,:,0],cmap='gray')
plt.show()

#%% output

point_list = []
for i in range(len(r_vals)):
    for j in range(len(Q_vals)):
        point_list.append([i,j])

import time

start_time = time.time()

# r_index = 0
# Q_index = 0

hD4 = 0.5
hD5 = 0.5
hD6 = 0.5

for point in point_list:
    np.save(f'/home/velni/Escritorio/TFM/py/deriv/data/0/D4/D4_U_r={point[0]}_Q=-{point[1]}', D4_U(point[0],I,-Q_vals[point[1]],hD4))
    # np.save(f'/home/velni/Escritorio/TFM/py/deriv/data/0/D5/D5_U_r={point[0]}_Q={point[1]}', D5_U(point[0],I,-Q_vals[abs(point[1])],hD5))
    # np.save(f'/home/velni/Escritorio/TFM/py/deriv/data/0/D6/D6_U_r={r_index}_Q={Q_index}', D6_U(r_index,I,-Q_vals[abs(point[1])],hD6))


print()
print()
print("--- runtime : %s seconds ---" % (time.time() - start_time))

#%%% check

import matplotlib.pyplot as plt

U = np.load('/home/velni/Escritorio/TFM/py/deriv/data/0/D5/D5_U_r=5_Q=0.npy')

plt.imshow(U[:,25,:,0],cmap='gray')

#%%

for r_index in range(len(r_vals)):
    np.save(f'/home/velni/Escritorio/TFM/py/deriv/data/0/D4/D4_U_r={r_index}_Q=-0', D4_U(r_index,I,-Q_vals[0],hD4))
    
