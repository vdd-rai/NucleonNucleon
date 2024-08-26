#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:15:53 2024

@author: velni
"""

import numpy as np
from numba import njit

### read f(r) values

# plus / minus : x sign
# p / m : hD sign

fm_plus_p_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/plus/fm_plus_p_y1.npy')
fm_plus_m_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/plus/fm_plus_m_y1.npy')
fm_minus_p_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/minus/fm_minus_p_y1.npy') 
fm_minus_m_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/minus/fm_minus_m_y1.npy') 

fm_plus_p_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/plus/fm_plus_p_y2.npy')
fm_plus_m_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/plus/fm_plus_m_y2.npy')
fm_minus_p_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/minus/fm_minus_p_y2.npy')
fm_minus_m_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/minus/fm_minus_m_y2.npy')

fm_plus_p_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/plus/fm_plus_p_y3.npy')
fm_plus_m_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/plus/fm_plus_m_y3.npy')
fm_minus_p_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/minus/fm_minus_p_y3.npy')
fm_minus_m_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/minus/fm_minus_m_y3.npy')
"""
fm_plus_p_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/plus/fm_plus_p_x1.npy')
fm_plus_m_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/plus/fm_plus_m_x1.npy')
fm_minus_p_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/minus/fm_minus_p_x1.npy') 
fm_minus_m_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/minus/fm_minus_m_x1.npy') 

fm_plus_p_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/plus/fm_plus_p_x2.npy')
fm_plus_m_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/plus/fm_plus_m_x2.npy')
fm_minus_p_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/minus/fm_minus_p_x2.npy')
fm_minus_m_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/minus/fm_minus_m_x2.npy')

fm_plus_p_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/plus/fm_plus_p_x3.npy')
fm_plus_m_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/plus/fm_plus_m_x3.npy')
fm_minus_p_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/minus/fm_minus_p_x3.npy')
fm_minus_m_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/minus/fm_minus_m_x3.npy')
"""
#

I = np.array([[1.,0j],[0j,1.]])
sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

hD = 0.01

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

# d1

@njit
def d1U_p_mass(coord,r_index,Q1,Q2,hD):
    """ Compute product approximation in +hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]]+hD,y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = fm_plus_p_y1[r_index][coord[0]][coord[1]][coord[2]]
    f_m = fm_minus_p_y1[r_index][coord[0]][coord[1]][coord[2]]

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
def d1U_m_mass(coord,r_index,Q1,Q2,hD):
    """ Compute product approximation in -hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]]-hD,y2[coord[1]],y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = fm_plus_m_y1[r_index][coord[0]][coord[1]][coord[2]]
    f_m = fm_minus_m_y1[r_index][coord[0]][coord[1]][coord[2]]

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

def d1_U_mass(r_index,Q1,Q2,hD):
    d1U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        d1U_vals[idx[0]][idx[1]][idx[2]] = (1./(2*hD))*( d1U_p_mass(idx,r_index,Q1,Q2,hD) - d1U_m_mass(idx,r_index,Q1,Q2,hD) )
    return d1U_vals

# d2

@njit
def d2U_p_mass(coord,r_index,Q1,Q2,hD):
    """ Compute product approximation in +hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]]+hD,y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = fm_plus_p_y2[r_index][coord[0]][coord[1]][coord[2]]
    f_m = fm_minus_p_y2[r_index][coord[0]][coord[1]][coord[2]]

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
def d2U_m_mass(coord,r_index,Q1,Q2,hD):
    """ Compute product approximation in -hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]]-hD,y3[coord[2]] ])
    xm = np.array([0,0,r_vals[r_index]/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = fm_plus_m_y2[r_index][coord[0]][coord[1]][coord[2]]
    f_m = fm_minus_m_y2[r_index][coord[0]][coord[1]][coord[2]]

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

def d2_U_mass(r_index,Q1,Q2,hD):
    d2U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        d2U_vals[idx[0]][idx[1]][idx[2]] = (1./(2*hD))*( d2U_p_mass(idx,r_index,Q1,Q2,hD) - d2U_m_mass(idx,r_index,Q1,Q2,hD) )
    return d2U_vals

# d3

@njit
def d3U_p_mass(coord,r_index,Q1,Q2,hD):
    """ Compute product approximation in +hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]]+hD ])
    xm = np.array([0,0,r_vals[r_index]/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = fm_plus_p_y3[r_index][coord[0]][coord[1]][coord[2]]
    f_m = fm_minus_p_y3[r_index][coord[0]][coord[1]][coord[2]]

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
def d3U_m_mass(coord,r_index,Q1,Q2,hD):
    """ Compute product approximation in -hD displaced
    """
    
    # matrices
    I = np.array([[1.,0j],[0j,1.]])
    sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

    # parameters
    y = np.array([ y1[coord[0]],y2[coord[1]],y3[coord[2]]-hD ])
    xm = np.array([0,0,r_vals[r_index]/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    f_p = fm_plus_m_y3[r_index][coord[0]][coord[1]][coord[2]]
    f_m = fm_minus_m_y3[r_index][coord[0]][coord[1]][coord[2]]

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

def d3_U_mass(r_index,Q1,Q2,hD):
    d3U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        d3U_vals[idx[0]][idx[1]][idx[2]] = (1./(2*hD))*( d3U_p_mass(idx,r_index,Q1,Q2,hD) - d3U_m_mass(idx,r_index,Q1,Q2,hD) )
    return d3U_vals

#%%% output single

import matplotlib.pyplot as plt

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

r_index = 12
Q_index = 4

U = d1_U_mass(r_index,I,Q_vals[Q_index],hD)

plt.imshow(U[:,25,:,0],cmap='gray')
plt.show()

#%% output

import time

start_time = time.time()

#r_index = 0
Q_index = 0

for r_index in range(len(r_vals)):
    np.save(f'/home/velni/Escritorio/TFM/py/deriv/data/m/d1/d1_U_mass_r={r_index}_Q={Q_index}', d1_U_mass(r_index,I,Q_vals[Q_index],hD))
    np.save(f'/home/velni/Escritorio/TFM/py/deriv/data/m/d2/d2_U_mass_r={r_index}_Q={Q_index}', d2_U_mass(r_index,I,Q_vals[Q_index],hD))
    np.save(f'/home/velni/Escritorio/TFM/py/deriv/data/m/d3/d3_U_mass_r={r_index}_Q={Q_index}', d3_U_mass(r_index,I,Q_vals[Q_index],hD))


print()
print()
print("--- runtime : %s seconds ---" % (time.time() - start_time))

#%%% check

import matplotlib.pyplot as plt

U = np.load('/home/velni/Escritorio/TFM/py/deriv/data/m/d1/d1_U_mass_r=23_Q=0.npy')

plt.imshow(U[:,25,:,0],cmap='gray')



