#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:31:05 2025

@author: velni
"""

import numpy as np
import scipy as sci

###

Q_vals = np.load('/home/velni/phd/w/tfm/py/sample/Q.npy')

Q = Q_vals[8]

I = np.array([[1.,0j],[0j,1.]])
sigma = [np.array([[0j,1.],[1.,0j]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0j],[0j,-1.]])]

M1 = sigma[0]
M2 = sigma[1]
M3 = sigma[2]

###

hD = 0.01

arg = -0.5*hD

x1 = arg*M1
x2 = arg*M2
x3 = arg*M3

### yes # np.exp calcula a exponencial ELEMENTO A ELEMENTO

X1 = sci.linalg.expm(1j*x1)
X2 = sci.linalg.expm(1j*x2)
X3 = sci.linalg.expm(1j*x3)

yes1 = np.dot(Q,X1)
yes2 = np.dot(Q,X2)
yes3 = np.dot(Q,X3)

### no

Y1 = np.cos(arg)*I + 1j*np.sin(arg)*M1
Y2 = np.cos(arg)*I + 1j*np.sin(arg)*M2
Y3 = np.cos(arg)*I + 1j*np.sin(arg)*M3

no1 = np.dot(Q,Y1)
no2 = np.dot(Q,Y2)
no3 = np.dot(Q,Y3)








