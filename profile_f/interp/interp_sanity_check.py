#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 17:54:24 2024

@author: velni
"""

import numpy as np
import matplotlib.pyplot as plt

#%% 

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')

data_sf0 = np.loadtxt('/home/velni/Escritorio/TFM/py/profile_f/data_sf0.txt')
data_sfm = np.loadtxt('/home/velni/Escritorio/TFM/py/profile_f/data_sfm.txt')
data_sf6 = np.loadtxt('/home/velni/Escritorio/TFM/py/profile_f/data_sf6.txt')

#%% read f(r) values

# plus / minus : x sign
# p / m : hD sign

f0_plus_p_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d1/plus/f0_plus_p_y1.npy')
f0_plus_m_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d1/plus/f0_plus_m_y1.npy')
f0_minus_p_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d1/minus/f0_minus_p_y1.npy') 
f0_minus_m_y1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d1/minus/f0_minus_m_y1.npy') 

f0_plus_p_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d2/plus/f0_plus_p_y2.npy')
f0_plus_m_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d2/plus/f0_plus_m_y2.npy')
f0_minus_p_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d2/minus/f0_minus_p_y2.npy')
f0_minus_m_y2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d2/minus/f0_minus_m_y2.npy')

f0_plus_p_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d3/plus/f0_plus_p_y3.npy')
f0_plus_m_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d3/plus/f0_plus_m_y3.npy')
f0_minus_p_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d3/minus/f0_minus_p_y3.npy')
f0_minus_m_y3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/d3/minus/f0_minus_m_y3.npy')

f0_plus_p_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D1/plus/f0_plus_p_x1.npy')
f0_plus_m_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D1/plus/f0_plus_m_x1.npy')
f0_minus_p_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D1/minus/f0_minus_p_x1.npy') 
fm_minus_m_x1 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/minus/fm_minus_m_x1.npy') 

f0_plus_p_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D2/plus/f0_plus_p_x2.npy')
f0_plus_m_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D2/plus/f0_plus_m_x2.npy')
f0_minus_p_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D2/minus/f0_minus_p_x2.npy')
f0_minus_m_x2 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D2/minus/f0_minus_m_x2.npy')

f0_plus_p_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D3/plus/f0_plus_p_x3.npy')
f0_plus_m_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D3/plus/f0_plus_m_x3.npy')
f0_minus_p_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D3/minus/f0_minus_p_x3.npy')
f0_minus_m_x3 = np.load('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/D3/minus/f0_minus_m_x3.npy')

#%%

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

hD = 0.01

idx_list = []
for i in range(len(y1)):
	for j in range(len(y2)):
       		for k in range(len(y3)):
            		idx_list.append([i,j,k])

plt.plot(data_sf0[:,0],data_sf0[:,1],'r-',label=r'$\mathcal{L}_S$')
plt.plot(data_sfm[:,0],data_sfm[:,1],'g-',label=r'$\mathcal{L}_M$')
plt.plot(data_sf6[:,0],data_sf6[:,1],'b-',label=r'$\mathcal{L}_G$')

plt.legend()

s = 75890
# CHANGE X TO MATCH f0
X = [np.linalg.norm( np.array([y1[idx_list[s][0]]+hD,y2[idx_list[s][1]],y3[idx_list[s][2]]]) + np.array([0.,0.,r])/2.) for r in r_vals]
plt.plot(X,f0_plus_p_y1[:,idx_list[s][0],idx_list[s][1],idx_list[s][2]],'k.')

ex = np.linalg.norm( np.array([y1[idx_list[23][0]]+hD,y2[idx_list[23][1]],y3[idx_list[23][2]]]) + np.array([0.,0.,2.27])/2. )
plt.plot(ex, np.interp(ex,data_sfm[:,0],data_sfm[:,1] ),'k*')

plt.show()
