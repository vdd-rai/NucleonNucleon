#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:46:23 2024

@author: velni
"""

import numpy as np
import time

from terms import v1,v2,v3,v6

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

point_list = []
for i in range(len(r_vals)):
    for j in range(len(Q_vals)):
            point_list.append([i,j])

#%%

V1 = np.zeros((len(r_vals),len(Q_vals)))
V2 = np.zeros((len(r_vals),len(Q_vals)))
V3 = np.zeros((len(r_vals),len(Q_vals)))
V6 = np.zeros((len(r_vals),len(Q_vals)))

for point in point_list:
        
    start_time = time.time()
        
    d1 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d1/d1_U_r={point[0]}_Q={point[1]}.npy')
    d2 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d2/d2_U_r={point[0]}_Q={point[1]}.npy')
    d3 = np.load(f'/home/velni/Escritorio/TFM/py/deriv/data/0/d3/d3_U_r={point[0]}_Q={point[1]}.npy')
        
    for idx in idx_list:
        # V1[point[0]][point[1]] += v1.v1(idx,d1,d2,d3) # 1 second # DONE
        # V2[point[0]][point[1]] += v2.v2(idx,d1,d2,d3) # 15 seconds
        # V3[point[0]][point[1]] += v3.v3(idx,d1,d2,d3) # 20 seconds
        # V6[point[0]][point[1]] += v6.v6(idx,d1,d2,d3) # 280 seconds
            
    print('')
    print("--- step : %s seconds ---" % (time.time() - start_time))
            
###

# np.save('/home/velni/Escritorio/TFM/py/pot/data/V1', V1)# DONE
np.save('/home/velni/Escritorio/TFM/py/pot/data/V2', V2)
np.save('/home/velni/Escritorio/TFM/py/pot/data/V3', V3)
np.save('/home/velni/Escritorio/TFM/py/pot/data/V6', V6)






