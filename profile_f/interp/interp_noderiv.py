import numpy as np
import matplotlib.pyplot as plt
import time

###

data_sf0 = np.loadtxt('/home/velni/phd/w/tfm/py/profile_f/data_sf0.txt')
data_sfm = np.loadtxt('/home/velni/phd/w/tfm/py/profile_f/data_sfm.txt')
data_sf6 = np.loadtxt('/home/velni/phd/w/tfm/py/profile_f/data_sf6.txt')

### data points

r0 = data_sf0[:,0]
f0 = data_sf0[:,1]

rm = data_sfm[:,0]
fm = data_sfm[:,1]

r6 = data_sf6[:,0]
f6 = data_sf6[:,1]

### sample points

y1 = np.load('/home/velni/phd/w/tfm/py/sample/y1.npy')
y2 = np.load('/home/velni/phd/w/tfm/py/sample/y2.npy')
y3 = np.load('/home/velni/phd/w/tfm/py/sample/y3.npy')

l1 = len(y1)
l2 = len(y2)
l3 = len(y3[0])

hD = 0.01

### arguments for f(r)

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')

# 3D grid

idx_list = []
for i in range(l1):
    for j in range(l2):
        for k in range(l3):
            idx_list.append([i,j,k])

############################### interpolation  

### 0

f0_plus = [] 
for r_idx,r in enumerate(r_vals): # for each r
    matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
    temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
    for f_idx,idx in enumerate(idx_list): # run over indices
        matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
    f0_plus.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/noderiv/f0_plus', f0_plus)

f0_minus = [] 
for r_idx,r in enumerate(r_vals): # for each r
    matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
    temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
    for f_idx,idx in enumerate(idx_list): # run over indices
        matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
    f0_minus.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/noderiv/f0_minus', f0_minus)


### m
"""
fm_plus = [] 
for r_idx,r in enumerate(r_vals): # for each r
    matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
    temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
    for f_idx,idx in enumerate(idx_list): # run over indices
        matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
   fm_plus.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/noderiv/fm_plus', fm_plus)

fm_minus = [] 
for r_idx,r in enumerate(r_vals): # for each r
    matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
    temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
    for f_idx,idx in enumerate(idx_list): # run over indices
        matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
    fm_minus.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/noderiv/fm_minus', fm_minus)
"""
### 6
"""
f6_plus = [] 
for r_idx,r in enumerate(r_vals): # for each r
    matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
    temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),r6,f6 ) for idx in idx_list] # interpolat>
    for f_idx,idx in enumerate(idx_list): # run over indices
        matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
    f6_plus.append(matrix_f) # save matrix for r value
 np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/noderiv/f6_plus', f6_plus)

f6_minus = [] 
for r_idx,r in enumerate(r_vals): # for each r
    matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
    temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx]idx[2]]]) - np.array([0.,0.,r])/2. ),r6,f6 ) for idx in idx_list] # interpolat>
    for f_idx,idx in enumerate(idx_list): # run over indices
        matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
    f6_minus.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/noderiv/f6_minus', f6_minus)
"""








