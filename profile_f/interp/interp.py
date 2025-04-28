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

""" UNUSED
disp_y1 = np.array([[y-hD, y+hD] for y in y1])
disp_y2 = np.array([[y-hD, y+hD] for y in y2])
disp_y3 = np.array([[y-hD, y+hD] for y in y3])
"""

### arguments for f(r)

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')

""" UNUSED
# arg for each dimension

arg1 = disp_y1 # not affected by x

arg2 = disp_y2 # not affected by x

arg3_p = [] # [i][j][k] : [i] value of r : [j] base point : [k] displaced point
for r in r_vals:
	temp_p = []
	for base in range(len(y3)):
		temp_p.append([(disp_y3[base][0]+r)/2., (disp_y3[base][1]+r)/2.])
	arg3_p.append(temp_p)

arg3_m = [] # [i][j][k] : [i] value of r : [j] base point : [k] displaced point
for r in r_vals:
        temp_m = []
        for base in range(len(y3)):
                temp_m.append([disp_y3[base][0]-(r/2), disp_y3[base][1]-(r/2)])
        arg3_m.append(temp_m)
"""

# 3D grid

idx_list = []
for i in range(l1):
    for j in range(l2):
       	for k in range(l3):
            idx_list.append([i,j,k])

###############################3 interpolation : 0 model

##### y derivatives
"""
# f(y+x/2) : y1 derivative

f0_plus_p_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]+hD,y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_p_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d1/f0_plus_p_y1', f0_plus_p_y1)

f0_plus_m_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]-hD,y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_m_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d1/f0_plus_m_y1', f0_plus_m_y1)

# f(y-x/2) : y1 derivative

f0_minus_p_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]+hD,y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_p_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d1/f0_minus_p_y1', f0_minus_p_y1)

f0_minus_m_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]-hD,y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_m_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d1/f0_minus_m_y1', f0_minus_m_y1)

# f(y+x/2) : y2 derivative

f0_plus_p_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]+hD,y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_p_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d2/f0_plus_p_y2', f0_plus_p_y2)

f0_plus_m_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]-hD,y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_m_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d2/f0_plus_m_y2', f0_plus_m_y2)

# f(y-x/2) : y2 derivative

f0_minus_p_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]+hD,y3[r_idx][idx[2]]]) - np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_p_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d2/f0_minus_p_y2', f0_minus_p_y2)

f0_minus_m_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]-hD,y3[r_idx][idx[2]]]) - np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_m_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d2/f0_minus_m_y2', f0_minus_m_y2)

# f(y+x/2) : y3 derivative

f0_plus_p_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]+hD]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_p_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d3/f0_plus_p_y3', f0_plus_p_y3)

f0_plus_m_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]-hD]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_m_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d3/f0_plus_m_y3', f0_plus_m_y3)

# f(y-x/2) : y3 derivative

f0_minus_p_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]+hD]) - np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_p_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d3/f0_minus_p_y3', f0_minus_p_y3)

f0_minus_m_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]-hD]) - np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_m_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/d3/f0_minus_m_y3', f0_minus_m_y3)

##### x derivatives

# f(y+x/2) : x1 derivative

f0_plus_p_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([hD,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_p_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D1/f0_plus_p_x1', f0_plus_p_x1)

f0_plus_m_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([-hD,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_m_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D1/f0_plus_m_x1', f0_plus_m_x1)

# f(y-x/2) : x1 derivative

f0_minus_p_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([hD,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_p_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D1/f0_minus_p_x1', f0_minus_p_x1)

f0_minus_m_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([-hD,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_m_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D1/f0_minus_m_x1', f0_minus_m_x1)

# f(y+x/2) : x2 derivative

f0_plus_p_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,hD,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_p_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D2/f0_plus_p_x2', f0_plus_p_x2)
"""
f0_plus_m_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,-hD,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_m_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D2/f0_plus_m_x2', f0_plus_m_x2)

# f(y-x/2) : x2 derivative

f0_minus_p_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,hD,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_p_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D2/f0_minus_p_x2', f0_minus_p_x2)

f0_minus_m_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,-hD,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_m_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D2/f0_minus_m_x2', f0_minus_m_x2)

# f(y+x/2) : x3 derivative

f0_plus_p_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r+hD])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_p_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D3/f0_plus_p_x3', f0_plus_p_x3)

f0_plus_m_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r-hD])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_plus_m_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D3/f0_plus_m_x3', f0_plus_m_x3)

# f(y-x/2) : x3 derivative

f0_minus_p_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,0.,r+hD])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_p_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D3/f0_minus_p_x3', f0_minus_p_x3)

f0_minus_m_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((l1,l2,l3)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) - np.array([0.,0.,r-hD])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        f0_minus_m_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/phd/w/tfm/py/profile_f/interp/data/0/D3/f0_minus_m_x3', f0_minus_m_x3)

##### Q derivatives (no displacement) DO NOT USE
"""
fm_Q = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r_idx,r in enumerate(r_vals): # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[r_idx][idx[2]]]) + np.array([0.,0.,r])/2. ),r0,f0 ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_Q.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/0/DQ/f0_Q', f0_Q)
"""









