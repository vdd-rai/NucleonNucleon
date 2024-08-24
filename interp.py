import numpy as np
import matplotlib.pyplot as plt
import time

###

data_sf0 = np.loadtxt('/home/velni/Escritorio/TFM/py/profile_f/data_sf0.txt')
data_sfm = np.loadtxt('/home/velni/Escritorio/TFM/py/profile_f/data_sfm.txt')
data_sf6 = np.loadtxt('/home/velni/Escritorio/TFM/py/profile_f/data_sf6.txt')

### data points

r0 = data_sf0[:,0]
f0 = data_sf0[:,1]

rm = data_sfm[:,0]
fm = data_sfm[:,1]

r6 = data_sf6[:,0]
f6 = data_sf6[:,1]

### sample points

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

hD = 0.01

""" UNUSED
disp_y1 = np.array([[y-hD, y+hD] for y in y1])
disp_y2 = np.array([[y-hD, y+hD] for y in y2])
disp_y3 = np.array([[y-hD, y+hD] for y in y3])
"""

### arguments for f(r)

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')

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
for i in range(len(y1)):
	for j in range(len(y2)):
       		for k in range(len(y3)):
            		idx_list.append([i,j,k])

###############################3 interpolation : massive model
"""
##### y derivatives

# f(y+x/2) : y1 derivative

fm_plus_p_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]+hD,y2[idx[1]],y3[idx[2]]]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_p_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/plus/fm_plus_p_y1', fm_plus_p_y1)

fm_plus_m_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]-hD,y2[idx[1]],y3[idx[2]]]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_m_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/plus/fm_plus_m_y1', fm_plus_m_y1)

# f(y-x/2) : y1 derivative

fm_minus_p_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]+hD,y2[idx[1]],y3[idx[2]]]) - np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_p_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/minus/fm_minus_p_y1', fm_minus_p_y1)

fm_minus_m_y1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]]-hD,y2[idx[1]],y3[idx[2]]]) - np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_m_y1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d1/minus/fm_minus_m_y1', fm_minus_m_y1)

# f(y+x/2) : y2 derivative

fm_plus_p_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]+hD,y3[idx[2]]]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_p_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/plus/fm_plus_p_y2', fm_plus_p_y2)

fm_plus_m_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]-hD,y3[idx[2]]]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_m_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/plus/fm_plus_m_y2', fm_plus_m_y2)

# f(y-x/2) : y2 derivative

fm_minus_p_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]+hD,y3[idx[2]]]) - np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_p_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/minus/fm_minus_p_y2', fm_minus_p_y2)

fm_minus_m_y2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]]-hD,y3[idx[2]]]) - np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_m_y2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d2/minus/fm_minus_m_y2', fm_minus_m_y2)

# f(y+x/2) : y3 derivative

fm_plus_p_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]+hD]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_p_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/plus/fm_plus_p_y3', fm_plus_p_y3)

fm_plus_m_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]-hD]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_m_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/plus/fm_plus_m_y3', fm_plus_m_y3)

# f(y-x/2) : y3 derivative

fm_minus_p_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]+hD]) - np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_p_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/minus/fm_minus_p_y3', fm_minus_p_y3)

fm_minus_m_y3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]-hD]) - np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_m_y3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/d3/minus/fm_minus_m_y3', fm_minus_m_y3)

##### x derivatives

# f(y+x/2) : x1 derivative

fm_plus_p_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) + np.array([hD,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_p_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/plus/fm_plus_p_x1', fm_plus_p_x1)

fm_plus_m_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) + np.array([-hD,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_m_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/plus/fm_plus_m_x1', fm_plus_m_x1)

# f(y-x/2) : x1 derivative

fm_minus_p_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) - np.array([hD,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_p_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/minus/fm_minus_p_x1', fm_minus_p_x1)

fm_minus_m_x1 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) - np.array([-hD,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_m_x1.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D1/minus/fm_minus_m_x1', fm_minus_m_x1)

# f(y+x/2) : x2 derivative

fm_plus_p_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) + np.array([0.,hD,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_p_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/plus/fm_plus_p_x2', fm_plus_p_x2)

fm_plus_m_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) + np.array([0.,-hD,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_m_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/plus/fm_plus_m_x2', fm_plus_m_x2)

# f(y-x/2) : x2 derivative

fm_minus_p_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) - np.array([0.,hD,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_p_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/minus/fm_minus_p_x2', fm_minus_p_x2)

fm_minus_m_x2 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) - np.array([0.,-hD,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_m_x2.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D2/minus/fm_minus_m_x2', fm_minus_m_x2)

# f(y+x/2) : x3 derivative

fm_plus_p_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) + np.array([0.,0.,r+hD])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_p_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/plus/fm_plus_p_x3', fm_plus_p_x3)

fm_plus_m_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) + np.array([0.,0.,r-hD])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_plus_m_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/plus/fm_plus_m_x3', fm_plus_m_x3)

# f(y-x/2) : x3 derivative

fm_minus_p_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) - np.array([0.,0.,r+hD])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_p_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/minus/fm_minus_p_x3', fm_minus_p_x3)

fm_minus_m_x3 = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) - np.array([0.,0.,r-hD])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_minus_m_x3.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/D3/minus/fm_minus_m_x3', fm_minus_m_x3)
"""
##### Q derivatives (no displacement)

fm_Q = [] # [i] value of r : [j] interpolated value of f in (y1[idx[0]],y2[idx[1]],y3[idx[2]])
for r in r_vals: # for each r
        matrix_f = np.zeros((50,50,50)) # generate a 3D matrix of zeros to store f values
        temp_f = [np.interp( np.linalg.norm( np.array([y1[idx[0]],y2[idx[1]],y3[idx[2]]]) + np.array([0.,0.,r])/2. ),rm,fm ) for idx in idx_list] # interpolat>
        for f_idx,idx in enumerate(idx_list): # run over indices
                matrix_f[idx[0]][idx[1]][idx[2]] = temp_f[f_idx] # save each f value on the corresponding 3D point
        fm_Q.append(matrix_f) # save matrix for r value
np.save('/home/velni/Escritorio/TFM/py/profile_f/interp/data/m/DQ/fm_Q', fm_Q)











