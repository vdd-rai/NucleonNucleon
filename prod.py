import numpy as np
import random as rd

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

### r,Q sampling

r_vals = np.load('/home/velni/Escritorio/TFM/py/sample/r.npy')
Q_vals = np.load('/home/velni/Escritorio/TFM/py/sample/Q.npy')

### data for f(r) interpolation

data_sf0 = np.loadtxt(f'/home/velni/Escritorio/TFM/py/profile_f/data_sf0.txt')
data_sfm = np.loadtxt(f'/home/velni/Escritorio/TFM/py/profile_f/data_sfm.txt')
data_sf6 = np.loadtxt(f'/home/velni/Escritorio/TFM/py/profile_f/data_sf6.txt')

### product approximation

def U_S(y,r,Q,model):
    """ generates product approximation at point y=(y_1,y_2,y_3) for given r,Q and model
    """
    """ y=list, r=float, Q=np.array, model=str ('0', 'm', '6')
    """

    # parameters
    y1,y2,y3 = y
    xm = np.array([0,0,r/2.])
    pos_p = y+xm
    pos_m = y-xm

    ar_p = np.linalg.norm(y+xm)
    ar_m = np.linalg.norm(y-xm)

    I = np.array([[1,0],[0,1]])
    sigma = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]

    # f(r) interpolation
    if model == '0':
        rp = data_sf0[:,0]
        fp = data_sf0[:,1]
    if model == 'm':
        rp = data_sfm[:,0]
        fp = data_sfm[:,1]
    if model == '6':
        rp = data_sf6[:,0]
        fp = data_sf6[:,1]

    f_p = np.interp(ar_p,rp,fp)
    f_m = np.interp(ar_m,rp,fp)

    # U(1)
    phi1 = np.array([ np.cos(f_m), np.sin(f_m)*(pos_m[0])/ar_m, np.sin(f_m)*(pos_m[1])/ar_m, np.sin(f_m)*(pos_m[2])/ar_m])

    # U(2)
    dirs = [Q @ s @ np.linalg.inv(Q) for s in sigma]

    Z = y1*dirs[0] + y2*dirs[1] + (y3+r/2.)*dirs[2]

    dirs_param = [ 0.5*np.trace(Z), 0.5*np.trace(sigma[0].dot(Z)), 0.5*np.trace(sigma[1].dot(Z)), 0.5*np.trace(sigma[2].dot(Z)) ]

    phi2 = np.array( [np.cos(f_p)+dirs_param[0], np.sin(f_p)*(1./ar_p)*dirs_param[1], np.sin(f_p)*(1./ar_p)*dirs_param[2], np.sin(f_p)*(1./ar_p)*dirs_param[3]] )
    
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


### test
"""
r = r_vals[0]
Q = Q_vals[0]

y_test = np.array([rd.random(),rd.random(),r/2. + rd.random()])

U_test = U_S(y_test, r, Q, '0')
print('U_S({}) = {}' .format(y_test,U_test))
print('')
print('|U| = {}' .format(np.linalg.norm(U_test)))
"""
### output

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

def U_S_eval(r,Q,model):
    U_vals = np.zeros((len(y1),len(y2),len(y3),4))
    for idx in idx_list:
        for coord in range(4):
            U_vals[idx[0]][idx[1]][idx[2]][coord] = U_S(np.array([y1[idx[0]], y2[idx[1]], y3[idx[2]]]), r, Q, model)[coord]
    return U_vals

for r_idx,r in enumerate(r_vals):
    np.save(f'/home/velni/Escritorio/TFM/py/prod/data/m/U_S_r={r_idx}_Q=0',U_S_eval(r,Q_vals[0],'m'))
