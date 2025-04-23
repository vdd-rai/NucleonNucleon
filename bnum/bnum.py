import numpy as np
import random as rd
from numba import njit

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

from funcs import bf

### r,Q sampling

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')
Q_vals = np.load('/home/velni/phd/w/tfm/py/sample/Q.npy')

y1 = np.arange(-5., 5., 0.2)
y2 = np.arange(-5., 5., 0.2)
y3 = np.arange(-5., 5., 0.2)

coord_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3)):
            coord_list.append([i,j,k])

hD = 0.2 

### Baryon number

def B(r_idx,Q_idx,mod):
    
    # import fields and derivatives
    U = np.load(f'/home/velni/phd/w/tfm/py/prod/data/{mod}/U_S_r={r_idx}_Q={Q_idx}.npy')
    d1U = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d1/d1_U_r={r_idx}_Q={Q_idx}.npy')
    d2U = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d2/d2_U_r={r_idx}_Q={Q_idx}.npy')
    d3U = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d3/d3_U_r={r_idx}_Q={Q_idx}.npy')

    # integrate
    dens = np.zeros((len(y1),len(y2),len(y3)))
    for coord in coord_list:
        dens[*coord] = bf.B_expanded(coord,U,d1U,d2U,d3U)

    B = (-1/(24*np.pi**2))*np.sum(dens)*(hD**3)

    # output
    return B,dens

### spatial distribution
"""
Q_idx = 5
den_r = [B(r_idx,Q_idx,0)[1] for r_idx in range(10,20)]

f = plt.figure()
sf = f.add_subplot(1,1,1)
	
for r_idx in range(10,15):
    sf.clear()
    sf.set_xlim([-5,5])
    sf.set_ylim([-5,5])
    sf.plot(np.arange(-5,5,0.2),den_r[:,25,25],'r-')
	
    plt.grid(True)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mathcal{B}(x)$')
    plt.title(f'$r={r_idx}$')
    plt.pause(0.01)

plt.show()
"""
### Parametric evolution

Bnums = [[B(r_idx,Q_idx,0)[0] for r_idx in range(0,20)] for Q_idx in range(0,5)]
#Bnums = [B(r_idx,15,0)[0] for r_idx in range(0,20)]

print(Bnums)
"""
plt.plot(np.arange(0,20,1),Bnums,'r-')
	
plt.xlabel(r'$r$')
plt.ylabel(r'$\mathcal{B}(r,Q=15)$')

plt.show()
"""








