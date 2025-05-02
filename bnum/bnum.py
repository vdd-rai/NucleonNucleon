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

y1 = np.load('/home/velni/phd/w/tfm/py/sample/y1.npy')
y2 = np.load('/home/velni/phd/w/tfm/py/sample/y2.npy')
y3 = np.load('/home/velni/phd/w/tfm/py/sample/y3.npy')

dy1 = np.load('/home/velni/phd/w/tfm/py/sample/dy1.npy')
dy2 = np.load('/home/velni/phd/w/tfm/py/sample/dy2.npy')
dy3 = np.load('/home/velni/phd/w/tfm/py/sample/dy3.npy')

idx_list = []
for i in range(len(y1)):
    for j in range(len(y2)):
        for k in range(len(y3[0])):
            idx_list.append([i,j,k])

hD = 0.2

### Baryon number

def Bdens(r_idx,Q_idx,mod):
    
    # import fields and derivatives
    U = np.load(f'/home/velni/phd/w/tfm/py/prod/data/{mod}/U_S_r={r_idx}_Q={Q_idx}.npy')
    d1U = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d1/d1_U_r={r_idx}_Q={Q_idx}.npy')
    d2U = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d2/d2_U_r={r_idx}_Q={Q_idx}.npy')
    d3U = np.load(f'/home/velni/phd/w/tfm/py/deriv/data/{mod}/d3/d3_U_r={r_idx}_Q={Q_idx}.npy')

    # compute density
    dens = np.zeros((len(y1),len(y2),len(y3[0])))
    for idx in idx_list:
        dens[*idx] = bf.B_expanded(idx,U,d1U,d2U,d3U)
    
    # output
    return dens

def Bnum(dens,r_idx,Q_idx):

    # integrate
    integ = 0
    for idx in idx_list:
        arg = 0
        arg = dens[0][Q_idx][*idx]*dy1[idx[0]]*dy2[idx[1]]*dy3[r_idx][idx[2]]
        integ = integ + arg

    B = (-1/(24*np.pi**2))*integ

    return B

### B numbers (B=2)
"""
import time

Bnums = []
for r in range(0,61):
    t0 = time.time()

    bnum_r = [B(r,Q,0)[0] for Q in range(0,32)]
    Bnums.append(bnum_r)

    stept = time.time() - t0
    print(f'done: {r+1}/61 ----- eta: {stept*(61-r-1)/60.} min')

np.save('/home/velni/phd/w/tfm/py/bnum/bnums', Bnums)

#Bnums = [B(r_idx,15,0)[0] for r_idx in range(0,20)]
"""
# plot
"""
Bs = np.load('/home/velni/phd/w/tfm/py/bnum/bnums.npy')
X = np.arange(0,61,1)

f = plt.figure()
sf = f.add_subplot(1,1,1)

for Q_idx in range(0,61):
        sf.clear()
        sf.set_xlim([0,60])
        sf.set_ylim([1.925,2.05])
        sf.plot(X,Bs[:,Q_idx],'r-')

        plt.xlabel(r'$r$')
        plt.ylabel(r'$B(r)$')
        #plt.title(f'$r={r_vals[r_idx]}$')
        
        #plt.savefig(f'/home/velni/phd/w/tfm/py/misc/bnums/B2_r={r_idx}.png',dpi='figure')

        plt.pause(0.01)

plt.show()
"""
### B densities (B=2)
"""
import time

for r in range(4,61):
    t0 = time.time()
    bdens = []

    bden_r = [Bdens(r,Q,0) for Q in range(0,32)]
    bdens.append(bden_r)

    stept = time.time() - t0
    print(f'done: {r+1}/61 ----- eta: {stept*(61-r-1)/60.} min')
    np.save(f'/home/velni/phd/w/tfm/py/bnum/bdens/B2/B2_dens_r={r}', bdens)
"""

f = plt.figure()
sf = f.add_subplot(1,1,1)

for r in range(0,61):
        X = np.load('/home/velni/phd/w/tfm/py/sample/y3.npy')[r]
        Bs = np.load(f'/home/velni/phd/w/tfm/py/bnum/bdens/B2/B2_dens_r={r}.npy')
        
        sf.clear()
        sf.set_xlim([X[0],X[-1]])
        sf.set_ylim([-100,1])
        sf.plot(X,Bs[0][0][20,20,:],'r-')
        #sf.imshow(Bs[r,0,:,25,:],cmap='gray')    

        plt.xlabel(r'$y_3$')
        plt.ylabel(r'$\mathcal{B}(y_3)$')
        plt.title(f'$r={round(r_vals[r],3)}$')

        print(Bnum(Bs,r,0))

        #plt.savefig(f'/home/velni/phd/w/tfm/py/misc/bnums/B2dens_r={r}.png',dpi='figure')

        plt.pause(0.01)

plt.show()
