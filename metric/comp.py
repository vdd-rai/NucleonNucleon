import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

###

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')

# jump 13 -> 14

g_15 = []
for r_index in range(10,17):
        g_15.append( np.load(f'/home/velni/phd/w/tfm/py/metric/data/0/g_15_r={r_index}.npy') )

g_24 = []
for r_index in range(10,40):
        g_24.append( np.load(f'/home/velni/phd/w/tfm/py/metric/data/0/g_24_r={r_index}.npy') )

g_14 = []
for r_index in range(10,40):
        g_14.append( np.load(f'/home/velni/phd/w/tfm/py/metric/data/0/g_14_r={r_index}.npy') )

g_45 = []
for r_index in range(10,30):
        g_45.append( np.load(f'/home/velni/phd/w/tfm/py/metric/data/0/g_45_r={r_index}.npy') )

###
"""
X = np.arange(0,32,1)

for idx in range(10,17):
	plt.clf()

	plt.xlim(0,31)
	plt.ylim(-5,5)

	plt.xlabel(r'$Q$')
	plt.ylabel(r'$g(Q)$')
	plt.title(f'$r={r_vals[idx]}$')

	plt.plot(X,g_15[idx-10],'k-',label=r'$g_{15}$')
	plt.plot(X,g_24[idx-10],'r-',label=r'$g_{24}$')

	plt.legend()
	plt.grid(True)
	plt.savefig(f'/home/velni/phd/w/tfm/py/misc/metr/new_step={idx}.png',dpi='figure')
"""
###

plt.clf()

X = np.arange(0,32,1)

#

plt.xlabel(r'$Q$')
plt.ylabel(r'$g(Q)$')

#plt.plot(X,g_15[14-10],'k-',label=r'$g_{15}$')
#plt.plot(X,g_14[14-10],'r-',label=r'$g_{14}$')
plt.plot(X,g_45[14-10],'g-',label=r'$g_{45}$')

plt.legend()
plt.grid(True)
plt.show()

