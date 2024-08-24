import numpy as np
import random as rd

### r sampling

r_vals = np.arange(1.731, 7.731 + 0.1, 0.1)


### Q sampling

int_vals = np.arange(-1, 1, 0.01)

u_vals = np.array([rd.choice(int_vals) for sample in range(32)])
v_vals = np.array([rd.choice(int_vals) for sample in range(32)])
w_vals = np.array([rd.choice(int_vals) for sample in range(32)])

hyp_vals = [(1./np.sqrt(1+ u**2 + v**2 + w**2))*np.array([1,u,v,w]) for u,v,w in zip(u_vals,v_vals,w_vals)]

Q_vals = [ np.array([[q[0]+q[1]*1j, q[2]+q[3]*1j],[-q[2]+q[3]*1j, q[0]-q[1]*1j]]) for q in hyp_vals]

### save data

np.save('/home/velni/Escritorio/TFM/py/sample/r.dat', r_vals)
np.save('/home/velni/Escritorio/TFM/py/sample/Q.dat', Q_vals)