import numpy as np

### read grid and r

y1 = np.load('/home/velni/phd/w/tfm/py/sample/y1.npy')
y2 = np.load('/home/velni/phd/w/tfm/py/sample/y2.npy')
y3 = np.load('/home/velni/phd/w/tfm/py/sample/y3.npy')

l1 = len(y1)
l2 = len(y2)
l3 = len(y3[0])

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')

### \tilde{y} sample

yt1 = np.linspace(-0.9,0.9,l1)
yt2 = np.linspace(-0.9,0.9,l2)
yt3 = np.linspace(-0.9,0.9,l3)

### differentials

dyt1 = yt1[1]-yt1[0]
dyt2 = yt2[1]-yt2[0]
dyt3 = yt3[1]-yt3[0]

#

dr = r_vals[1]-r_vals[0]

#

def dy1_gen(yt1):
	dy1 = []
	py1 = np.zeros(len(yt1))
	for i in range(len(yt1)):
		arg = (1./(1-yt1[i])**2)
		py1[i] = arg
		dy1.append(py1*dyt1)

	return np.array(dy1)

def dy2_gen(yt2):
	dy2 = []
	py2 = np.zeros(len(yt2))
	for i in range(len(yt2)):
		arg = (1./(1-yt2[i])**2)
		py2[i] = arg
		dy2.append(py2*dyt2)

	return np.array(dy2)

def dy3_gen(yt3,r):
	dy3_pos = []
	for i in range(len(yt3)):
		if yt3[i] >= 0.5:
			dytemp = dr/2. - (yt3[i]*dyt3)/(4*(yt3[i]-1)**3)
			dy3_pos.append(dytemp)
		elif (yt3[i] < 0.5) and (yt3[i] >= 0):
			dytemp = -2*(yt3[i]-1)*dr + (4*yt3[i] + r*(2-4*yt3[i]) - 1)*dyt3
			dy3_pos.append(dytemp)
		else:
			pass

	dy3_neg = [-dy for dy in dy3_pos]

	dy3 = np.array(list(reversed(dy3_neg))+dy3_pos)

	return dy3

### output

dy1 = dy1_gen(yt1)

dy2 = dy2_gen(yt2)

dy3 = np.zeros((len(r_vals),l3))
for r_idx,r in enumerate(r_vals):
	dy3_temp = dy3_gen(yt3,r)
	dy3[r_idx] = dy3_temp

np.save('/home/velni/phd/w/tfm/py/sample/dy1', dy1)
np.save('/home/velni/phd/w/tfm/py/sample/dy2', dy2)
np.save('/home/velni/phd/w/tfm/py/sample/dy3', dy3)
