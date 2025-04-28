import numpy as np

l1 = 40
l2 = 40
l3 = 80

yt1 = np.linspace(-0.9,0.9,l1)
yt2 = np.linspace(-0.9,0.9,l2)
yt3 = np.linspace(-0.9,0.9,l3)

y1 = [yt1[i]/(1-yt1[i]**2) for i in range(l1)]
y2 = [yt2[i]/(1-yt2[i]**2) for i in range(l2)]

def y3_gen(yt3,r):
    y3_pos = []
    for i in range(len(yt3)):
        if yt3[i] >= 0.5:
            ytemp = r/2. + (2*yt3[i]-1)/(8*(yt3[i]-1)**2)
            y3_pos.append(ytemp)
        elif (yt3[i] < 0.5) and (yt3[i] >= 0):
            ytemp = yt3[i]*(2*r*(1-yt3[i])-(1-2*yt3[i]))
            y3_pos.append(ytemp)
        else:
            pass

    y3_neg = [-y for y in y3_pos]

    y3 = list(reversed(y3_neg))+y3_pos

    return y3

### output

r_vals = np.load('/home/velni/phd/w/tfm/py/sample/r.npy')

np.save('/home/velni/phd/w/tfm/py/sample/y1',y1)
np.save('/home/velni/phd/w/tfm/py/sample/y2',y2)

y3 = np.zeros((len(r_vals),l3))
for r_idx,r in enumerate(r_vals):
    y3_temp = y3_gen(yt3,r)
    y3[r_idx] = y3_temp

np.save('/home/velni/phd/w/tfm/py/sample/y3',y3)

### plot
"""
import matplotlib.pyplot as plt

plt.plot(yt3,y3_gen(yt3,1.131),'r-')
plt.show()
"""
