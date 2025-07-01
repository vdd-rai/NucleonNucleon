import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0+4,61-4,1)
Y = np.loadtxt('VSSIV.txt')

plt.plot(X,Y,'r-')
plt.show()
