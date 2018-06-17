# Kernel densities 
from pomegranate import *
import matplotlib.pyplot as plt
import timeit
import numpy as np

data = np.concatenate( (np.random.randn(12), [2, 5, 9] ))

plt.figure( figsize=(10, 6))

d1 = NormalDistribution(0, 1)
d1.fit( data )
d1.plot( n=25000, edgecolor='c', facecolor='c', bins=50, alpha=0.3, label="Normal" )

d2 = GaussianKernelDensity(data)
d2.plot( n=25000, edgecolor='r', facecolor='r', bins=50, alpha=0.3, label="Gaussian Kernel Density" )
plt.legend()
plt.show()