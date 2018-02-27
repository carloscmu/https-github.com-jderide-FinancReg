import numpy as np
import matplotlib.pyplot as plt


lba = 0.05
UB = 2.0*lba
a0 = 1.0
a1= 1.0
S = 0.5*UB+1.00
exe = (S*UB)/2.0
print exe
print lba/(1.0-S*UB)

def epi(r):
	aux = 1.0-lba/r
	if aux <=0:
		return 0.0
	elif aux>=(S*UB):
		print 'eval lin'
		return a0-a1*r*(1.0-exe)
	else:
		return ((1.0/S*UB)**2)*(aux**2.0)*(a0-a1*(r+lba)/2.0)

npts = 10
r = np.linspace(0.9*lba,0.060,npts)
f = r*0.0

for k in xrange(0,npts):
	f[k] = epi(r[k])
	print r[k],f[k]

plt.plot(r,f)
plt.show()
