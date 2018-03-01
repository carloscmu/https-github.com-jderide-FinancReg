import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

lba = 0.05
UB = 0.3
a0 = 1.0
a1= 1.0
S = 0.5*UB+1.00
exe = (S*UB)/2.0
b = S*UB
print(exe)
print(b)
#print lba/(1.0-S*UB)

def epi(r):
	aux = 1.0-lba/r
	if aux <=0:
		return 0.0
	elif aux>=(S*UB):
#		print 'eval lin'
		return a0-a1*r*(1.0-exe)
	else:
		return (1.0/((S*UB)**2))*(aux**2.0)*(a0-a1*r*(1.0-aux/2.0))

npts = 100
r = np.linspace(0.9*lba,0.150,npts)
f = r*0.0

for k in range(0,npts):
	f[k] = epi(r[k])
#	print r[k],f[k]

plt.plot(r,f,linewidth=3.0)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.title(r'\rm Expected Utility $\mathbf{E}\{\pi(r(1-\epsilon))\}$')
plt.xlabel(r'$r$')
plt.ylabel(r'$\mathbf{E}\{\pi(r(1-\epsilon))\}$')
xc = lba/(1.0-S*UB)
plt.axvline(x=xc, color='k', linestyle='--',linewidth=0.5)
plt.axvline(x=lba, color='k', linestyle='--',linewidth=0.5)
plt.annotate(r'$\frac{\lambda}{1-S\cdot UB}$', xy=(xc, 0.0), xytext=(xc, 0.0))
plt.annotate(r'$\lambda$', xy=(lba, 0.0), xytext=(lba, 0.0))
plt.xticks([])
plt.yticks([])
plt.savefig('ExpectedUt.eps')
