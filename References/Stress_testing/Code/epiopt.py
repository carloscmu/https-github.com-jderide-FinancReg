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
lopt = lba/(1.0-S*UB)
ru = a0/(a1*(1.0-exe))
print('ru %f'%ru)
def epiopt(x):
	if x <=0 or x>=ru:
		return 0.0
	elif x>= lopt and x<=ru:
#		print 'eval lin'
		return a0-a1*x*(1.0-exe)
	else:
		return a0-a1*lopt*(1.0-exe)

npts = 100
x = np.linspace(0.01,1.1*ru,npts)
piopt = x*0.0

for k in range(0,npts):
	piopt[k] = epiopt(x[k])
	print(x[k],piopt[k])

plt.plot(x,piopt,linewidth=3.0)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.title(r'{\rm Optimal expected Utility }$\mathbf{E}\{\pi(r^*(x))\}$')
plt.xlabel(r'$x$')
plt.ylabel(r'$\mathbf{E}\{\pi(r^*(x))\}$')
plt.axvline(x=lopt, color='k', linestyle='--',linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--',linewidth=0.5)
plt.axvline(x=ru, color='k', linestyle='--',linewidth=0.5)
plt.annotate(r'$\frac{\lambda}{1-S\cdot UB}$', xy=(lopt, 0.0), xytext=(lopt, 0.0))
#plt.annotate(r'$\lambda$', xy=(lba, 0.0), xytext=(lba, 0.0))
plt.annotate(r'$\frac{a^0}{a^1}\frac{1}{1-E\{\epsilon\}}$', xy=(ru, 0.1), xytext=(ru, 0.1))
plt.xticks([])
plt.yticks([])
plt.savefig('OptExpUt.eps')
