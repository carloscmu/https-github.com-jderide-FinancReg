import numpy as np
#from Libraries import *
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def pro(lbda,r,p,UB):
	aux = (1.0/p)*(1.0-lbda/r)
	if aux < 0.0:
		return 1.0
	elif aux > UB:
		return 0.0
	else:
		return 1.0-aux/UB

def condpi(x,a0,a1,UB,lbda):
	ubint = min(UB,1.0-lbda/x)
	lbint = max(0.0,1.0-a0/(a1*x))
	if lbint >= ubint:
		return 0.0
	else:
		int1 = (a0/UB)*(ubint-lbint)
		int2 = (a1*x/(2.0*UB))*((1.0-ubint)**2.0-(1.0-lbint)**2.0)
		return int1+int2

lbda = 0.05
a0 = 1.0
a1 = a0
UB = 2.0*lbda
p = 0.2

Nodes = set([0,1,2])
Edges = set([(0,1),(1,2)])
Arcs = set([(0,1),(1,0),(1,2),(2,1)])

npts = len(Nodes)
#x0 = np.zeros(npts)

#



def ExUCP(x):
	n = np.size(x)
	ptilde = np.zeros((n,1))
	A = np.zeros((n,n))
	q = np.ones(n)*1.0/n
	p0 = np.zeros(n)
	r = x
	for k in xrange(0,n):
		p0[k] = pro(lbda,r[k],1.0,UB)
		ptilde[k] = q[k]*p0[k]

	for i,j in Arcs:
		A[i,j] = pro(lbda,r[i],p,UB)

	P = np.linalg.solve(np.eye(n)-A, ptilde)
	ExUCP = 0.0
	for n in Nodes:
		ExUCP = ExUCP + (1.0-P[n])*condpi(x[n],a0,a1,UB,lbda)
	return ExUCP

NN = 100
xx = np.linspace(lbda,2*lbda,NN)
ff = 0.0*xx
for ii in range(0,NN):
	ff[ii] = ExUCP(xx[ii]*np.ones(npts))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(xx,ff)
plt.ylabel('Expected utility CP')
plt.show()

#for ii in range(0,Npts):
#	print 'ExUt(%1.2f) = %2.4f'%(xx[ii], ExUCP[ii])
