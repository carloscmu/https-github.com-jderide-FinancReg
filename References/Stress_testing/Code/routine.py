import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import linalg as LA
import numpy as np
import nlopt

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def epiopt(x,l):
	if x <=0 or x>=ru:
		return 0.0
	elif x>= l and x<=ru:
#		print 'eval lin'
		return a0-a1*x*(1.0-exe)
	else:
		return a0-a1*l*(1.0-exe)

def cpopt(x,grad):
	s = 0.0
	N = np.size(x)
	for n in range(0,N):
		s = s + epiopt(x,lopt[n])
	return s


Nodes = set([0,1,2])
Edges = set([(0,1),(1,2)])
Arcs = set([(0,1),(1,0),(1,2),(2,1)])

npts = len(Nodes)

lbda = 0.05
a0 = 1.0
a1 = a0
UB = 0.25
#p = 1.0/npts
p = 0.5
exe = UB/2.0

A = np.zeros((npts,npts))
q = np.ones(npts)*1.0/npts

for i,j in Arcs:
	A[i,j] = 1.0

Saux = np.zeros((npts,npts))
for n in range(1,npts):
	Saux = Saux + (p**n)*LA.matrix_power(A, n)
	for i in range(0,npts):
		Saux[i,i] = 0.0
S = (Saux+np.eye(npts)).dot(q)
lopt = lbda/(1.0-S*UB)
ru = a0/(a1*(1.0-exe))

NGrid = 1000
xx = np.linspace(0.01,0.50,NGrid)
ff = xx*0.0
for k in range(0,NGrid):
	ff[k] = cpopt(xx[k],0)
	print('x=%1.2f f(x)=%1.2f'%(xx[k],ff[k]))

plt.plot(xx,ff,linewidth=3.0)
plt.title(r'{\rm CP - Optimal expected Utility }$\mathbf{E}\{\sum_i \pi_i(r_i^*(x))\}$')
plt.xlabel(r'$x$')
plt.ylabel(r'$\mathbf{E}\{\sum_{i} \pi_i(r_i^*(x))\}$')
plt.savefig('CPExpUt.pdf')


##########################################################################################3

##########################################################################################3
