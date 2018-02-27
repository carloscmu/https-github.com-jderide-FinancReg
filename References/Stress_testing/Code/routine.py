import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import linalg as LA
import numpy as np


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


Nodes = set([0,1,2])
Edges = set([(0,1),(1,2)])
Arcs = set([(0,1),(1,0),(1,2),(2,1)])

npts = len(Nodes)

lbda = 0.05
a0 = 1.0
a1 = a0
UB = 2.0*lbda
p = 1.0/3.0

ptilde = np.zeros((npts,1))
A = np.zeros((npts,npts))
q = np.ones(npts)*1.0/npts

for i,j in Arcs:
	A[i,j] = 1.0

S = np.zeros((npts,npts))
for n in xrange(1,npts):
	print n
	print LA.matrix_power(A,n)
	S = S + (p**n)*LA.matrix_power(A, n)
	for i in xrange(0,npts):
		S[i,i] = 0.0
	print S
S = (S+np.eye(npts)).dot(q)
print S
print UB

#npts = 10
#r = np.linspace(0.9*lba,0.060,npts)
#f = r*0.0

#for k in xrange(0,npts):
#	f[k] = epi(r[k])
#	print r[k],f[k]

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.plot(r,f)
#plt.show()


##########################################################################################3

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

def epi(r):
	aux = 1.0-lba/r
	if aux <=0:
		return 0.0
	elif aux>=(S*UB):
		print 'eval lin'
		return a0-a1*r*(1.0-exe)
	else:
		return ((1.0/S*UB)**2)*(aux**2.0)*(a0-a1*(r+lba)/2.0)

def ExUCP(x):
# Check this code
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

def epi_o(r):
	aux = 1.0-lba/r
	if aux <=0:
		return 0.0
	elif aux>=(S*UB):
		print 'eval lin'
		return a0-a1*r*(1.0-exe)
	else:
		return ((1.0/S*UB)**2)*(aux**2.0)*(a0-a1*(r+lba)/2.0)


#def svector(q,p,
