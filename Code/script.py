import numpy as np
#from Libraries import *

def pro(lbda,r,p,UB):
	aux = (1.0/p)*(1.0-lbda/r)
	if aux < 0.0:
		return 1.0
	elif aux > UB:
		return 0.0
	else:
		return 1.0-aux/UB

def pro0(lbda,r,UB):
	aux = (1.0-lbda/r)
	if aux < 0.0:
		return 1.0
	elif aux > UB:
		return 0.0
	else:
		return 1.0-aux/UB

lbda = 0.05

a0 = 1.0
a1 = a0
UB = 2.0*lbda
print 'Assume the distribution of epsilon is U[0,UB]'
p = 0.2

Nodes = set([0,1,2])
Edges = set([(0,1),(1,2)])
Arcs = set([(0,1),(1,0),(1,2),(2,1)])

A = np.zeros((3,3))


q = {}
p0 = {}
r = {}
ptilde = np.zeros((3,1))
x = np.zeros((3,1))
for n in Nodes:
	r[n] = lbda*(1.0+0.02+0.0*np.random.rand())
	q[n] = 1.0/3.0
	p0[n] = pro0(lbda,r[n],UB)
	ptilde[n] = q[n]*p0[n]
	x[n] = (r[n]+lbda)/2.0

#data = FinNet(a0,a1,lbda,Nodes,Edges,q,p0)

for i,j in Arcs:
	A[i,j] = pro(lbda,r[i],p,UB)


print A
print r
print lbda
print p0
print q
#print ptilde

P = np.linalg.solve(np.eye(3)-A, ptilde)

print P
#print np.linalg.inv(np.eye(3)-A)

def condpi(x,a0,a1,UB,lbda):
	ubint = min(UB,1.0-lbda/x)
	lbint = max(0.0,1.0-a0/(a1*x))
	print lbint,ubint
	if lbint >= ubint:
		return 0.0
	else:
		int1 = (a0/UB)*(ubint-lbint)
		int2 = (a1*x/(2.0*UB))*((1.0-ubint)**2.0-(1.0-lbint)**2.0)
		return int1+int2
print x
print a0,a1
print UB
eopis = {}
ExUCP = 0.0
for n in Nodes:
	eopis[n] = (1.0-P[n])*condpi(x[n],a0,a1,UB,lbda)
	ExUCP = ExUCP + eopis[n]

print eopis
print ExUCP

xx = np.linspace(lbda,2.0*lbda,10)
ExpUt = 0.0*xx

eopis = {}
ExUCP = 0.0
for n in Nodes:
	eopis[n] = (1.0-P[n])*condpi(x[n],a0,a1,UB,lbda)
	ExUCP = ExUCP + eopis[n]

