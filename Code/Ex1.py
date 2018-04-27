import numpy as np
from numpy import linalg as LA

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

A = np.zeros((4,4))

p = 0.05

A[0,1] = p+2*p**2
A[0,2] = p+p**2+p**3
A[0,3] = p+p**2+p**3
A[1,2] = p+p**2+p**3
A[1,3] = p+p**2+p**3
A[2,3] = 2*p**2+2*p**3

A = A.transpose()+A


eps, leps  = LA.eig(A)

#print A
#print eps
#print leps
#print leps[:,0]

B = np.zeros((5,5))
B[0,1] = p
B[0,2] = p**2
B[0,3] = p**2
B[1,2] = p
B[1,3] = p
B[2,3] = p**2

BB = B.transpose()+B
BB[0,4] = 1.0-sum(BB[0,0:4])
#BB[4,0] = BB[0,4]
BB[1,4] = 1.0-sum(BB[1,0:4])
#BB[4,1] = BB[1,4]
BB[2,4] = 1.0-sum(BB[2,0:4])
#BB[4,2] = BB[2,4]
BB[3,4] = 1.0-sum(BB[3,0:4])
#BB[4,3] = BB[3,4]
BB[4,4] = 1.0
print BB

epsB, lepsB  = LA.eig(BB)
print epsB
print lepsB
print is_pos_def(BB)
print lepsB[:,4]
print epsB[4]

lbda = 0.05
for l in xrange(0,4):
	print 'Minimum ratio x_%i %2.2f'%(l,100*lbda/(1.0-lepsB[l,4]))

