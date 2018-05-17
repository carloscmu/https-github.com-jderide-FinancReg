import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)
ext = 'pdf'

bdlo = 0.005
bdup = 1.0
lbda = 0.05


Nodes = set([0,1,2])
Arcs = set([(1,0),(1,2),(0,2)])
Scn = set([0])
alpha = {}
for k in Scn:
    alpha[k] = 1.0/len(Arcs)
npts = len(Nodes)
q = {}
for n in Nodes:
    q[n] = 1.0/npts
E = {}
for k in Scn:
    E[k] = {}
    for i,j in Arcs:
        E[k][i,j] = 0.5

a0 = 1.0
a1 = 1.0
b = 0.25
exe = b/2.0

Saux = {}
Eaux = {}
for k in Scn:
    AA = np.eye(npts)
    BB = np.zeros((npts,npts))
    Saux[k] = AA
    Eaux[k] = BB
    for (n,m) in E[k]:
        Eaux[k][n,m] = E[k][n,m]
    for n in range(1,npts):
        Saux[k] += np.matmul(Eaux[k],Saux[k])
        for i in range(0,npts):
            Saux[k][i,i] = 0.0
print(Saux)
lopt = {}
lopt_cp = {}
S = {}
S_cp = {}
for k in Scn:
    lopt[k] = {}
    lopt_cp[k] = {}
    S[k] = {}
    S_cp[k] = {}
    for n in Nodes:
        S[k][n] = q[n]
        S_cp[k][n] = q[n]
        for m in Nodes:
            if (m,n) in E[k]:
                S[k][n] += E[k][m,n]*q[m]
            if Saux[k][m,n] != 0.0:
                S_cp[k][n] += Saux[k][m,n]*q[m]
        lopt[k][n] = lbda/(1.0-S[k][n]*b)
        lopt_cp[k][n] = lbda/(1.0-S_cp[k][n]*b)
ru = a0/(a1*(1.0-exe))

def epi_cp(x,S,li):
    ep0 = (1.0/b*S)*(1.0-lbda/(max(x,li)))
    ep1 = a0-a1*((max(x,li)+lbda)/2.0)
    return ep1*(ep0**2)

NN = 1000
for k in Scn:
    tt = np.linspace(bdlo,bdup,NN)
    xx = {}
    sumep = tt*0.0
    for i in Nodes:
        xx[i] = tt*0.0
        for l in range(NN):
            xx[i][l] = epi_cp(tt[l],S_cp[k][i],lopt[k][i])
            sumep[l] += xx[i][l]
        plt.plot(tt, xx[i])
#plt.plot(tt, sumep)
plt.xlabel(r'Policy level')
plt.ylabel(r'$\mathbf{E}_{CP}$')
plt.title(r'Expected utility function for each agent')
plt.grid(True)
plt.savefig('ExpUtCP.'+ext)
plt.show()
