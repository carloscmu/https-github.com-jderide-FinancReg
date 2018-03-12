import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def expectation(x,n):
    if x>=0.0 and x<= lopt[n]:
        return a0-a1*lopt[n]*(1.0-exe)
    else:
        return a0-a1*x*(1.0-exe)

def variances(x,n):
    if x>=0.0 and x<= lopt[n]:
        return ((a1*lopt[n]*S[n]*UB)**2.0)/(12.0)
    else:
        return ((a1*x*S[n]*UB)**2.0)/(12.0)

def cov(x,y,n,m):
    if x>=0.0 and x<=lopt[n]:
        if y>=0.0 and y<= lopt[m]:
            return (((a1*UB)**2.0)*lopt[n]*S[n]*lopt[m]*S[m])/(12.0)
        else:
            return (((a1*UB)**2.0)*lopt[n]*S[n]*y*S[m])/(12.0)
    else:
        if y>=0.0 and y<= lopt[m]:
            return (((a1*UB)**2.0)*x*S[n]*lopt[m]*S[m])/(12.0)
        else:
            return (((a1*UB)**2.0)*x*S[n]*y*S[m])/(12.0)

def ExCPRA(x):
    NN = len(x)
    exps = 0.0
    for n in range(0,NN):
        exps = exps + expectation(x[n],n)
    return exps

def varCPRA(x):
    NN = len(x)
    vari = 0.0
    covs = 0.0
    for nn in range(0,NN):
        vari = vari + variances(x[nn],nn)
        for mm in range(0,nn):
            covs = covs + 2.0*cov(x[nn],x[mm],nn,mm)
    return (vari+covs)

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

figext = 'eps'
bdlo = 0.05
bdup = 0.15

Nodes = set([0,1,2,3,4])
Edges = set([(0,1),(1,2),(1,3),(1,4)])
Arcs = set([(0,1),(1,2),(1,3),(1,4),(1,0),(2,1),(3,1),(4,1)])
npts = len(Nodes)

PositionsN = {}
PositionsN[0] = [-2.0,0.0]
PositionsN[1] = [0.0,0.0]
PositionsN[2] = [2.0,2.0]
PositionsN[3] = [2.0,0.0]
PositionsN[4] = [2.0,-2.0]

G = nx.Graph()
for n in Nodes:
    G.add_node(n,pos=PositionsN[n])
for i,j in Edges:
    G.add_edge(i,j)
pos = nx.get_node_attributes(G,'pos')
nx.draw(G,pos,node_size=850,with_labels=True, node_color='lightgray',font_size=16)
plt.savefig('MNt.'+figext)
plt.close()


lbda = 0.05
a0 = 1.0
a1 = 1.0
UB = 0.25
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
#    print(Saux)
S = (Saux+np.eye(npts)).dot(q)
lopt = lbda/(1.0-S*UB)
ru = a0/(a1*(1.0-exe))
print(S)
print(lopt)
print(ru)

NN = 1000
xx = np.linspace(bdlo,bdup,NN)
#X, Y = np.meshgrid(xx,xx)
#ExpCPRA = np.zeros((NN,NN))
#VarsCPRA = np.zeros((NN,NN))
ExpCPRA = 0.0*xx
VarsCPRA = 0.0*xx
for k in range(0,NN):
    ExpCPRA[k] = ExCPRA(xx[k]*np.ones(npts))
    VarsCPRA[k] = varCPRA(xx[k]*np.ones(npts))
plt.figure()
plt.plot(xx,ExpCPRA)
plt.title(r'$\mathbf{E}\{\sum_i\pi_i^*\}$')
plt.savefig('MNt-ExpCPRA.'+figext)
plt.close()
plt.figure()
plt.plot(xx,VarsCPRA)
plt.title(r'${\rm var}\{\sum_i\pi_i^*\}$')
plt.savefig('MNt-VarsCPRA.'+figext)
plt.close()

NTT = 5
theta = np.linspace(0.0,1e3*(NTT-1),NTT)
FCPRA = {}
for nu in range(0,NTT):
    FCPRA[nu] = ExpCPRA - (theta[nu]/2.0)*VarsCPRA
    pname='MNt-FCPRA_%1.2f.'%theta[nu]
    plt.figure()
    plt.plot(xx,FCPRA[nu])
    plt.title(r'$\mathbf{E}\{\sum_i\pi_i^*\}-\frac{\theta}{2}{\rm var}(\sum_i\pi^*_i), \theta=%1.3f$'%theta[nu])
    plt.savefig(pname+figext)
    plt.close()

NAUX = 100
tt = np.linspace(bdlo,bdup,NAUX)
Vars = {}
CovS = {}
for ii in range(0,npts):
    Vars[ii] = 0.0*tt
    for kk in range(0,NAUX):
        Vars[ii][kk] = variances(tt[kk],ii)
        for jj in range(0,ii):
            CovS[ii,jj] = np.zeros((NAUX,NAUX))
            for ll in range(0,NAUX):
                CovS[ii,jj][kk,ll] = cov(tt[kk],tt[ll],ii,jj)
[x0,x1] = np.meshgrid(tt,tt)
for (i,j) in CovS.keys():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x0, x1, CovS[i,j])
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(r'${\rm cov}\{\pi_%i^*,\,\pi^*_%i\}$'%(i,j))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.savefig(('MNt-Cov%i%i.'%(i,j))+figext)
    plt.close()
for i in Nodes:
    plt.figure()
    plt.plot(tt,Vars[i])
    plt.savefig(('MNt-Var_%i.'%i)+figext)
    plt.close()
