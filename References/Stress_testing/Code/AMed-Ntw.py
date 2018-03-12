import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def expectation(x,n):
    return a0-a1*max(x,lopt[0][n])*(1.0-S[0][n]*UB)
    # Check the term S[o][n]*Ub or *exe

def variances(x,n):
    return ((a1*max(x,lopt[0][n])*S[0][n]*UB)**2.0)/(12.0)

def cov(x,y,n,m):
    return (((a1*UB)**2.0)*max(x,lopt[0][n])*S[0][n]*max(y,lopt[0][m])*S[0][m])/(12.0)

def covAmb(x,y,n,m):
    ret = 0.0
    for k in range(1,KAmb):
        ret0 = max(x,lopt[k][n])*(1.0-S[k][n]*exe)*(max(x,lopt[0][n])*(1.0-S[0][n]*exe))
        ret1 = max(y,lopt[k][m])*(1.0-S[k][m]*exe)*(max(y,lopt[0][m])*(1.0-S[0][m]*exe))
        ret = ret+alpha[k]*ret0+ret1
    return ret

def ExCP(x):
    NN = len(x)
    exps = 0.0
    for n in range(0,NN):
        exps = exps + expectation(x[n],n)
    return exps

def varCP(x):
    NN = len(x)
    vari = 0.0
    covs = 0.0
    for nn in range(0,NN):
        vari = vari + variances(x[nn],nn)
        for mm in range(0,nn):
            covs = covs + 2.0*cov(x[nn],x[mm],nn,mm)
    return (vari+covs)

def varACP(x):
    NN = len(x)
    vari = 0.0
    covs = 0.0
    for nn in range(0,NN):
        vari = vari + covAmb(x[nn],x[nn],nn,nn)
        for mm in range(0,nn):
            covs = covs + 2.0*covAmb(x[nn],x[mm],nn,mm)
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
p = np.array([0.5,0.25,0.75])
alpha = np.array([0.5,0.25,0.25])
exe = UB/2.0
A = np.zeros((npts,npts))
q = np.ones(npts)*1.0/npts
for i,j in Arcs:
	A[i,j] = 1.0
lopt = {}
S = {}
KAmb = len(p)
for k in range(0,KAmb):
    Saux = np.zeros((npts,npts))
    for n in range(1,npts):
        Saux = Saux + (p[k]**n)*LA.matrix_power(A, n)
        for i in range(0,npts):
            Saux[i,i] = 0.0
    #    print(Saux)
    S[k] = (Saux+np.eye(npts)).dot(q)
    lopt[k] = lbda/(1.0-S[k]*UB)
ru = a0/(a1*(1.0-exe))
print(S)
print(lopt)
print(ru)

NN = 1000
xx = np.linspace(bdlo,bdup,NN)
#X, Y = np.meshgrid(xx,xx)
#ExpCPRA = np.zeros((NN,NN))
#VarsCPRA = np.zeros((NN,NN))
ExpCP = 0.0*xx
VarsCP = 0.0*xx
VarsACP = 0.0*xx
for k in range(0,NN):
    ExpCP[k] = ExCP(xx[k]*np.ones(npts))
    VarsCP[k] = varCP(xx[k]*np.ones(npts))
    VarsACP[k] = varACP(xx[k]*np.ones(npts))
plt.figure()
plt.plot(xx,ExpCP)
plt.title(r'$\mathbf{E}\{\sum_i\pi_i^*\}$')
plt.savefig('MNt-ExpCP.'+figext)
plt.close()
plt.figure()
plt.plot(xx,VarsCP)
plt.title(r'${\rm var}\{\sum_i\pi_i^*\}$')
plt.savefig('MNt-VarsCP.'+figext)
plt.close()

NTT = 3
theta = np.linspace(0.0,1e3*(NTT-1),NTT)
NGG = 3
gamma = np.linspace(0.0,1e3*(NGG-1),NGG)
FCP = {}
for nu in range(0,NTT):
    for kappa in range(0,NGG):
        FCP[nu,kappa] = ExpCP - (theta[nu]/2.0)*VarsCP - (gamma[kappa]/2.0)*VarsACP
        pname='MNt-FCP_%1.2f_%1.2f.'%(theta[nu],gamma[kappa])
        plt.figure()
        plt.plot(xx,FCP[nu,kappa])
        plt.title(r'$\mathbf{E}\{\sum_i\pi_i^*\}-\frac{\theta}{2}{\rm var}(\sum_i\pi^*_i)-\frac{\gamma}{2}{\rm var}_{\alpha}(E\sum_i\pi^*_i), \theta=%1.3f,\gamma=%1.3f$'%(theta[nu],gamma[kappa]))
        plt.savefig(pname+figext)
        plt.close()

NAUX = 100
tt = np.linspace(bdlo,bdup,NAUX)
Vars = {}
CovS = {}
VarsA = {}
CovSA = {}
for ii in range(0,npts):
    Vars[ii] = 0.0*tt
    VarsA[ii] = 0.0*tt
    for kk in range(0,NAUX):
        Vars[ii][kk] = variances(tt[kk],ii)
        VarsA[ii][kk] = covAmb(tt[kk],tt[kk],ii,ii)
        for jj in range(0,ii):
            CovS[ii,jj] = np.zeros((NAUX,NAUX))
            CovSA[ii,jj] = np.zeros((NAUX,NAUX))
            for ll in range(0,NAUX):
                CovSA[ii,jj][kk,ll] = covAmb(tt[kk],tt[ll],ii,jj)
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x0, x1, CovSA[i,j])
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(r'${\rm cov}_\alpha\{E\pi_%i^*,\,\pi^*_%i\}$'%(i,j))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.savefig(('MNt-CovA%i%i.'%(i,j))+figext)
    plt.close()
for i in Nodes:
    plt.figure()
    plt.plot(tt,VarsA[i])
    plt.savefig(('MNt-VarA_%i.'%i)+figext)
    plt.close()
    plt.figure()
    plt.plot(tt,Vars[i])
    plt.savefig(('MNt-VarA_%i.'%i)+figext)
    plt.close()

plt.figure()
plt.plot(xx,VarsACP)
plt.title(r'${\rm var}_\alpha\{E\sum_i\pi_i^*\}$')
plt.savefig('MNt-VarsACP.'+figext)
plt.close()
