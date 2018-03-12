import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time

def expectation(x,n):
    return a0-a1*max(x,lopt[0][n])*(1.0-S[0][n]*exe)
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
        ret = ret+alpha[k]*(ret0+ret1)
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

def plotNtw(Nodes, Edges, PositionsN,fname):
    fig = plt.figure()
    G = nx.Graph()
    for n in Nodes:
        G.add_node(n,pos=PositionsN[n])
    for i,j in Edges:
        G.add_edge(i,j)
    pos = nx.get_node_attributes(G,'pos')
    nx.draw(G,pos,node_size=850,with_labels=True, node_color='lightgray',font_size=16)
    plt.savefig(fname)
    plt.close(fig)

def sol3d(X,Y,F,fname,ptitle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, F)
    ax.set_xlabel(r'$x_0=x_2$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(ptitle)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.savefig(fname)
    plt.close(fig)

def objfunplt(X,Y,F,fname,ptitle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, F)
    ax.set_xlabel(r'$x_0=x_2$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(ptitle)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.savefig(fname)
    plt.close(fig)

def ofUniPolplt(x,f,fname,ptitle):
    fig = plt.figure()
    plt.plot(x,f)
    plt.xlabel(r'$x$')
    plt.title(ptitle)
    plt.savefig(fname)
    plt.close(fig)

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

start_time = time.time()

figext = 'eps'
bdlo = 0.00
bdup = 0.2

Nodes = set([0,1,2])
Edges = set([(0,1),(1,2)])
Arcs = set([(0,1),(1,0),(1,2),(2,1)])
npts = len(Nodes)

PositionsN = {}
PositionsN[0] = [-2.0,0.0]
PositionsN[1] = [0.0,0.0]
PositionsN[2] = [2.0,0.0]

plotNtw(Nodes, Edges, PositionsN,'AToy.'+figext)

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
print('KAmb+%i\n'%KAmb)
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
ExpCP = np.zeros((NN,NN))
VarsCP = np.zeros((NN,NN))
VarsACP = np.zeros((NN,NN))

X, Y = np.meshgrid(xx,xx)
for k in range(0,NN):
    for l in range(0,NN):
        ExpCP[k,l] = ExCP([xx[k],xx[l],xx[k]])
        VarsCP[k,l] = varCP([xx[k],xx[l],xx[k]])
        VarsACP[k,l] = varACP([xx[k],xx[l],xx[k]])

ptitle1 = r'$\mathbf{E}\{\sum_i\pi_i^*\}$'
ptitle2 = r'${\rm var}\{\sum_i\pi_i^*\}$'
ptitle3 = r'${\rm var}_\alpha\{E\sum_i\pi_i^*\}$'

sol3d(X,Y,ExpCP,'AToy-ExCP.'+figext,ptitle1)
sol3d(X,Y,VarsCP,'AToy-VarsCP.'+figext,ptitle2)
sol3d(X,Y,VarsACP,'AToy-VarsACP.'+figext,ptitle3)

Ex_UniPoli = xx*0.0
Var_UniPoli = xx*0.0
VarA_UniPoli = xx*0.0
for k in range(0,NN):
    Ex_UniPoli[k] = ExCP([xx[k],xx[k],xx[k]])
    Var_UniPoli[k] = varCP([xx[k],xx[k],xx[k]])
    VarA_UniPoli[k] = varACP([xx[k],xx[k],xx[k]])
ofUniPolplt(xx,Ex_UniPoli,'AToy-ExCPUP.'+figext,ptitle1)
ofUniPolplt(xx,Var_UniPoli,'AToy-VarCPUP.'+figext,ptitle2)
ofUniPolplt(xx,VarA_UniPoli,'AToy-VarACPUP.'+figext,ptitle3)

F_UniPoli = {}

NTT = 3
theta = np.linspace(0.0,1e3*(NTT-1),NTT)
NGG = 3
gamma = np.linspace(0.0,1e3*(NGG-1),NGG)
FCP = {}
for nu in range(0,NTT):
    for kappa in range(0,NGG):
        FCP[nu,kappa] = ExpCP - (theta[nu]/2.0)*VarsCP - (gamma[kappa]/2.0)*VarsACP
        F_UniPoli[nu,kappa] = Ex_UniPoli - (theta[nu]/2.0)*Var_UniPoli - (gamma[kappa]/2.0)*VarA_UniPoli
        pname='AToy-FCP_%1.2f_%1.2f.'%(theta[nu],gamma[kappa])
        pnameUP='AToy-FCPUP_%1.2f_%1.2f.'%(theta[nu],gamma[kappa])
        ptitle = r'$\mathbf{E}\{\sum_i\pi_i^*\}-\frac{\theta}{2}{\rm var}(\sum_i\pi^*_i)-\frac{\gamma}{2}{\rm var}_{\alpha}(E\sum_i\pi^*_i), \theta=%1.3f,\gamma=%1.3f$'%(theta[nu],gamma[kappa])
        fname = pname+figext
        objfunplt(X,Y,FCP[nu,kappa],fname,ptitle)
        ofUniPolplt(xx,F_UniPoli[nu,kappa],pnameUP+figext,ptitle)

print("--- %s seconds ---" % (time.time() - start_time))
