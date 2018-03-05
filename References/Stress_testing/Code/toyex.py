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

def ExCPRA(x,y):
    exps = 2.0*expectation(x,0)+expectation(y,1)
    return exps

def varCPRA(x,y):
    var =  2.0*variances(x,0)+variances(y,1)
    covs = 2.0*(cov(x,y,0,1)+cov(x,x,0,2)+cov(x,y,1,2))
    return (var+covs)

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

figext = 'eps'
bdlo = 0.05
bdup = 0.07

Nodes = set([0,1,2])
Edges = set([(0,1),(1,2)])
Arcs = set([(0,1),(1,0),(1,2),(2,1)])
npts = len(Nodes)

PositionsN = {}
PositionsN[0] = [-2.0,0.0]
PositionsN[1] = [0.0,0.0]
PositionsN[2] = [2.0,0.0]

G = nx.Graph()
for n in Nodes:
    G.add_node(n,pos=PositionsN[n])
for i,j in Edges:
    G.add_edge(i,j)
pos = nx.get_node_attributes(G,'pos')
nx.draw(G,pos,node_size=850,with_labels=True, node_color='lightgray',font_size=16)
plt.savefig('ToyNtw.'+figext)
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
    print(Saux)
S = (Saux+np.eye(npts)).dot(q)
lopt = lbda/(1.0-S*UB)
ru = a0/(a1*(1.0-exe))

NN = 1000
xx0 = np.linspace(bdlo,bdup,NN)
xx1 = np.linspace(bdlo,bdup,NN)

X, Y = np.meshgrid(xx0,xx1)
ExpCPRA = np.zeros((NN,NN))
VarsCPRA = np.zeros((NN,NN))
for k in range(0,NN):
    for l in range(0,NN):
        ExpCPRA[k,l] = ExCPRA(xx0[k],xx1[l])
        VarsCPRA[k,l] = varCPRA(xx0[k],xx1[l])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, ExpCPRA)
ax.set_xlabel(r'$x_0=x_2$')
ax.set_ylabel(r'$x_1$')
ax.set_title(r'$\mathbf{E}\{\sum_i\pi_i^*\}$')
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
plt.savefig('ExpCPRA.'+figext)
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, VarsCPRA)
ax.set_xlabel(r'$x_0=x_2$')
ax.set_ylabel(r'$x_1$')
ax.set_title(r'${\rm var}\{\sum_i\pi_i^*\}$')
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
plt.savefig('VarsCPRA.'+figext)
plt.close()

NTT = 2
theta = np.linspace(0.0,1e4*(NTT-1),NTT)
FCPRA = {}
for nu in range(0,NTT):
#    FCPRA[nu] = np.zeros((NN,NN))
    FCPRA[nu] = ExpCPRA - (theta[nu]/2.0)*VarsCPRA
    pname='FCPRA_%1.2f.'%theta[nu]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, FCPRA[nu])
    ax.set_xlabel(r'$x_0=x_2$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(r'$\mathbf{E}\{\sum_i\pi_i^*\}-\frac{\theta}{2}{\rm var}(\sum_i\pi^*_i)$')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.savefig(pname+figext)
    plt.close()

NAUX = 100
tt = np.linspace(bdlo,bdup,NAUX)
Vars = {}
Vars[0] = tt*0.0
Vars[1] = tt*0.0
CovS = {}
CovS[0,1] = np.zeros((NAUX,NAUX))
CovS[0,2] = np.zeros((NAUX,NAUX))
CovS[1,2] = np.zeros((NAUX,NAUX))
for k in range(0,NAUX):
    Vars[0][k] = variances(tt[k],0)
    Vars[1][k] = variances(tt[k],1)
    for l in range(0,NAUX):
        CovS[0,1][k,l] = cov(tt[k],tt[l],0,1)
        CovS[0,2][k,l] = cov(tt[k],tt[l],0,2)
        CovS[1,2][k,l] = cov(tt[k],tt[l],1,2)
[x0,x1] = np.meshgrid(tt,tt)
for (i,j) in CovS.keys():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x0, x1, CovS[i,j])
    ax.set_xlabel(r'$x_0=x_2$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(r'${\rm cov}\{\pi_%i^*,\,\pi^*_%i\}$'%(i,j))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.savefig(('Cov%i%i.'%(i,j))+figext)
    plt.close()

plt.figure()
plt.plot(tt,Vars[0])
plt.savefig('Var_0.'+figext)
plt.close()
plt.figure()
plt.plot(tt,Vars[1])
plt.savefig('Var_1.'+figext)
plt.close()
