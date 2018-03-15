#==========================================================================
# Code for "Robust Regulation of Financial Systems"
# Author: JULIO DERIDE
# This version: March 14, 2018.
#==========================================================================
# This code implements the analytical solutions for the model proposed in
#
# Deride, J., Ramirez, C., "A Model for Robust Regulation of Financial Networks"
#
#
import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import time
from scipy.optimize import minimize
from aniplts import *

def expectation(x,n):
    return a0-a1*max(x,lopt[0][n])*(1.0-S[0][n]*exe)

def variances(x,n):
    return ((a1*max(x,lopt[0][n])*S[0][n]*UB)**2.0)/(12.0)

def cov(x,y,n,m):
    return (((a1*UB)**2.0)*max(x,lopt[0][n])*S[0][n]*max(y,lopt[0][m])*S[0][m])/(12.0)

def covAmb(x,y,n,m):
    ret = 0.0
    for k in range(1,KAmb):
        ret0 = max(x,lopt[k][n])*(1.0-S[k][n]*exe)-(max(x,lopt[0][n])*(1.0-S[0][n]*exe))
        ret1 = max(y,lopt[k][m])*(1.0-S[k][m]*exe)-(max(y,lopt[0][m])*(1.0-S[0][m]*exe))
        ret = ret+alpha[k]*(ret0*ret1)
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

def sol3d(X,Y,F,fname,ptitle,pltext):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_surface(X, Y, F)
    surf = ax.plot_surface(X, Y, F, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_xlabel(r'$x_0=x_2$')
    ax.set_ylabel(r'$x_1$')
    ax.set_title(ptitle)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.savefig(fname+pltext)
    angles = np.linspace(0,360,51)[:-1] # A list of 20 angles between 0 and 360
    rotanimate(ax, angles,fname+'gif',delay=20)
    plt.close(fig)

def ofUniPolplt(x,f,fname,ptitle):
    fig = plt.figure()
    plt.plot(x,f)
    plt.xlabel(r'$x$')
    for k in range(0,KAmb):
        for i in range(0,npts):
            plt.axvline(x=lopt[k][i], linestyle='dotted', linewidth=0.3, color=snlcol[i])
            plt.annotate(r'$\lambda_%i[p_{%i}]$'%(i,k), xy=(lopt[k][i], 2.75), xytext=(lopt[k][i], 2.75),size=5)
    plt.title(ptitle)
    plt.savefig(fname,bbox_inches="tight")
    plt.close(fig)

def objfunction(x,theta,gamma):
    return -ExCP(x) + (theta/2.0)*varCP(x) + (gamma/2.0)*varACP(x) - 1e-1*sum(abs(x[n]) for n in range(0,npts))

def objfunctionUP(x,theta,gamma):
    return -ExCP(x*np.ones(npts)) + (theta/2.0)*varCP(x*np.ones(npts)) + (gamma/2.0)*varACP(x*np.ones(npts)) - 1e-2*abs(x)

def writsol(fname):
    f = open(fname,'w')
    f.write('Resilient financial regulation\n\n')
    f.write('\n Lambda: %1.3f'%lbda)
    f.write('\n a0: %1.3f'%a0)
    f.write('\n a1: %1.3f'%a1)
    f.write('\n UB: %1.3f'%UB)
    f.write('\n Lower bound for plotting %1.3f\n'%bdlo)
    f.write('\n Upper bound for plotting %1.3f\n'%bdup)
    aux = '\n Nodes:'
    aux1 = '\n Positions:\n'
    for n in Nodes:
        aux = aux+'\t'+('%i'%n)
        aux1 = aux1 + '\t' + ('%i;(%1.3f,%1.3f)'%(n,PositionsN[n][0],PositionsN[n][1])) + '\n'
    f.write(aux)
    f.write(aux1)
    aux = '\n Edges:\n'
    for (i,j) in Edges:
        aux = aux+('(%i,%i)\n' %(i,j))
    f.write(aux+'\n\n')
    f.write('Ambiguity\n k \t alpha \t p_alpha:\n')
    for k in range(0,KAmb):
        f.write('%i\t%1.3f\t%1.3f\n'%(k,alpha[k],p[k]))
    auxS = '\n\n S: k\t n :\n'
    auxL = '\n \lambda: k\t n :\n'
    for k in range(0,KAmb):
        for n in range(0,npts):
            auxS = auxS + ('%i\t%i\t%f\n'%(k,n,S[k][n]))
            auxL = auxL + ('%i\t%i\t%f\n'%(k,n,lopt[k][n]))
    f.write(auxS)
    f.write(auxL)
    f.write('\n r_u: %1.3f\n\n'%ru)
    aux = '\n\n Optimal Sols:\n theta[nu];gamma[kappa];Xopt[nu,kappa][0];Xopt[nu,kappa][1];Xopt[nu,kappa][2];Fopt[nu,kappa]\n'
    aux1 = '\n\n Optimal Sols:\n theta[nu];gamma[kappa];Xopt[nu,kappa];Fopt[nu,kappa]\n'
    for nu in range(0,NTT):
        for kappa in range(0,NGG):
            aux = aux+'%1.3f ; %1.3f ; %1.5f ; %1.5f; %1.5f ; %1.5f \n'%(theta[nu],gamma[kappa],Xopt[nu,kappa][0],Xopt[nu,kappa][1],Xopt[nu,kappa][2],Fopt[nu,kappa])
            aux1 = aux1+'%1.3f ; %1.3f ; %1.5f ; %1.5f \n'%(theta[nu],gamma[kappa],Topt[nu,kappa],Ftopt[nu,kappa])
    f.write(aux)
    f.write('\n\n'+aux1)
    f.close()

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

snlcol = ['g', 'r', 'c']


start_time = time.time()

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
ptitle3 = r'${\rm var}_\alpha\{\mathbf{E}\sum_i\pi_i^*\}$'

sol3d(X,Y,ExpCP,'AToy-ExCP.',ptitle1,figext)
sol3d(X,Y,VarsCP,'AToy-VarsCP.',ptitle2,figext)
sol3d(X,Y,VarsACP,'AToy-VarsACP.',ptitle3,figext)

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

NTT = 1
theta = np.linspace(0.0,1e3*(NTT-1),NTT)
NGG = 1
gamma = np.linspace(0.0,1e3*(NGG-1),NGG)
FCP = {}
for nu in range(0,NTT):
    for kappa in range(0,NGG):
        FCP[nu,kappa] = ExpCP - (theta[nu]/2.0)*VarsCP - (gamma[kappa]/2.0)*VarsACP
        F_UniPoli[nu,kappa] = Ex_UniPoli - (theta[nu]/2.0)*Var_UniPoli - (gamma[kappa]/2.0)*VarA_UniPoli
        fname='AToy-FCP_%1.2f_%1.2f.'%(theta[nu],gamma[kappa])
        fnameUP='AToy-FCPUP_%1.2f_%1.2f.'%(theta[nu],gamma[kappa])
        ptitle = r'$\mathbf{E}\{\sum_i\pi_i^*\}-\frac{\theta}{2}{\rm var}(\sum_i\pi^*_i)-\frac{\gamma}{2}{\rm var}_{\alpha}(\mathbf{E}\sum_i\pi^*_i), \theta=%1.3f,\gamma=%1.3f$'%(theta[nu],gamma[kappa])
        sol3d(X,Y,FCP[nu,kappa],fname,ptitle,figext)
        ofUniPolplt(xx,F_UniPoli[nu,kappa],fnameUP+figext,ptitle)

print("--- %s seconds ---" % (time.time() - start_time))

x0 = np.array([0.1,0.1,0.1])
t0 = 0.075
NTT = 1
theta = np.linspace(0.0,1e3*(NTT-1),NTT)
NGG = 101
gamma = np.linspace(0.0,1e1*(NGG-1),NGG)
Fopt = {}
Ftopt = {}
Xopt = {}
Topt = {}
for nu in range(0,NTT):
    for kappa in range(0,NGG):
        res = minimize(objfunction, x0, args=(theta[nu],gamma[kappa]),method='Nelder-Mead',options={'xtol': 1e-12})#, 'disp': True})
        Xopt[nu,kappa] = res.x
        Fopt[nu,kappa] = -objfunction(res.x,theta[nu],gamma[kappa])
        res = minimize(objfunctionUP, t0, args=(theta[nu],gamma[kappa]),method='Nelder-Mead',options={'xtol': 1e-12})#, 'disp': True})
        Topt[nu,kappa] = res.x
        Ftopt[nu,kappa] = -objfunctionUP(res.x,theta[nu],gamma[kappa])
writsol('Sol.dat')

tt = np.linspace(bdlo,bdup,NN)
ff = np.zeros((NN,NGG))
for kappa in range(0,NGG):
    for i in range(0,NN):
        ff[i,kappa] = -objfunctionUP(tt[i],0.0,gamma[kappa])
#    fig = plt.figure()
#    plt.plot(tt,ff[nu,kappa])
#    plt.axvline(x=Topt[nu,kappa])
#    for k in range(0,KAmb):
#        for i in range(0,npts):
#            plt.axvline(x=lopt[k][i], linestyle='dotted', linewidth=0.3, color=snlcol[i])
#    pname = r'Objective function, universal policy for $\theta=%4.0f$, $\gamma=%4.0f$'%(theta[nu],gamma[kappa])
#    fname = 'AToy-ObFnUP+%i%i.'%(theta[nu],gamma[kappa])
#    plt.title(pname)
#    plt.savefig(fname+figext)
#    plt.close()
pfilean = 'AToy-animated.'
GG,TT = np.meshgrid(gamma,tt)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(GG, TT, (ff), cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_xlabel(r'Ambiguity parameter $\gamma$')
ax.set_ylabel(r'Uniform policy $x$')
ax.set_title(r'Objective function, universal policy')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig(pfilean+figext)
#from aniplts import *
angles = np.linspace(0,360,51)[:-1] # A list of 20 angles between 0 and 360
# # create an animated gif (20ms between frames)
rotanimate(ax, angles,pfilean+'gif',delay=20)
# # create a movie with 10 frames per seconds and 'quality' 2000
#rotanimate(ax, angles,'AToy-movie.mp4',fps=10,bitrate=2000)
# # create an ogv movie
#rotanimate(ax, angles, 'AToy-movie.ogv',fps=10)
plt.show()
plt.close(fig)


print("--- %s seconds ---" % (time.time() - start_time))
