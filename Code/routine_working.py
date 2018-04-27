import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import linalg as LA
import numpy as np
import nlopt

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
		s = s+epiopt(x[n],lopt[n])
	return s


Nodes = set([0,1,2])
Edges = set([(0,1),(1,2)])
Arcs = set([(0,1),(1,0),(1,2),(2,1)])

npts = len(Nodes)

lbda = 0.05
a0 = 1.0
a1 = a0
UB = 0.25
p = 1.0/npts
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

x0 = np.ones(npts)*(lbda-0.05)

opt = nlopt.opt(nlopt.LN_BOBYQA, npts)
opt.set_lower_bounds(np.zeros(npts))
opt.set_upper_bounds(np.ones(npts))
opt.set_max_objective(cpopt)
opt.set_xtol_rel(1e-8)
xopt = opt.optimize(x0)
minf = opt.last_optimum_value()
print("optimum at ", xopt)
status_sol=opt.last_optimize_result()
p_s='Code '
if status_sol <= 0.0:
	p_s=p_s+'Error'
elif status_sol ==  1:
	p_s=p_s+'Generic success return value.'
elif status_sol == 2:
	p_s=p_s+'Optimization stopped because stopval (above) was reached.'
elif status_sol == 3:
	p_s=p_s+'Optimization stopped because ftol_rel or ftol_abs (above) was reached.'
elif status_sol == 4:
	p_s=p_s+'Optimization stopped because xtol_rel or xtol_abs (above) was reached.'
elif status_sol == 5:
	p_s=p_s+'Optimization stopped because maxeval (above) was reached.'
else:
	p_s=p_s+'Optimization stopped because maxtime (above) was reached.'
print "Result code = ", p_s


##########################################################################################3

##########################################################################################3
