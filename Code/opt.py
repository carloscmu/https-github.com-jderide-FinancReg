from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory,SolverStatus,TerminationCondition

solver = 'gurobi'
#solver = 'ipopt'

def racp(S,b,a,lbda,theta):
	def S_init(model,i):
		return S[i]
	def lambda_init(model,i):
		return lbda/(1.0-S[i]*b)
	model = AbstractModel('Central Planner Risk Averse')
	I = set(S.keys())
	model.I = Set(initialize=I)
	model.lbi = Param(model.I,initialize=lambda_init)
	model.S = Param(model.I,initialize=S_init)
	model.a0 = Param(initialize=a[0])
	model.a1 = Param(initialize=a[1])
	model.theta = Param(initialize=theta)
	model.b = Param(initialize=b)
	model.x = Var(model.I, bounds=(0.0,1.0))
	model.u = Var(model.I)
	def ob_cp(model):
		ret0 = sum( model.a0-model.a1*model.u[i]*(1.0-model.S[i]*model.b/2.0) for i in model.I)
		ret1 = sum(  sum( model.S[i]*model.u[i]*model.S[j]*model.u[j] for j in model.I) for i in model.I)
		ret2 = sum( model.x[i] for i in model.I)
		return ret0-(model.theta/2.0)*ret1*((model.a0*model.b)**2.0)/12.0+1e-3*ret2
	model.ObCP = Objective(rule=ob_cp, sense=maximize)
	def conmax1(model,i):
		return model.x[i] <= model.u[i]
	model.M1 = Constraint(model.I, rule=conmax1)
	def conmax2(model,i):
		return model.lbi[i] <= model.u[i]
	model.M2 = Constraint(model.I, rule=conmax2)
	optsolver = SolverFactory(solver)
	inst = model.create_instance()
	inst.dual = Suffix(direction=Suffix.IMPORT)
	results = optsolver.solve(inst)#,tee=True)
	x = {}
	for i in I:
		x[i] = value(inst.x[i])
		print(value(inst.x[i]),value(inst.lbi[i]),value(inst.u[i]))
	return x, value(inst.ObCP)

import numpy as np

b = 0.25
S = {0:7.0/12.0, 1:2.0/3.0, 2:7.0/12.0}
lbda = 5.0/100.0
a = {0:1.0, 1:1.0}
I = set(S.keys())
xopt = {}
Ntheta = 1000
theta = np.linspace(0,100000,Ntheta)
Obj = np.zeros(Ntheta)
for i in I:
	xopt[i] = np.zeros(Ntheta)
for l in range(Ntheta):
	xaux, faux = racp(S,b,a,lbda,theta[l])
	for i in I:
		xopt[i][l] = xaux[i]
	Obj[l] = faux

import matplotlib.pyplot as plt
from matplotlib import rc

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


fig, ax1 = plt.subplots()
for i in S.keys():
	ax1.plot(theta,xopt[i])
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel(r'$x_i$')#, color='b')
#ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(theta, Obj)#, 'r.')
ax2.set_ylabel(r'RA-CP Objectuve')#, color='r')
#ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.show()

'''
plt.title('Optimal policy for risk-averse Central Planner')
plt.ylabel(r'$x_i$')
plt.xlabel(r'$\theta$')
plt.show()
'''
