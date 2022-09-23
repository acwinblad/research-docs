#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob

# load configuration values
config = np.loadtxt('./data/config-floquet.txt')
rc = int(config[0])
mc = int(config[1])
alpha = float(config[2])
phimin = config[3]
phimax = config[4]
nphi = int(config[5])
#phi = np.array([(i/nphi)**(1/2) for i in range(nphi)])*phimax
strnphi = str(nphi)

ns = 2*rc+1
nm = 2*mc+1
m0 = (mc-0)*ns
mf = (mc+1)*ns

# load calculated values and states
energy = np.loadtxt('./data/eng-matrix.txt')
stateslist = sorted( glob.glob( './data/eigenstate-phi-*.txt') )

# calculate weight/projection on 0th order Block
weight = np.zeros( (nphi, ns*nm) )
for i, statefilename in enumerate(stateslist):
  states = np.loadtxt( statefilename, dtype=complex, skiprows=m0, max_rows=mf )
  weight[i,:] = np.diag( np.real( np.matmul( states.conj().T, states ) ) , k=0 )

#wavg = np.average(weight)
#wstd = np.std(weight)
#threshold = wavg
#weight[weight<threshold] = 0
#weight[weight>=threshold] = 1

# calculate a weighted/projected density of states as a function of phi
Emax = np.max(energy)
Emin = np.min(energy)
Emax = -2
Emin = -4.0
nE = 250
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])

# place weighted eigenvalues in an energy box-bin (also a normal dos)
wE = np.zeros((nphi,nE))
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    idx = np.where(np.logical_and(energy[:,i]>E[j],energy[:,i]<E[j+1]))[0]
    wE[i,j+1] = np.sum(weight[i,idx])
    gE[i,j+1] = np.sum(idx)

#wE[wE!=0]=1
# setup plot for weighted density of states
fig, ax = plt.subplots(1,1)

# set x-axis
xticks = np.linspace(-1,1,5, endpoint=True)
xlabelarray = np.linspace( alpha * phimin**2, alpha * phimax**2, 5, endpoint=True)
ax.set_xticks(xticks)
ax.set_xticklabels(['%1.2e' % val for val in xlabelarray])
ax.set_xlabel('$\phi_{B_{eff}} = %1.4f\phi_E^2$' % alpha)

# set y-axis
yticks = np.linspace(-1,1,5, endpoint=True)
ylabelarray = np.linspace(Emin, Emax, 5, endpoint=True)
ax.set_yticks(yticks)
ax.set_yticklabels(['%1.2f' % val for val in ylabelarray])
ax.set_ylabel('$E(\phi_E)$')

# plot and save figures
img = ax.imshow( ( np.flipud(wE.transpose()) )**(1.0), interpolation='spline16', cmap='Blues', extent=[-1,1,-1,1])
plt.savefig('./figures/dos-projection.pdf', bbox_inches='tight')

# normal dos
img = ax.imshow( (np.flipud(gE.transpose()) )**2, interpolation='spline16', cmap='Blues', extent=[-1,1,-1,1])
plt.savefig('./figures/dos-full.pdf', bbox_inches='tight')
