#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import os
import glob

np.set_printoptions(linewidth=np.inf, precision=2)

def hjjn(_n, _i, _j, _phi0, _kj):
  cs = np.cos(_kj)
  if _i == _j:
    return -( sp.jv( _n, _phi0*cs )+sp.jv( _n, -_phi0*cs ) ) * np.exp( 1.0j*_n*np.pi/2 )
  elif _i+1 == _j:
    return -sp.jv( _n,_phi0 )
  elif _i-1 == _j:
    return -(-1)**_n*sp.jv( _n,_phi0 )
  else:
    return 0

# clear out old data if it exists to save room
files = glob.glob('./data/eigen*')
for f in files:
  os.remove(f)

# variable values
# mc [5-10]
mc = 10
Nm = 2*mc+1

# rc [10-20]
rc = 20
Ns = 2*rc+1

# static values
ka = 0.1
hw = 40.1

# incoming light intensity
nphi = 100
alpha = 4*ka**2/(2*np.pi*hw)
phimin = np.sqrt(0.0/alpha)
phimax = np.sqrt(5*10e-5/alpha)
phi0 = np.array( [ (phimin + i/nphi)**(1/2) for i in range(nphi) ] ) * phimax

# Calculating Floquet data set
energy = np.zeros( (Nm*Ns, nphi) )
for k in range(nphi):

# calculate the block matrices
  Hn = np.zeros( [Nm,Ns,Ns], "complex" )
  for n in range(Nm):
    for i in range(Ns):
      for j in range(Ns):
        kj = (j-rc)*ka
        Hn[n,i,j] = hjjn( -n, i, j, phi0[k], kj )

# build hermitian matrix
  Qmn = np.zeros( [Nm*Ns, Nm*Ns],"complex" )
  for i in range(Nm):
    for j in range(Nm-i):
      midx = mc-j
      r1 = (i+j)*Ns
      r2 = (i+j+1)*Ns
      c1 = j*Ns
      c2 = (j+1)*Ns
      if i == 0:
        Qmn[r1:r2,r1:r2] = Hn[i,:,:] - midx*hw*np.eye(Ns)
      else:
        Qmn[r1:r2,c1:c2] = Hn[i,:,:]


# calculate eigenvalues and vectors as a function of phi_0
  energy[:,k], states = np.linalg.eigh(Qmn)

# save eigenstates in individual text files
  np.savetxt('./data/eigenstate-phi-%03i.txt' % (k), states, fmt = '%1.8f')

# save energy matrix which is a function of phi_0
np.savetxt('./data/eng-matrix.txt', energy, fmt='%1.8f')

# save configuration file for plotting scripts
np.savetxt('./data/config-floquet.txt', [rc, mc, alpha, phimin, phimax, nphi])
