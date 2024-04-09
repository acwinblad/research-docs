# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:24:08 2019

@author: Hua Chen
"""
from mpl_toolkits import mplot3d
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math
import sys
import pprint as pp
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
plt.ion()

def pwSchrsquare(Ngrid, phi, facz, Vscalar, kx, ky):
    ''' Obtaining the eigensolutions of 2D Schrodinger electrons subject to a periodic 
    magnetic field forming a square Bravais lattice. Ngrid (must be an odd number) is the linear size of the momentum 
    space grid, phi is the dimensionless parameter representing the vector potential strength, and 
    tzeeman is a dimensionless parameter representing the strength of the Zeeman field accompanying 
    the magnetic field.'''
    
    Ntot = Ngrid**2
    Klist = np.zeros((Ntot, 2))
    Ncount = 0
    for i in range(Ngrid):
        for j in range(Ngrid):
            Klist[Ncount] = np.array([i - (Ngrid - 1)*0.5, j - (Ngrid -1)*0.5])
            Ncount += 1
    
    V1x = -1.0j*phi
    V1y = 1.0j*phi
    V2 = -0.25*phi**2
    V0 = phi**2
    Vz = facz*phi + 0.5*Vscalar
    
    Hmat = np.zeros((Ntot,Ntot),dtype=np.complex)
    
    for i in range(Ntot-1):
        Hmat[i,i] = (kx + Klist[i,0])**2+(ky + Klist[i,1])**2 + V0
        for j in range(i+1, Ntot):
            tK = Klist[i] - Klist[j]
            if np.abs(np.linalg.norm(tK)-1.0)<1e-6:
                Hmat[i,j] = tK[0]*V1x*(Klist[i,1] + ky) + tK[1]*V1y*(Klist[i,0] + kx) + Vz
                Hmat[j,i] = -tK[0]*V1x*(Klist[i,1] + ky) - tK[1]*V1y*(Klist[i,0] + kx) + Vz
            if np.abs(np.linalg.norm(tK)-2.0)<1e-6:
                Hmat[i,j] = V2
                Hmat[j,i] = V2
    i = Ntot-1
    Hmat[i,i] = (kx + Klist[i,0])**2+(ky + Klist[i,1])**2 + V0
    eigval, eigvec = np.linalg.eigh(Hmat)  # Hermitian--eigval is sorted in ascending order
#    print(np.matmul(np.conj(eigvec[:,0]),np.matmul(Hmat, eigvec[:,0]) ))
    return eigval, eigvec.T    
#    return Hmat    
    

def pwSchrmat(Ngrid, phi, facz, Vscalar):
    ''' generate the matrices for the Schrodinger Hamiltonian '''
    
    Ntot = Ngrid**2
    Klist = np.zeros((Ntot, 2))
    Ncount = 0
    for i in range(Ngrid):
        for j in range(Ngrid):
            Klist[Ncount] = np.array([i - (Ngrid - 1)*0.5, j - (Ngrid -1)*0.5])
            Ncount += 1
    
    V1x = -1.0j*phi
    V1y = 1.0j*phi
    V2 = -0.25*phi**2
    V0 = phi**2
    Vz = facz*phi + 0.5*Vscalar
    
    Hmat0 = np.zeros((Ntot,Ntot),dtype=np.complex)
    Hmatx = np.zeros((Ntot,Ntot),dtype=np.complex)
    Hmaty = np.zeros((Ntot,Ntot),dtype=np.complex)
    Hmatx2 = np.zeros((Ntot,Ntot),dtype=np.complex)
    Hmaty2 = np.zeros((Ntot,Ntot),dtype=np.complex)
    
    for i in range(Ntot-1):
        Hmat0[i,i] = Klist[i,0]**2+Klist[i,1]**2 + V0
        Hmatx[i,i] = 2.0*Klist[i,0]
        Hmaty[i,i] = 2.0*Klist[i,1]
        Hmatx2[i,i] = 1.0
        Hmaty2[i,i] = 1.0
        for j in range(i+1, Ntot):
            tK = Klist[i] - Klist[j]
            if np.abs(np.linalg.norm(tK)-1.0)<1e-6:
                Hmat0[i,j] = tK[0]*V1x*(Klist[i,1]) + tK[1]*V1y*(Klist[i,0]) + Vz
                Hmatx[i,j] = tK[1]*V1y
                Hmaty[i,j] = tK[0]*V1x
                Hmat0[j,i] = -tK[0]*V1x*(Klist[i,1]) - tK[1]*V1y*(Klist[i,0]) + Vz
                Hmatx[j,i] = - tK[1]*V1y
                Hmaty[j,i] = -tK[0]*V1x
            if np.abs(np.linalg.norm(tK)-2.0)<1e-6:
                Hmat0[i,j] = V2
                Hmat0[j,i] = V2
    i = Ntot-1
    Hmat0[i,i] = (Klist[i,0])**2+(Klist[i,1])**2 + V0
    Hmatx[i,i] = 2.0*(Klist[i,0])
    Hmaty[i,i] = 2.0*(Klist[i,1])
    Hmatx2[i,i] = 1.0
    Hmaty2[i,i] = 1.0
    return Hmat0, Hmatx, Hmaty, Hmatx2, Hmaty2   
#    return Hmat 


def bands(direction=np.array([1, 0]), Ngrid = 5, phi = 0.1, facz = 0, Vscalar = 0, Nband=6, knum=61): 
    '''plot band structure'''
    klist = np.linspace(-1.0, 1.0, knum)   
    k1dlist = np.linspace(-1.0*np.linalg.norm(direction), np.linalg.norm(direction), knum)   
    eiglist = np.zeros((knum, Nband))
    for i in range(knum):
        teig = pwSchrsquare(Ngrid, phi, facz, Vscalar, klist[i]*direction[0], klist[i]*direction[1])
        eiglist[i,:] = np.sort(np.real(teig[0]))[:Nband]    
    bandmat = np.zeros((knum*Nband,2))
    for i in range(Nband):
        bandmat[knum*i:knum*(i+1),0] = k1dlist.T 
        bandmat[knum*i:knum*(i+1),1] = eiglist[:,i]
    x, y = bandmat.T
#    print(bandmat)
    pos = np.where(np.abs(np.diff(x))>max(np.abs(direction)))[0]
    x[pos] = np.nan
    y[pos] = np.nan
    plt.axes([0.2,0.2,0.6,0.6])
    plt.plot(x,y,color='k')
    plt.xlim([min(x),-min(x)])
#    plt.ylim([-0.5,+0.5]) # Adjust this range as needed
    plt.ylim([min(y)-0.5,max(y)+0.5]) # Adjust this range as needed    
    plt.ylabel('Energy (eV)')
    plt.xlabel('$k$ ($K$)')
    plt.tick_params(axis='x',which='both',top='off')
    outfile = 'pwSchrsquare.pdf'
    plt.savefig(outfile,bbox_inches='tight') 
    
def gen_nnkp(nkps, pbc = 1):
    ''' Generate a regular k-mesh and obtain the nearest neighbors of each k point.
    nkps must be an odd number in order for kmesh to include (0,0). pbc controls the
    boundary condition. pbc = 1 is the usual periodic boundary condition so that the 
    Brillouin zone is a torus. When pbc = 0 the k points with neighbors translated by
    a finite lattice vector will have these neighbors labeled by -ik-1, where ik is the 
    label of the neighbour in the opposite direction, and the -1 is needed because ik 
    may be 0.'''
    # generate k mesh first, each direction has nkps steps
    kmesh = np.zeros((nkps**2, 2))
    ncount = 0
    kstep = 1.0/nkps
    kshift = np.array([-kstep,-kstep])*0.5*(nkps-1)
    for i in range(nkps):
        for j in range(nkps):
            kmesh[ncount] = np.array([i*kstep, j*kstep])+kshift
            ncount += 1
    
    # find nearest neighbors
    nnkp = np.zeros((nkps**2, 5),dtype=np.int)
    bvec = np.array([[0, 1.0], [1.0, 0], [0, -1.0], [-1.0, 0]])
    bvec *= kstep
    # print(bvec)
    Knnvec = np.array([[0,1.0],[0,-1.0],[1.0,0],[-1.0,0],[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]])
    for i in range(nkps**2):
        k0 = kmesh[i]
        if nnkp[i,4] < 4:  # if a kpoint has less than four neighbors
            for j in range(nkps**2):
                k1 = kmesh[j]
                k01 = k1 - k0
                for ii in range(4):
                    if np.linalg.norm(k01-bvec[ii])<1e-6:
                        nnkp[i, ii] = j
                        nnkp[i, 4] += 1
                        break
                if nnkp[i, 4] >= 4:
                    break
    # the only k points with less than 4 neighbors are those at the boundaries
    for i in range(nkps**2):
        k0 = kmesh[i]
        if nnkp[i, 4] < 4:
            for j in range(nkps**2):
                k1 = kmesh[j]
                for jj in range(8):
                    k01 = k1 + Knnvec[jj] - k0
                    for ii in range(4):
                        if np.linalg.norm(k01-bvec[ii])<1e-6:
                            if np.abs(pbc - 1) < 1e-6:                                
                                nnkp[i, ii] = j                                
                            else:
                                if ii > 1:
                                    nnkp[i, ii] = -1 - nnkp[i, ii-2]
                                else:
                                    nnkp[i, ii] = -1 - nnkp[i, ii+2]
                            nnkp[i, 4] += 1    
                            break
                    else:
                        continue
                    break
                if nnkp[i, 4] >= 4:
                    break
    
    #save kmesh, nnkp, and bvec
    np.savetxt('kmesh_'+str(nkps)+'.dat', kmesh, fmt='%12.6f')
    np.savetxt('nnkp_'+str(nkps)+'.dat', nnkp, fmt='%10d') 
    np.savetxt('bvec_'+str(nkps)+'.dat', bvec, fmt='%12.6f')
    return kmesh, bvec, nnkp

def wannier_init(nkps, nbands, whichband = np.array([]), Ngrid = 5, phi = 0.1, facz = 0, Vscalar = 0, rcenter = np.array([]), pbc = 1):
    ''' Initialization for wannierisation. Generate kmesh, nnkp, unk, eigenvalues, and mmn matrix. 
    Update -- also give the initial umn matrix by projecting the Bloch states to nbands Gaussian functions.
    The centers of the gaussian functions are specified by rcenter. '''
    
    # generate kmesh and nnkp
    kmesh = np.zeros((nkps**2, 2))
    bvec = np.zeros((4, 2))
    nnkp = np.zeros((nkps**2, 5),dtype=np.int)
    kmesh, bvec, nnkp = gen_nnkp(nkps, pbc)
    
    # calculate eigenvalues and eigenfunctions of the specified bands
    eigval = np.zeros((nkps**2, nbands))  # eigenvalues
    unk = np.zeros((nkps**2, nbands, Ngrid**2),dtype=np.complex)  # eigenfunctions (periodic part of the Bloch function)
    for i in range(nkps**2):
        tkx, tky = kmesh[i]
        teigval, teigvec = pwSchrsquare(Ngrid, phi, facz, Vscalar, tkx, tky)
        eigval[i] = teigval[whichband]
        unk[i] = teigvec[whichband]
#        print(teigval)
#        for ii in range(nbands):
#            for jj in range(Ngrid**2):
#                if np.abs(teigval[jj] - eigval[i, ii]) < 1e-6:
#                    unk[i, ii] = teigvec[jj]
#                    break
#    print(np.shape(teigvec))
#    print(teigvec[0])
    #store eigval and unk
    np.savetxt('eigval_'+str(nkps)+'.dat', eigval, fmt='%16.10f')
    np.savetxt('unk_'+str(nkps)+'.dat', unk.reshape((nkps**2,nbands*Ngrid**2)).view(float), fmt='%16.10f')
    
    # if loading from a file is needed
    # tunk=np.loadtxt('unk_'+str(nkps)+'.dat').view(complex).reshape((nkps**2,nbands,Ngrid**2))
    
    # calculate the overlap (mmn) matrix
    nntot = 4 # 4 nearest neighbors for square lattice
    mmn = np.zeros((nkps**2, nntot, nbands, nbands),dtype=np.complex)
    if np.abs(pbc - 1) < 1e-6:      # periodic boundary condition in momentum space
        for ik in range(nkps**2):
            for ib in range(nntot):
                mkb = np.zeros((nbands,nbands),dtype=np.complex) # temporary matrix for storing Mmn matrix for each k and b
                ikb = nnkp[ik,ib]
                for m in range(nbands):
                    for n in range(nbands):
                        tumk = np.conj(unk[ik,m])
                        tunkb = unk[ikb,n]
                        mkb[m,n] = np.inner(tumk,tunkb)
                mmn[ik,ib] = mkb
    else:
        for ik in range(nkps**2):
            for ib in range(nntot):
                mkb = np.zeros((nbands,nbands),dtype=np.complex) # temporary matrix for storing Mmn matrix for each k and b
                ikb = nnkp[ik,ib]
                if ikb < 0:
                    ikbb = -1-ikb  # since ikb = -1 - ik defined in gen_nnkp
                    for m in range(nbands):
                        for n in range(nbands):
                            tumk = np.conj(unk[ik,m])
                            tunkb = unk[ikbb,n]
                            mkb[m,n] = np.dot(tumk,tunkb)
                    mmn[ik,ib] = 2.0*np.eye(nbands,dtype=np.complex) - mkb
                else:                
                    for m in range(nbands):
                        for n in range(nbands):
                            tumk = np.conj(unk[ik,m])
                            tunkb = unk[ikb,n]
                            mkb[m,n] = np.dot(tumk,tunkb)
                    mmn[ik,ib] = mkb         
    # store the mmn matrix
    np.savetxt('mmn_'+str(nkps)+'.dat', mmn.reshape((nkps**2,nntot*nbands**2)).view(float), fmt='%16.10f')                
    # if loading from a file is needed
    # mmn = np.loadtxt('mmn_'+str(nkps)+'.dat').view(complex).reshape((nkps**2,nntot,nbands,nbands))
    
    # calculate the projection of unk to gaussians
    umn = np.zeros((nkps**2, nbands, nbands), dtype = np.complex)
    gsigma = 8  # width of the gaussian function, change if needed
    # Need Klist
    Ntot = Ngrid**2
    Klist = np.zeros((Ntot, 2))
    Ncount = 0
    for i in range(Ngrid):
        for j in range(Ngrid):
            Klist[Ncount] = np.array([i - (Ngrid - 1)*0.5, j - (Ngrid -1)*0.5])
            Ncount += 1                
            
    for ik in range(nkps**2):
        tk = kmesh[ik]
        tU = np.zeros((nbands,nbands),dtype=np.complex)
        for nn in range(nbands):
            tr = rcenter[nn]
            for mm in range(nbands):
                tunk = unk[ik,mm]
                for iKK in range(Ngrid**2):
                    tkK = tk + Klist[iKK]
                    fac = -1.0j*np.inner(tkK,tr) - 0.5*gsigma**2*np.inner(tkK,tkK)
                    tU[mm,nn] += np.conj(tunk[iKK])*np.exp(fac)
        # orthonormalization
#        print(tU)
        tUQ, tUR = np.linalg.qr(tU)
        tQRsign = np.diag(np.sign(np.diag(tUR))) 
        umn[ik] = np.matmul(tUQ,tQRsign)  #otherwise the QR decomposition is not unique
#        if nbands == 1:
#            umn[ik] = tU/np.abs(tU)
#    print(rcenter)
        
    return kmesh, bvec, nnkp, eigval, unk, mmn, umn    

def wannier_omega(nkps, nbands, Ngrid, mmn, bvec, csheet, sheet):
    ''' Calculate the spread '''
    
    rn = np.zeros((nbands,2)) # average of position
    rn2 = np.zeros(nbands) # average of position
    r2n = np.zeros(nbands) # average of position squared
    wb = 1.0/(2.0*np.linalg.norm(bvec[1])**2) # Eq. B1 of PRB 56, 12847 (1997).
    
    for ik in range(nkps**2):
        for ib in range(4):
            for iband in range(nbands):
                tMnn = mmn[ik, ib, iband, iband]
                tImlnMnn = np.imag(np.log(tMnn * csheet[ik,ib,iband])) - sheet[ik,ib,iband]
                rn[iband] += wb * tImlnMnn * bvec[ib]  # Eq. 31
                r2n[iband] += wb * ( 1.0 - np.abs(tMnn)**2 + tImlnMnn**2 )
    rn *= -1.0/nkps**2
    r2n *= 1.0/nkps**2
    for iband in range(nbands):
        rn2[iband] = np.inner(rn[iband],rn[iband])
    omegan = r2n - rn2
    
    # gauge invariant part of omega
#    omegaI = 0.0
#    for ik in range(nkps**2):
#        for ib in range(4):
#            omegaI += nbands
#            for ib1 in range(nbands):
#                for ib2 in range(nbands):
#                    omegaI -= np.abs(mmn[ik,ib,ib1,ib2])**2
#    omegaI *= wb/nkps**2
#    print('Omega_I = %f' % (omegaI))
    
    omega_od = 0.0
    for ik in range(nkps**2):
        for ib in range(4):
            for m in range(nbands):
                for n in range(nbands):
                    if np.abs(m-n) > 0.1:
                        omega_od += wb* (np.abs(mmn[ik,ib,m,n]))**2
    omega_od *= 1.0/nkps**2
    
    omega_d = 0.0
    for ik in range(nkps**2):
        for ib in range(4):
            for n in range(nbands):
                brn = np.inner(bvec[ib],rn[n])
                tMnn = mmn[ik, ib, n, n]
                tImlnMnn = np.imag(np.log(tMnn * csheet[ik,ib,n])) - sheet[ik,ib,n]
                omega_d += wb* (tImlnMnn + brn)**2
    omega_d *= 1.0/nkps**2
    
    print(omega_od, omega_d, np.sum(omegan)-omega_d-omega_od)
    return rn, omegan

def wannier_omegaI(nkps, nbands, mmn, wb):
    ''' Calculate the gauge invariant spread '''
    
    # gauge invariant part of omega
    omegaI = 0.0
    for ik in range(nkps**2):
        for ib in range(4):
            omegaI += nbands
            for ib1 in range(nbands):
                for ib2 in range(nbands):
                    omegaI -= np.abs(mmn[ik,ib,ib1,ib2])**2
    omegaI *= wb/nkps**2    
    return omegaI

    
def wannier_domega(nkps, nbands, Ngrid, mmn, bvec, csheet, sheet):
    ''' Calculate the gradient of the spread '''

    rn = np.zeros((nbands,2)) # average of position
    wb = 1.0/(2.0*np.linalg.norm(bvec[1])**2) # Eq. B1 of PRB 56, 12847 (1997).
    domega = np.zeros((nkps**2,nbands,nbands),dtype=np.complex)
    
    # calculate rn
    for ik in range(nkps**2):
        for ib in range(4):
            for iband in range(nbands):
                tMnn = mmn[ik, ib, iband, iband]
                tImlnMnn = np.imag(np.log(tMnn * csheet[ik,ib,iband])) - sheet[ik,ib,iband]
                rn[iband] += wb * tImlnMnn * bvec[ib]  # Eq. 31
    rn *= -1.0/nkps**2    
    
    # calculate G -- Eq. 52
    for ik in range(nkps**2):
        for ib in range(4):            
            tmmn = mmn[ik,ib]
            tmnn = np.diag(np.diag(tmmn))
            timnn = np.diag(1.0/np.diag(tmmn)) # inverse of tmnn
            Rmn = np.matmul(tmmn,np.conj(tmnn))   # calculate R -- Eq. 45
            qnn = np.diag(np.imag(np.log(np.diag(tmmn)))) + np.diag(np.dot(rn, bvec[ib]))    # calculate qn -- Eq. 47
            Rtmn = np.matmul(tmmn, timnn)  # calculate Rtilde -- Eq. 48
            Tmn = np.matmul(Rtmn, qnn)  # calculate T -- Eq. 51
            domega[ik] += wb * ( 0.5*(Rmn - np.conj(np.transpose(Rmn))) + 0.5j*(Tmn + np.conj(np.transpose(Tmn))) )
    domega *= 4.0/nkps**2
    
    return domega            

def wannier_cg(domega,domega0,g0norm,it,ng,nkps,nbands,wb):
    ''' conjugate gradient using the Fletcher-Reeves formula '''
    g1 = domega.reshape((nkps**2*nbands**2))
    g1norm = np.real(np.dot(np.conj(g1),g1))
#    print(g1norm)
    # calculate CG coefficient    
    if it==0 or ng >= 5:
        gfac = 0.0
        ng = 0
    else:
        if g0norm > 0.0:
            gfac = g1norm/g0norm
#            print(gfac)
            if gfac > 3.0:
                gfac = 0.0
                ng = 0
            else:
                ng += 1
        else:
            gfac = 0.0
            ng = 0
    g0norm = g1norm

    # calculate search direction
    domega1 = domega + domega0 * gfac
#    print(gfac)    
    # calculate gradient along search direction
    g0 = -1.0*np.real(np.dot(np.conj(g1),domega1.reshape(nkps**2*nbands**2)))
    g0 /= 4.0*4.0*wb    # 4 wb with the same values
#    print(g0)
    # check search direction is not uphill
    if g0 > 0.0:  # uphill
        if ng > 0:
            domega1 = domega
            ng = 0
            gfac = 0.0
            # recalculate gradient along search direction
            g0 = -np.real(np.dot(np.conj(g1),domega1.reshape(nkps**2*nbands**2)))
            g0 /= 4.0*4.0*wb    # 4 wb with the same values
            # if still uphill, reverse
            if g0 > 0.0:
                domega1 *= -1.0
                g0 *= -1.0
        else:
            domega1 *= -1.0
            g0 *= -1.0
    
    # return
    return g0norm, g0, domega1, ng

def wannier_phase(rguide, irguide, nbands, nkps, mmn, bvec):
    ''' Use guiding centers, basically following the wannier90 code '''
    sheet = np.zeros((nkps**2,4,nbands))
    csheet = np.zeros((nkps**2,4,nbands),dtype=np.complex)
    for iw in range(nbands):
        csum = np.zeros(4,dtype=np.complex)
        # get average phase for each bvec
        for ik in range(nkps**2):
            for inn in range(4):
                csum[inn] += mmn[ik, inn, iw, iw]
        # Find a vector rguide so that the phase of csum[inn] is approximately the phase
        # exp[ -i bvec[inn].rguide]. 
        # Define xx[inn] = -Im Ln csum[inn] mod 2pi, then bvec[inn].rguide ~ xx[inn]
        # Take an arbitrary branch cut for the first three xx[inn], and determine rguide
        # from these. For other k determine the most consistent branch cut and then update rguide
        # rguide is obtained by minimizing sum_inn [bvec[inn].rguide - xx[inn]]^2 or
        # by making the derivative vanish:
        # sum_j smat[i,j] * rguide[iw, j] = svec[i]
        # where smat[i,j] = sum_inn bvec[inn,i]*bvec[inn,j] 
        # svec[i] = sum_inn bvec[i,inn]*xx[inn]
        
        smat = np.zeros((2,2))
        svec = np.zeros(2)
        xx = np.zeros(4)

        for inn in range(4):
            if inn < 2:
                xx[inn] = -np.imag(np.log(csum[inn]))
            else:
                xx0 = np.inner(bvec[inn],rguide[iw])
                csumt = np.exp(1.0j * xx0)
                xx[inn] -= np.imag(np.log(csum[inn]*csumt)) 
            for i in range(2):
                for j in range(2):
                    smat[i,j] += bvec[inn,i]*bvec[inn,j]
                svec[i] += bvec[inn, i]*xx[inn]
            if inn >= 2:
                invs = np.linalg.inv(smat)
                if irguide > 0:
                    rguide[iw] = np.matmul(invs, svec)
                        
    #obtain branch cut choice using rguide
    for ik in range(nkps**2):
        for inn in range(4):
            for iw in range(nbands):
                sheet[ik,inn,iw] = np.inner(bvec[inn],rguide[iw])
    csheet = np.exp(1.0j*sheet)
    
    # now check the proper sheet is picked for LnMmn: 
    # rnkb = Im Ln Mnn + bvec.rguide is approximately 0 if good
    rnkb = np.zeros((nkps**2, 4, nbands))
    for ik in range(nkps**2):
        for inn in range(4):
            for iw in range(nbands):
                rnkb[ik, inn, iw] += np.inner(bvec[inn], rguide[iw])
    return csheet, sheet, rguide
                
            
    
    
def wannier_update(dW,mmn,umn,nnkp,nkps,nbands,wb,pbc = 1):
    ''' update unitary rotation Umn and Mmn matrices '''
    
    # check anti-hermitian
#    tr = np.zeros((nbands,nbands),dtype=np.complex)
#    for ik in range(nkps**2):
#        expdw = sl.expm(domega[ik])
#        tr += np.matmul(np.conjugate(np.transpose(expdw)),expdw)
#    tr /= nkps**2
#    print(tr)
    
    # generate rotation matrices and update Umn
    du = np.zeros((nkps**2,nbands,nbands),dtype=np.complex)
    for ik in range(nkps**2):        
        expdw = sl.expm(dW[ik])
        du[ik] = expdw
        umnk = np.matmul(umn[ik],expdw)
        umn[ik] = umnk
    
    #update Mmn
    if np.abs(pbc - 1) < 1e-6:      # periodic boundary condition in momentum space
        for ik in range(nkps**2):
            for ib in range(4):
                ikb = nnkp[ik,ib]
                tmmn = mmn[ik,ib]
                dumnkc = np.conj(np.transpose(du[ik]))
                dumnkb = du[ikb]
                mmn[ik,ib] = np.matmul(dumnkc,np.matmul(tmmn,dumnkb))
    else:
        for ik in range(nkps**2):
            for ib in range(4):
                ikb = nnkp[ik,ib]
                tmmn = mmn[ik,ib]
                dumnkc = np.conj(np.transpose(du[ik]))
                if ikb < 0:
                    ikbb = -1-ikb  # since neighbor is labeled by - ik -1 in gen_nnkp
                    dumnkb = du[ikbb]
                else:
                    dumnkb = du[ikb]
                mmn[ik,ib] = np.matmul(dumnkc,np.matmul(tmmn,dumnkb))            
    return umn, mmn    
    
def wannier_findR(rn,nbands):
    ''' find the Bravais lattice vector of each wannier function ''' 
    nR = np.zeros((nbands,2)) # R vector of each wannier function
    maxR = 5 # maximum of R vector for searching
    origin = np.array([-np.pi, -np.pi])
    a1vec = np.array([2*np.pi, 0.0])
    a2vec = np.array([0.0, 2*np.pi])
    cell0 = Polygon([tuple(origin),tuple(a1vec+origin),tuple(a1vec+a2vec+origin),tuple(a2vec+origin)])
    for iw in range(nbands):
        tr = rn[iw]
        tpoint = Point(tr)
        if not (cell0.contains(tpoint)):
            icount = 0
            for i in np.arange(-maxR,maxR+1):
                for j in np.arange(-maxR,maxR+1):
                    if np.abs(i)+np.abs(j) > 1e-6:
                        dR = i*a1vec + j*a2vec
                        tpoint = Point(tr - dR)
                        if cell0.contains(tpoint):
                            nR[iw] = dR
                            break
                        icount += 1
                else:
                    continue
                break
            if icount == maxR**2-1:
                print('Error: increase maxR')
    return nR, a1vec, a2vec

def wannier_shiftR(rn, nbands, umn, mmn, nkps, kmesh, nnkp, bvec):
    ''' Shift all wannier functions to the unit cell at origin. ''' 
    nR = np.zeros((nbands,2)) # R vector of each wannier function
    nR, a1vec, a2vec = wannier_findR(rn, nbands)
#    print(nR)
    ushift = np.zeros((nkps**2,nbands),dtype=np.complex)
    for ik in range(nkps**2):
        tk = kmesh[ik]
        for iw in range(nbands):
            tR = nR[iw]
            fac = np.exp(1.0j*np.inner(tk,tR))
            ushift[ik, iw] = fac
            umn[ik,:,iw] *= fac
    
    # calculate the new wannier center
    rnnew = np.zeros((nbands,2)) # average of position
    wb = 1.0/(2.0*np.linalg.norm(bvec[1])**2) # Eq. B1 of PRB 56, 12847 (1997).
    for ik in range(nkps**2):
        for ib in range(4):
            ikb = nnkp[ik, ib]
            for iband in range(nbands):
                tMnn = mmn[ik, ib, iband, iband]
                tMnn *= np.conj(ushift[ik,iband])*ushift[ikb,iband]
                tImlnMnn = np.imag(np.log(tMnn))
                rnnew[iband] += wb * tImlnMnn * bvec[ib]  # Eq. 31
    rnnew *= -1.0/nkps**2    
            
    return rnnew, umn
    
def wannier_hamiltonian(umn, eigval, nkps, nbands, kmesh):
    ''' Calculate the tight-binding Hamiltonian in the wannier function basis.
    <0m|H|Rn> = \sum_k U^\dag E_k U e^{-i k \cdot R}/nkps**2. For a discrete sum over k 
    the maximum R is the inverse of the nearest neighbor distance of the k mesh.
    '''
    
    # Generate R grid
    Rgrid = np.int(nkps*0.5)*2-1 # Use a smaller WS cell
    Rmesh = np.zeros((Rgrid**2, 3))
    Ncount = 0
    for i in range(Rgrid):
        for j in range(Rgrid):
            tR = np.array([i - 0.5*(Rgrid-1), j - 0.5*(Rgrid-1)])*2*np.pi
            Rmesh[Ncount,2] = np.linalg.norm(tR)
            Rmesh[Ncount,0:2] = tR
            Ncount += 1
            
    # sort Rgrid by |R|
    Rmesh = Rmesh[np.argsort(Rmesh[:,2])]  # sort by the last column
#    print(Rmesh[:,0:2])
    # Fourier transform H
    hmatr = np.zeros((Rgrid**2, nbands, nbands), dtype = np.complex)
    for iR in range(Rgrid**2):
        tR = Rmesh[iR,0:2]
        for ik in range(nkps**2):
            tk = kmesh[ik]
            fac = np.exp(-1.0j*np.inner(tk,tR))
            tU = umn[ik]
            tUc = np.conj(np.transpose(tU))
            if nbands > 1:
                teig = np.diag(eigval[ik])
            else:
                teig = np.array([eigval[ik]])
            hmatr[iR] += np.matmul(tUc,np.matmul(teig,tU))*fac
    hmatr /= nkps**2
            
    # return the real space Hamiltonian in wannier basis
    return hmatr, Rmesh, Rgrid

def wannier_utran(umn, unk, nkps, nbands, Ngrid):
    ''' Rotate unk to the wannier gauge by multiplying umn on it '''
    unkw = np.zeros((nkps**2, nbands, Ngrid**2),dtype=np.complex) 
    for ik in range(nkps**2):
        for iw in range(nbands):
            for iband in range(nbands):
                unkw[ik, iw] += unk[ik, iband]*umn[ik, iband, iw]
    return unkw

def wannier_wfreal(unkw, Klist, kmesh, Ngrid, nkps, rx, ry):
    ''' return the value of the real space Wannier function at the given position rvec. '''
    wr = 0.0j
    for iG in range(Ngrid**2):
        for ik in range(nkps**2):
            tkG = kmesh[ik] + Klist[iG]
            wr += unkw[ik, iG]*np.exp(1.0j* (tkG[0]*rx + tkG[1]*ry))
#    nk0=np.int((nkps+1)*(nkps-1)/2)
#    for iG in range(Ngrid**2):
#        tkG = Klist[iG]
#        wr += unkw[nk0,iG]*np.exp(1.0j* (tkG[0]*rx + tkG[1]*ry))

#    maxR = 1
#    for iG in range(Ngrid**2):
#        for ik in range(nkps**2):
#            tkG = kmesh[ik] + Klist[iG]
#            for ri in range(-maxR,maxR+1):
#                for rj in range(-maxR,maxR+1):
#                    tRx = 2*np.pi*ri
#                    tRy = 2*np.pi*rj
#                    wr += unkw[ik, iG]*np.exp(1.0j* (tkG[0]*(rx-tRx) + tkG[1]*(ry-tRy)))               
    wr /= nkps
    return wr
            
def wannier_plot(maxR, nkps, nbands, Ngrid, iw = -1):
    ''' Plot wannier functions (labeled by iw) in real space '''
    plt.rcParams.update({'font.size': 18})    
    # Need Klist
    Ntot = Ngrid**2
    Klist = np.zeros((Ntot, 2))
    Ncount = 0
    for i in range(Ngrid):
        for j in range(Ngrid):
            Klist[Ncount] = np.array([i - (Ngrid - 1)*0.5, j - (Ngrid -1)*0.5])
            Ncount += 1
#    print(Klist)
    
    # Need kmesh
    kmesh = np.zeros((nkps**2,2))
    kmesh = np.loadtxt('kmesh_'+str(nkps)+'.dat') 
    
    # Need unk
    unk = np.zeros((nkps**2, nbands, Ngrid**2),dtype=np.complex) 
    unk = np.loadtxt('unk_'+str(nkps)+'.dat').view(complex).reshape((nkps**2,nbands,Ngrid**2))
    
    # Need umn
    umn = np.zeros((nkps**2, nbands, nbands),dtype=np.complex)
    umn = np.loadtxt('umn_'+str(nkps)+'.dat').view(complex).reshape((nkps**2,nbands,nbands))   

    # rotate unk to wannier gauge
    unkw = wannier_utran(umn, unk, nkps, nbands, Ngrid)            
            
    # main plotting routine
    
    if iw < 0:  # plot all wannier functions if iw is negative
        istart = 0
        iend = nbands
    else:
        istart = iw
        iend = iw + 1
        
    xlim = maxR*2*np.pi
    ylim = maxR*2*np.pi
    x = np.linspace(-xlim, xlim, 60)
    y = np.linspace(-ylim, ylim, 60)
    
    X, Y = np.meshgrid(x, y)
    Xr = np.ravel(X)
    Yr = np.ravel(Y)
    wr = np.zeros((Xr.size),dtype=np.complex)
    
    for iplot in np.arange(istart, iend):
        tunk = unkw[:,iplot,:]
        for i in range(X.size):            
            wr[i] = wannier_wfreal(tunk, Klist, kmesh, Ngrid, nkps, Xr[i], Yr[i])
    #        wr[i] = np.sin(Xr[i]) + np.sin(Yr[i]) # test
        wrnorm = np.abs(wr).reshape(X.shape)    
        wrphase = np.angle(wr).reshape(X.shape)  
             
        plt.figure(1000,figsize=(6,6))
        ax=plt.gca()
        ax.set_aspect('equal')
        plt.contourf(X,Y, wrnorm, 20, cmap = 'hot')
        ax.set_facecolor('black')
        plt.colorbar(fraction=0.046, pad=0.04)
        xcap = 'x (1/K)'
        ycap = 'y (1/K)'
        plt.ylabel(ycap)
        plt.xlabel(xcap)                    
        outfile = 'wf_norm_'+ str(nkps) + '_' + str(iplot) + '.pdf'
        plt.savefig(outfile,bbox_inches='tight') 
        plt.close(1000)
        
        plt.figure(1001,figsize=(6,6))
        ax=plt.gca()
        ax.set_aspect('equal')
        plt.contourf(X,Y, wrphase, 20, cmap = 'hot')
        ax.set_facecolor('black')
        plt.colorbar(fraction=0.046, pad=0.04)
        xcap = 'x (1/K)'
        ycap = 'y (1/K)'
        plt.ylabel(ycap)
        plt.xlabel(xcap)                    
        outfile = 'wf_phase_'+ str(nkps) + '_' + str(iplot) + '.pdf'
        plt.savefig(outfile,bbox_inches='tight') 
        plt.close(1001)
    
    
def wannier_interpolation(nkps, nbands, whichband, Ngrid, phi, facz, Vscalar, nkpath = 100):
    ''' Plot interpolated band structure and compare with the plane wave results. '''
    plt.rcParams.update({'font.size': 18})  
    # Need Klist
    Ntot = Ngrid**2
    Klist = np.zeros((Ntot, 2))
    Ncount = 0
    for i in range(Ngrid):
        for j in range(Ngrid):
            Klist[Ncount] = np.array([i - (Ngrid - 1)*0.5, j - (Ngrid -1)*0.5])
            Ncount += 1
#    print(Klist)
    
    # Need kmesh
    kmesh = np.zeros((nkps**2,2))
    kmesh = np.loadtxt('kmesh_'+str(nkps)+'.dat') 
    
    # Need umn
    umn = np.zeros((nkps**2, nbands, nbands),dtype=np.complex)
    umn = np.loadtxt('umn_'+str(nkps)+'.dat').view(complex).reshape((nkps**2,nbands,nbands))   
    
    # Need eigval
    eigval = np.zeros((nkps**2,nbands))
    eigval = np.loadtxt('eigval_'+str(nkps)+'.dat')

    # Calculate the real space Hamiltonian
    hmatr, Rmesh, Rgrid = wannier_hamiltonian(umn, eigval, nkps, nbands, kmesh)            
            
    # main plotting routine
    
    KG = np.array([0,0])
    KX = np.array([0.5,0])
    KM = np.array([0.5,0.5])
    Kpathseg = np.array([KG, KX, KM, KG])
    Kseglength = np.zeros(Kpathseg.shape[0]-1)
    for i in range(Kpathseg.shape[0]-1):
        Kseglength[i] = np.linalg.norm(Kpathseg[i+1] - Kpathseg[i])
    kltot = np.sum(Kseglength)
    
    kpath = np.zeros((nkpath*2,2))
    kpathl = np.zeros(nkpath*2)
    ncount = 0
    tklength = 0.0
    for i in range(Kpathseg.shape[0]-1):
        tkstepnum = np.round(Kseglength[i]/kltot*nkpath).astype(int)
        tkstep = Kseglength[i]/tkstepnum
        for j in range(tkstepnum):
            tk = Kpathseg[i] + j* (Kpathseg[i+1]-Kpathseg[i])/tkstepnum          
            kpath[ncount] = tk
            kpathl[ncount] = tklength
            ncount += 1
            tklength += tkstep
    nktotfinal = ncount
    
    eigval_pw = np.zeros((nktotfinal,nbands))
    eigval_wan = np.zeros((nktotfinal,nbands))

    for ik in range(nktotfinal):
        tk = kpath[ik] 
        # eigenvalues from plane wave code
        teigval, teigvec = pwSchrsquare(Ngrid, phi, facz, Vscalar, tk[0], tk[1])
        eigval_pw[ik] = np.sort(np.real(teigval))[whichband]
        #eigenvalues from interpolated Hamiltonian
        thmat = np.zeros((nbands,nbands),dtype = np.complex)
        for ir in range(Rgrid**2):
            tr = Rmesh[ir, 0:2]
            fac = np.exp(1.0j*np.inner(tk,tr))
            thmat += fac * hmatr[ir]
        thmat = 0.5* (thmat + np.conj(np.transpose(thmat)))
        eigval_wan[ik] = np.sort(np.real(np.linalg.eigvals(thmat)))
    print(eigval_wan)
    bandmat = np.zeros((nktotfinal*nbands,3))
    for i in range(nbands):
        bandmat[nktotfinal*i:nktotfinal*(i+1),0] = kpathl[0:nktotfinal].T 
        bandmat[nktotfinal*i:nktotfinal*(i+1),1] = eigval_pw[:,i]
        bandmat[nktotfinal*i:nktotfinal*(i+1),2] = eigval_wan[:,i]
    x, y, z = bandmat.T
#    print(bandmat)
    plt.figure(1000,figsize=(8,6))
    tick_labels=[]
    tick_locs=[]
    tick_labels.append('$\Gamma$')
    tick_locs.append(0)
    tick_labels.append('X'.strip())
    tick_locs.append(Kseglength[0])
    tick_labels.append('R'.strip())
    tick_locs.append(Kseglength[0] + Kseglength[1])
    tick_labels.append('$\Gamma$'.strip())
    tick_locs.append(Kseglength[0] + Kseglength[1]+ Kseglength[2])
    pos = np.where(np.abs(np.diff(x))> 1.0)[0]
    x[pos] = np.nan
    y[pos] = np.nan
    z[pos] = np.nan
    plt.axes([0.2,0.2,0.6,0.6])
    plt.xlim([min(x),max(x)])
    plt.ylim([0.1,max(y)+0.1]) # Adjust this range as needed
#    plt.ylim([min(y)-0.5,max(y)+0.5]) # Adjust this range as needed   
    plt.plot(x,y,color='k')
    plt.plot(x,z,color='r')
    plt.xticks(tick_locs,tick_labels)
    for n in range(1,len(tick_locs)):
        plt.plot([tick_locs[n],tick_locs[n]],[plt.ylim()[0],plt.ylim()[1]],color='gray',linestyle='-',linewidth=0.5) 
    plt.ylabel('Energy ($\hbar^2K^2/2m$)')
    plt.xlabel('$k$ ($K$)')
    plt.tick_params(axis='x',which='both',top='off')
    outfile = 'wann_interpolation.pdf'
    plt.savefig(outfile,bbox_inches='tight')     
            
 

def wannier_main(nkps = 19, nbands = 4, whichband = np.array([0,1,4,5]), rcenter = np.array([[0,0],[0,np.pi],[np.pi,0],[np.pi,np.pi]]), Ngrid = 9, phi = 0.5, facz = 0, Vscalar = 0, nstep = 500, steplength = 1.0, pbc = 1):
    ''' main wannierisation routine '''    
    nntot = 4
    kmesh = np.zeros((nkps**2, 2))
    bvec = np.zeros((4, 2))
    nnkp = np.zeros((nkps**2, 5),dtype=np.int)  
    eigval = np.zeros((nkps**2, nbands))  
    unk = np.zeros((nkps**2, nbands, Ngrid**2),dtype=np.complex)    
    mmn = np.zeros((nkps**2, nntot, nbands, nbands),dtype=np.complex)
    umn = np.zeros((nkps**2, nbands, nbands),dtype=np.complex)

    # initialization
    kmesh, bvec, nnkp, eigval, unk, mmn, umn = wannier_init(nkps, nbands, whichband, Ngrid, phi, facz, Vscalar, rcenter, pbc)
    
#    print(bvec)
    wb = 1.0/(2.0*np.linalg.norm(bvec[1])**2) # Eq. B1 of PRB 56, 12847 (1997).
#    print(wb)
    domega0 = np.zeros((nkps**2,nbands,nbands),dtype=np.complex)
    g0norm = 0.0
    ng = 0

    
    mmn0 = mmn.copy()
       
    #update Mmn
    for ik in range(nkps**2):
        for ib in range(4):
            ikb = nnkp[ik,ib]
            tmmn = mmn[ik,ib]
            dumnkc = np.conj(np.transpose(umn[ik]))
            if ikb < 0:
                dumnkb = umn[-1-ikb]
            else:
                dumnkb = umn[ikb]                
            mmn[ik,ib] = np.matmul(dumnkc,np.matmul(tmmn,dumnkb))

    irguide = 0
    rguide = rcenter.copy()          
    csheet, sheet, rguide = wannier_phase(rguide, irguide, nbands, nkps, mmn, bvec)
    irguide = 1
    print(rguide)
    
    # calculate initial spread
    rn, omegan = wannier_omega(nkps, nbands, Ngrid, mmn, bvec, csheet, sheet)
    print('Initial wannier centers:') 
    print(rn/(2.0*np.pi))
    print('Initial wannier spread:')
    print(omegan)
    omegaI = wannier_omegaI(nkps, nbands, mmn, wb)
    print('Gauge invariant spread:')
    print(omegaI)

    # main iteration loop

    for it in range(nstep):                    
        
        # calculate gradient of spread
        domega = wannier_domega(nkps, nbands, Ngrid, mmn, bvec, csheet, sheet)
        
        # calculate search direction
        g0norm, g0, domega0, ng = wannier_cg(domega,domega0,g0norm,it,ng,nkps,nbands,wb)
        
        # update U and mmn
#        domega0 = domega
        dW = domega0 * steplength/(4.0*4.0*wb)
        umn, mmn = wannier_update(dW,mmn,umn,nnkp,nkps,nbands,wb,pbc)
        
        # determine guiding center
        csheet, sheet, rguide = wannier_phase(rguide, irguide, nbands, nkps, mmn, bvec)
        
        # calculate wannier centers and spread
        rn, omegan = wannier_omega(nkps, nbands, Ngrid, mmn, bvec, csheet, sheet)
        
        # print wannier center and spread
        print('Step: %d, Total spread = %f' % (it,np.sum(omegan)))
        print(*omegan, sep=',  ')
        
    print('Final wannier centers:')
    print(rn/(2.0*np.pi), omegan)
    
#    mmntest = mmn0.copy()    
#    for ik in range(nkps**2):
#        for ib in range(4):
#            ikb = nnkp[ik,ib]
#            tmmn = mmn0[ik,ib]
#            dumnkc = np.conj(np.transpose(umn[ik]))
#            dumnkb = umn[ikb]
#            mmntest[ik,ib] = np.matmul(dumnkc,np.matmul(tmmn,dumnkb))
#    rn, omegan = wannier_omega(nkps, nbands, Ngrid, mmntest, bvec)
#    print('Final wannier centers--test:')
#    print(rn/(2.0*np.pi))    
        
#    nR, a1vec, a2vec = wannier_findR(rn,nbands)
#    print(nR)
    
#    rn1, umn = wannier_shiftR(rn, nbands, umn, mmn, nkps, kmesh, nnkp, bvec)
#    print('Shifted wannier centers:')
#    print(rn1/(2.0*np.pi))

    #update Mmn
#    for ik in range(nkps**2):
#        for ib in range(4):
#            ikb = nnkp[ik,ib]
#            tmmn = mmn0[ik,ib]
#            dumnkc = np.conj(np.transpose(umn[ik]))
#            if ikb < 0:
#                dumnkb = umn[-1-ikb]
#            else:
#                dumnkb = umn[ikb]                
#            mmn[ik,ib] = np.matmul(dumnkc,np.matmul(tmmn,dumnkb))
    # determine guiding center
#    irguide=0
#    csheet, sheet, rguide = wannier_phase(rcenter, irguide, nbands, nkps, mmn, bvec)    
#    rn, omegan = wannier_omega(nkps, nbands, Ngrid, mmn, bvec, csheet, sheet)
#    print('Shifted wannier centers--test:')
#    print(rn1/(2.0*np.pi), omegan)
    
    hmatr, Rmesh, Rgrid = wannier_hamiltonian(umn, eigval, nkps, nbands, kmesh)
#    print('Real space grid:')
#    print(Rmesh)
#    print('Real space Hamiltonian:')
#    print(hmatr)
    
    # save umn
    np.savetxt('umn_'+str(nkps)+'.dat', umn.reshape((nkps**2,nbands**2)).view(float), fmt='%16.10f')
    
    # save Hamiltonian
    f = open('hmat_r_'+str(nkps)+'.dat','w')
    f.close()
    f = open('hmat_r_'+str(nkps)+'.dat','a')
    for ir in range(Rgrid**2):
        print('R = ', file = f)
        tR = Rmesh[ir].reshape((1,3))
        np.savetxt(f, tR, fmt='%16.10f', delimiter=',  ')
        np.savetxt(f, hmatr[ir], fmt='%16.10f', delimiter=',  ')
    f.close()
    # save Rmesh
    np.savetxt('Rmesh_'+str(nkps)+'.dat', Rmesh, fmt='%16.10f')



def chern(nkps = 10, whichband = 0, Ngrid = 9, phi = 0.6, facz = 0.2, Vscalar = 0):
    '''Calculate the Chern number for given phi and facz '''
    # generate a mesh
    kmesh = np.zeros(((nkps+1)**2, 2))    # +1 is for the extra points at the boundary
    ncount = 0
    kstep = 1.0/nkps
    kshift = np.array([-0.5,-0.5]) # to make the mesh symmetric with respect to (0,0)
    for i in range(nkps+1):
        for j in range(nkps+1):
            kmesh[ncount] = np.array([i*kstep, j*kstep])+kshift
            ncount += 1
    
    # find the four points on each plaquette
    ksquare = np.zeros((nkps**2, 4),dtype=np.int)
    dksquare = np.array([[0,0],[1,0],[1,1],[0,1]])*kstep
    si = 0
    for i in range((nkps+1)**2):
        ki = kmesh[i]
        if np.abs(ki[0] - 1.0 - kshift[0]) > 0.5*kstep and np.abs(ki[1] - 1.0 - kshift[1]) > 0.5*kstep:
            ksquare[si,0] = i
            for j in range((nkps+1)**2):
                kj = kmesh[j]
                for inn in range(1,4):
                    if np.linalg.norm(ki + dksquare[inn] - kj) < 1e-6:
                        ksquare[si,inn] = j
            si += 1
            
    # calculate the eigenvectors on each mesh point
    uk = np.zeros(((nkps+1)**2, Ngrid**2),dtype=np.complex)
    for ik in range((nkps+1)**2):
        tk = kmesh[ik]
        teval, tevec = pwSchrsquare(Ngrid, phi, facz, 0, tk[0], tk[1])
        uk[ik] = tevec[whichband]
    
    # calculate the Chern number
    ch = 0.0
    for ik in range(nkps**2):
        tsq = ksquare[ik]
        tu = uk[tsq]
        u1 = np.inner(np.conj(tu[0]),tu[1])
        u2 = np.inner(np.conj(tu[1]),tu[2])
        u3 = np.inner(np.conj(tu[2]),tu[3])
        u4 = np.inner(np.conj(tu[3]),tu[0])
        ch += -np.angle(u1*u2*u3*u4)
    ch /= 2*np.pi
    return ch            
        
    
def chern_phasediag(nkps = 10, whichband = 0, Ngrid = 11, phirange = np.array([0,2]), faczrange = np.array([0,0.5]), Vscalar = 0, phasegrid = 50):
    ''' phase diagram of the chern number '''
    plt.rcParams.update({'font.size': 18})        
    x = np.linspace(phirange[0]+0.0001, phirange[1], phasegrid)
    y = np.linspace(faczrange[0]*4+0.0001, faczrange[1]*4, phasegrid)
    
    X, Y = np.meshgrid(x, y)
    Xr = np.ravel(X)
    Yr = np.ravel(Y)
    chlist = np.zeros((Xr.size))  # for storing the Chern number

    # generate a mesh
    kmesh = np.zeros(((nkps+1)**2, 2))    # +1 is for the extra points at the boundary
    ncount = 0
    kstep = 1.0/nkps
    kshift = np.array([-0.5,-0.5]) # to make the mesh symmetric with respect to (0,0)
    for i in range(nkps+1):
        for j in range(nkps+1):
            kmesh[ncount] = np.array([i*kstep, j*kstep])+kshift
            ncount += 1
    
    # find the four points on each plaquette
    ksquare = np.zeros((nkps**2, 4),dtype=np.int)
    dksquare = np.array([[0,0],[1,0],[1,1],[0,1]])*kstep
    si = 0
    for i in range((nkps+1)**2):
        ki = kmesh[i]
        if np.abs(ki[0] - 1.0 - kshift[0]) > 0.5*kstep and np.abs(ki[1] - 1.0 - kshift[1]) > 0.5*kstep:
            ksquare[si,0] = i
            for j in range((nkps+1)**2):
                kj = kmesh[j]
                for inn in range(1,4):
                    if np.linalg.norm(ki + dksquare[inn] - kj) < 1e-6:
                        ksquare[si,inn] = j
            si += 1    
    uk = np.zeros(((nkps+1)**2, Ngrid**2),dtype=np.complex)

    # calculate the chern number for each pair of parameter values phi and facz     
    starttime = time.time()
    for i in range(X.size): 
        # obtain the five matrices for constructing the Hamiltonian
        Hmat0, Hmatx, Hmaty, Hmatx2, Hmaty2 = pwSchrmat(Ngrid, Xr[i], Yr[i]*0.25, 0)
        # calculate the eigenvectors on each mesh point
        for ik in range((nkps+1)**2):
            tk = kmesh[ik]
            Hmat = Hmat0 + tk[0]*Hmatx + tk[1]*Hmaty + tk[0]**2*Hmatx2 + tk[1]**2*Hmaty2
            eigval, eigvec = np.linalg.eigh(Hmat) 
            uk[ik] = (eigvec.T)[whichband]        
        # calculate the Chern number
        ch = 0.0
        for ik in range(nkps**2):
            tsq = ksquare[ik]
            tu = uk[tsq]
            u1 = np.inner(np.conj(tu[0]),tu[1])
            u2 = np.inner(np.conj(tu[1]),tu[2])
            u3 = np.inner(np.conj(tu[2]),tu[3])
            u4 = np.inner(np.conj(tu[3]),tu[0])
            ch += -np.angle(u1*u2*u3*u4)
        ch /= 2*np.pi 
        chlist[i] = ch
        if i == 100 - 1:
            measuretime = time.time() - starttime
            print('Estimated total time: %f seconds' % (measuretime*(X.size)/100))
    finaltime = time.time()
    elapsetime = (finaltime - starttime)        
    print('Chern number list generated successfully. Total time: %f seconds.' % elapsetime)   
    # save
    np.savetxt('chern_'+str(Ngrid)+ '_' + str(phasegrid) +'.dat', chlist, fmt='%16.10f')
    # plot    
    chmesh = chlist.reshape(X.shape)      
    plt.figure(1000,figsize=(6,6))
    ax=plt.gca()
    ax.set_aspect('equal')
    nbound=np.round(np.max(np.abs(chlist)))
    print(np.max(np.abs(chlist)))
    plt.contourf(X, Y, chmesh, np.arange(np.round(np.min(chlist))-0.125, np.round(np.max(chlist))+1.125,0.25), cmap = 'bwr',vmin=-nbound,vmax=nbound)
    ax.set_facecolor('black')
    plt.colorbar(fraction=0.046, pad=0.04, ticks=np.arange(np.round(np.min(chlist)), np.round(np.max(chlist))+1, dtype=np.int))
    xcap = '$\phi$'
    ycap = '$gm/m_e$'
    plt.ylabel(ycap)
    plt.xlabel(xcap)                    
    outfile = 'chern_'+ str(Ngrid) + '_' + str(phasegrid) + '.pdf'
    plt.savefig(outfile,bbox_inches='tight') 
    plt.close(1000)    
    
    
        
        
def meff_phasediag(Ngrid = 11, phirange = np.array([0,2]), faczrange = np.array([0,0.5]), Vscalar = 0,phasegrid = 50):
    ''' Phase diagram of the inverse effective mass '''         
    plt.rcParams.update({'font.size': 18})        
    x = np.linspace(phirange[0], phirange[1], phasegrid)
    y = np.linspace(faczrange[0]*4, faczrange[1]*4, phasegrid)
    
    X, Y = np.meshgrid(x, y)
    Xr = np.ravel(X)
    Yr = np.ravel(Y)
    meff = np.zeros((Xr.size))
    
    dk = 0.00001   # dk used for finite difference approximation
    for i in range(X.size):      
        # diagonalize the Hamiltonian and get the inverse effective mass
        # use the 9-point stencil approximation of the Laplacian. Need eigenenergies at
        # (0,0), (dk,0) and (dk,dk). Let the eigenenergies be e0, e1, and e2, respectively.
        # The inverse effective mass is (16*e1+4*e2-20*e0)/(6*dk**2) = (8*e1+2*e2-10*e0)/(3*dk**2)
        teval, tevec = pwSchrsquare(Ngrid, Xr[i], Yr[i]*0.25, 0, 0, 0)
        e0 = np.sort(np.real(teval))[0]
        teval, tevec = pwSchrsquare(Ngrid, Xr[i], Yr[i]*0.25, 0, dk, 0) 
        e1 = np.sort(np.real(teval))[0]
        teval, tevec = pwSchrsquare(Ngrid, Xr[i], Yr[i]*0.25, 0, dk, dk)
        e2 = np.sort(np.real(teval))[0]
        meff[i] = (8*e1+2*e2-10*e0)/(3*dk**2)
    meffmesh = meff.reshape(X.shape)      
    plt.figure(1000,figsize=(6,6))
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.contourf(X, Y, meffmesh, 20, cmap = 'bwr',vmin=-np.max(meffmesh))
    ax.set_facecolor('black')
    plt.colorbar(fraction=0.046, pad=0.04)
    xcap = '$\phi$'
    ycap = '$gm/m_e$'
    plt.ylabel(ycap)
    plt.xlabel(xcap)                    
    outfile = 'meff_'+ str(Ngrid) + '_' + str(phasegrid) + '.pdf'
    plt.savefig(outfile,bbox_inches='tight') 
    plt.close(1000)
        
