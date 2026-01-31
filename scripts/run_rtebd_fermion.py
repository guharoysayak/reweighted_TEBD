"""

Outputs are written to:
  runs/<timestamp>_L{L}_chi{chi}_g{g}_T{T}_N{N}/
"""

from __future__ import annotations

import argparse
import sys
import json
import logging
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import expm, svd



def dagger(M):
    """
    Return the conjugate transpose (Hermitian adjoint) of a matrix.

    Args:
        M: 2D array-like.

    Returns:
        Conjugate-transposed matrix with the same shape swapped.
    """
    return np.conjugate(np.transpose(M))

def mat_dot2(A,B):
    """
    Matrix product of two matrices using NumPy dot.

    Args:
        A: Left matrix.
        B: Right matrix.

    Returns:
        A @ B.
    """
    return np.dot(A,B)

def mat_dot4(A,B,C,D):
    """
    Multiply four matrices in order: A @ B @ C @ D.

    This is a convenience wrapper used in trace expressions.

    Args:
        A: First matrix.
        B: Second matrix.
        C: Third matrix.
        D: Fourth matrix.

    Returns:
        Product A @ B @ C @ D.
    """
    return np.dot(A,np.dot(B,np.dot(C,D)))


def s_x():
    """
    Return the Pauli X matrix as a 2x2 NumPy matrix.
    """
    sx = np.matrix([[0,1.],[1.,0]])
    return sx

def s_y():
    """
    Return the Pauli Y matrix as a 2x2 NumPy matrix.
    """
    sy = np.matrix([[0,-1j],[1j,0]])
    return sy

def s_z():
    """
    Return the Pauli Z matrix as a 2x2 NumPy matrix.
    """
    sz = np.matrix([[1.,0],[0,-1.]])
    return sz




def pauli_tilde():
    """
    Return the reweighted Pauli basis {I, gX, gY, g^2 Z}.

    Uses the module-level scalar `g` set by `main()`.
    """
    return [np.eye(2),g*np.array(s_x()),g*np.array(s_y()),g*g*np.array(s_z())]

def pauli_bar():
    """
    Return the inverse-reweighted Pauli basis {I, X/g, Y/g, Z/g^2}.

    Uses the module-level scalar `g` set by `main()`.
    """
    return [np.eye(2),(1/g)*np.array(s_x()),(1/g)*np.array(s_y()),(1/(g*g))*np.array(s_z())]




def U_mat(dt,U):
    """
    Convert a two-site unitary U (4x4) into its 4x4x4x4 superoperator tensor.

    The returned tensor acts in the (reweighted) Pauli basis used by the MPDO
    update.

    Args:
        dt: Time step (kept for signature compatibility; not used).
        U: Two-site unitary matrix (4x4).

    Returns:
        Tensor U_all with shape (4,4,4,4).
    """

    Ud = dagger(U)
    U_all = np.zeros((4,4,4,4),dtype=np.complex128)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    sg1 = np.kron(pauli_bar()[i],pauli_bar()[j])
                    sg2 = np.kron(pauli_tilde()[k],pauli_tilde()[l])
                    U_all[i][j][k][l] = (1/4)*np.trace(mat_dot4(sg1,U,sg2,Ud))
    return U_all



def non_int_fermi(J):
    """
    Two-site Hamiltonian for the non-interacting fermion model.

    Args:
        J: Hopping strength.

    Returns:
        4x4 two-site Hamiltonian matrix in the computational basis.
    """
    hi = J*np.array([[0,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,0]])
    return hi

def init_fermi_psi_sch(L):
    """
    Initialize a product Schrödinger state on the full 2^L Hilbert space.

    Pattern is alternating occupation (|1,0,1,0,...|) as in the original script.

    Args:
        L: System size (must be even).

    Returns:
        State vector of length 2**L.
    """
    t1 = np.array([1,0])
    t2 = np.array([0,1])
    psi = 1
    for i in range(int(L/2)):
        psi = np.kron(psi,t2)
        psi = np.kron(psi,t1)
    return psi

def init_fermi_psi(L):
    """
    Initialize a site-factorized two-component spinor state used to build the MPDO.

    The pattern follows the original 8-site periodic initialization.

    Args:
        L: System size.

    Returns:
        Array of shape (L, 2) containing the local state at each site.
    """
    t1 = np.array([1,0])
    t2 = np.array([0,1])
    psi = np.zeros((L,2),dtype=np.complex128)

    for i in range(L):
        if (i+1)%8 in [1,2,7,0]:
            psi[i] += t2
        elif (i+1)%8 in [3,4,5,6]:
            psi[i] += t1
            
    return psi



def init_MPDO_dict_fermi(L):
    """
    Build the initial MPDO tensor dictionary for the fermion model.

    Each site tensor A_i has shape (1, 4, 1) in the Pauli transfer-matrix
    representation.

    Args:
        L: System size.

    Returns:
        Dictionary mapping 'A{i}' -> tensor with shape (1, 4, 1).
    """
    psi = init_fermi_psi(L)
    A_dict = {}
    for i in range(L):
        A_temp = np.zeros(4,dtype=np.complex128)
        rho = np.outer(psi[i],np.conjugate(psi[i]))
        for j in range(4):
            A_temp[j] = np.trace(mat_dot2(pauli_bar()[j],rho))
        key = str("A"+str(i))
        A_dict[key] = np.reshape(A_temp,(1,4,1))
    return A_dict

def n_op():
    """
    Number operator n = |1><1| in the local (|0>,|1>) basis.

    Returns:
        2x2 array for the local number operator.
    """
    return np.array([[0,0],[0,1]])

class MPDO:
    """
    MPDO representation and rTEBD evolution for the fermion model.

    This class stores MPDO tensors in a Pauli-transfer representation and evolves
    them via TEBD-style two-site updates (with reweighting controlled by `g`).
    Measurement routines compute energies, densities, traces, and selected
    correlators.
    """
    # Hamiltonian parameters
    J = 1
    k = np.pi/4
    
    def __init__(self,L,chi,T,N):
        """
        Initialize the MPDO and precompute the two-site evolution gate.

        Args:
            L: System size.
            chi: Maximum bond dimension.
            T: Total evolution time.
            N: Number of Trotter steps (dt = T/N).
        """
        self.L = L
        self.chi = chi
        self.T = T
        self.N = N
        self.dt = self.T/self.N
        
        Hi = non_int_fermi(self.J)
        self.sU = expm(-1j*self.dt*Hi)
        self.U = U_mat(self.dt,self.sU)
        
        # Initialize MPDO
        self.A_dict = init_MPDO_dict_fermi(self.L)
        self.lmbd_position = 0
        
        # Initialize Schrodinger wavefunction - all spins up
        if sch_bool == True:
            self.schrodinger_psi = init_fermi_psi_sch(self.L)
            self.E_persite_sch = np.zeros(self.L-1,dtype=np.complex128)
            self.ni_persite_sch = np.zeros(self.L,dtype=np.complex128)
            self.E_total_sch = 0
            self.renyi_test_left = 0
            self.renyi_test_right = 0
            self.E_fourier_sch = 0
            
        self.tr_TEBD = 0
        self.norm_trace = 0
        self.renyi_left = 0
        self.renyi_right = 0
        self.renyi_full = 0
        self.E_persite = np.zeros(self.L-1,dtype=np.complex128)
        self.ni_persite = np.zeros(self.L,dtype=np.complex128)
        self.ni_connect = 0
        self.E_total_TEBD = 0
        self.sq_trace = 0
        self.E_fourier = 0
        self.ck_correl = 0
        self.Gij = np.zeros(self.L,dtype=np.complex128)
        
    # Updates the Schrodinger wavefunction
    def applyU_schrodinger(self, ind, sU): # U is 4x4
        """
        Apply the two-site unitary to the full Schrödinger wavefunction (debug check).

        Args:
            ind: Left site index (applies on sites ind and ind+1).
            sU: Two-site unitary (4x4).
        """
        self.schrodinger_psi = np.reshape(self.schrodinger_psi, (2**ind,4,2**(self.L-2-ind)))
        self.schrodinger_psi = np.einsum('ij, ajb -> aib', sU, self.schrodinger_psi,optimize='optimal')
        self.schrodinger_psi = self.schrodinger_psi.flatten()
        
    def applyU(self,ind,dirc,U,sU,lm=False):
        """
        Apply the MPDO two-site update at a bond and perform truncation.

        Args:
            ind: Pair [i, i+1] specifying the bond.
            dirc: 'left' or 'right' canonicalization direction for placing lambda.
            U: Superoperator tensor with shape (4,4,4,4).
            sU: Physical two-site unitary (4x4), used only when Schrödinger check is on.
            lm: If True, perform a pure lambda move (identity update).
        """
        
        # This part relocates lmbd to the right position
        if lm == False:
            if dirc == 'left':
                self.lmbd_relocate(ind[1])
            elif dirc == 'right':
                self.lmbd_relocate(ind[0])
        
        # Need to evolve the schrodinger picture when lm = False
        if lm == False and sch_bool == True:
            self.applyU_schrodinger(ind[0], sU)
            
        A1 = self.A_dict["A"+str(ind[0])]
        A2 = self.A_dict["A"+str(ind[1])]
        chi1 = np.shape(A1)[0]
        chi2 = np.shape(A2)[2]
        
        s1 = np.einsum('ijkl,akb,blc->aijc',U,A1,A2,optimize='optimal')
        
        s2 = np.reshape(s1,(4*chi1,4*chi2))

        try:
            Lp,lmbd,R=np.linalg.svd(s2,full_matrices=False)
        except np.linalg.LinAlgError as err:
            if "SVD did not converge" in str(err):
                Lp,lmbd,R=svd(s2,full_matrices=False,lapack_driver='gesvd')
                f = open("py_print.txt","a")
                f.write("SVD convergence issue")
                f.close()
            else:
                raise
        chi12 = np.min([4*chi1,4*chi2])
        chi12_p = np.min([self.chi,chi12])
        

        lmbd = np.diag(lmbd)
    
        # Truncation step
        lmbd = lmbd[:chi12_p,:chi12_p]
        Lp = Lp[:,:chi12_p]
        R = R[:chi12_p,:]
    
        if (dirc == 'left'):
            A1 = np.reshape(np.dot(Lp,lmbd),(chi1,4,chi12_p))
            A2 = np.reshape(R,(chi12_p,4,chi2))
            self.lmbd_position = ind[0]
            
        elif (dirc == 'right'):
            A1 = np.reshape(Lp,(chi1,4,chi12_p))
            A2 = np.reshape(np.dot(lmbd,R),(chi12_p,4,chi2))
            self.lmbd_position = ind[1]
            
        self.A_dict["A"+str(ind[0])] = A1
        self.A_dict["A"+str(ind[1])] = A2
        
    # Function to move lmbd right
    def move_lmbd_right(self,ind):
        """
        Shift the bond singular values (lambda) one bond to the right.
        """
        I = np.reshape(np.eye(16),(4,4,4,4))
        self.applyU([ind,ind+1],'right',I,0,lm=True)
    
    # Function to move lmbd left
    def move_lmbd_left(self,ind):
        """
        Shift the bond singular values (lambda) one bond to the left.
        """
        I = np.reshape(np.eye(16),(4,4,4,4))
        self.applyU([ind,ind+1],'left',I,0,lm=True)
        
    def sweepU(self):
        """
        Perform one full TEBD sweep (even bonds then odd bonds) and measure observables.
        """
    
        even_sites = [[i,i+1] for i in np.arange(0,self.L-1,2)]
        odd_sites = [[i,i+1] for i in np.arange(1,self.L-1,2)]
        odd_sites.reverse()
        
        for i in even_sites:
            self.applyU(i,'right',self.U,self.sU)
            
        for i in odd_sites:
            self.applyU(i,'left',self.U,self.sU)
        
        if sch_bool == True:
            self.measure_schrodinger()
        self.measure_TEBD()
        self.measure_correlations()
    
    # Relocates lmbd from lmbd_position to ind
    def lmbd_relocate(self,ind):
        """
        Relocate the lambda position to a target bond index.

        Args:
            ind: Target bond index for lambda placement.
        """
        step = ind - self.lmbd_position
        for i in range(np.abs(step)):
            if step > 0:
                self.move_lmbd_right(self.lmbd_position)
            elif step < 0:
                self.move_lmbd_left(self.lmbd_position-1)
                
    # Measures the expectation value using the Schrodinger wavefunction
    def measure_schrodinger(self):
        """
        Compute reference observables using the full Schrödinger wavefunction (slow).
        """
        
        self.E_total_sch = 0
        c_ij = np.zeros(self.L-1,dtype=np.complex128)
        for ind in range(self.L-1):
            trunc_psi = np.reshape(self.schrodinger_psi,(2**ind,4,2**(self.L-2-ind)))
            c_ij[ind] = np.einsum('aib,ij,ajb',np.conjugate(trunc_psi),non_int_fermi(self.J),trunc_psi,optimize='optimal')
            self.E_persite_sch[ind] = c_ij[ind]
            self.E_total_sch += self.E_persite_sch[ind]
        
        self.E_fourier_sch = 0
        for i in range(self.L-1):
            self.E_fourier_sch += np.e**(1j*(i+1)*self.k)*self.E_persite_sch[i]
        self.E_fourier_sch = -(1/self.L)*self.E_fourier_sch
        
        for i in range(self.L):
            trunc_psi = np.reshape(self.schrodinger_psi,(2**i,2,2**(self.L-1-i)))
            self.ni_persite_sch[i] = np.einsum('aib,ij,ajb',np.conjugate(trunc_psi),n_op(),trunc_psi,optimize='optimal')
            
    
        psi_mat = np.reshape(self.schrodinger_psi,(int(2**(self.L/2)),int(2**(self.L/2))))
        rho_left = np.dot(dagger(psi_mat),psi_mat)
        rho_right = np.dot(psi_mat,dagger(psi_mat))
        tr_left_test = np.trace(np.dot(rho_left,rho_left))
        tr_right_test = np.trace(np.dot(rho_right,rho_right))
        total_trace = np.trace(rho_left)
        self.renyi_test_left = -np.log2(tr_left_test/(total_trace**2))
        self.renyi_test_right = -np.log2(tr_right_test/(total_trace**2))
                
    def measure_TEBD(self):
        """
        Measure trace, energies, densities, and optional Renyi quantities from the MPDO.
        """
        
        self.left_trace = []
        self.right_trace = []
        self.build_left()
        self.build_right()
        
        
        # Measure trace
        temp = self.A_dict["A0"][:,0,:]
        for i in range(1,self.L):
            temp = np.tensordot(temp,self.A_dict["A"+str(i)][:,0,:],axes=1)
        temp = temp.flatten()[0]
        self.tr_TEBD = temp
        
        if _sqtrace == True:
            temp1 = np.einsum('iaj,kal->ikjl',self.A_dict["A0"][:,:1,:],self.A_dict["A0"][:,:1,:],optimize='optimal')
            temp2 = np.einsum('iaj,kal->ikjl',g*self.A_dict["A0"][:,1:,:],g*self.A_dict["A0"][:,1:,:],optimize='optimal')
            temp = temp1 + temp2
            for i in range(1,self.L):
                temp = np.einsum('ikjl,jbm->ikmbl',temp,self.A_dict["A"+str(i)],optimize='optimal')
                temp1 = np.einsum('ikmbl,lbn->ikmn',temp[:,:,:,:1,:],self.A_dict["A"+str(i)][:,:1,:],optimize='optimal')
                temp2 = np.einsum('ikmbl,lbn->ikmn',g*temp[:,:,:,1:,:],g*self.A_dict["A"+str(i)][:,1:,:],optimize='optimal')
                temp = temp1 + temp2
            self.sq_trace = (1/2**self.L)*temp.flatten()[0]
            self.norm_trace = self.tr_TEBD/np.sqrt(self.sq_trace)
            self.renyi_full = -np.log2(self.sq_trace/(self.tr_TEBD)**2)
        
        if _renyi == True:
            # Renyi entropy left
            temp1 = np.einsum('iaj,kal->ikjl',self.A_dict["A0"][:,:1,:],self.A_dict["A0"][:,:1,:],optimize='optimal')
            temp2 = np.einsum('iaj,kal->ikjl',g*self.A_dict["A0"][:,1:,:],g*self.A_dict["A0"][:,1:,:],optimize='optimal')
            temp = temp1 + temp2
            for i in range(1,int(self.L/2)):
                temp = np.einsum('ikjl,jbm->ikmbl',temp,self.A_dict["A"+str(i)],optimize='optimal')
                temp1 = np.einsum('ikmbl,lbn->ikmn',temp[:,:,:,:1,:],self.A_dict["A"+str(i)][:,:1,:],optimize='optimal')
                temp2 = np.einsum('ikmbl,lbn->ikmn',g*temp[:,:,:,1:,:],g*self.A_dict["A"+str(i)][:,1:,:],optimize='optimal')
                temp = temp1 + temp2
            for i in range(int(self.L/2),self.L):
                temp1 = np.einsum('ikjl,jm->ikml',temp,self.A_dict["A"+str(i)][:,0,:],optimize='optimal')
                temp = np.einsum('ikml,ln->ikmn',temp1,self.A_dict["A"+str(i)][:,0,:],optimize='optimal')
            left_trace_sq = (1/2**(self.L/2))*temp.flatten()[0]
            self.renyi_left = -np.log2(left_trace_sq/(self.tr_TEBD**2))
        
            # Renyi entropy right
            temp1 = np.einsum('iaj,kal->ikjl',self.A_dict["A"+str(self.L-1)][:,:1,:],self.A_dict["A"+str(self.L-1)][:,:1,:],optimize='optimal')
            temp2 = np.einsum('iaj,kal->ikjl',g*self.A_dict["A"+str(self.L-1)][:,1:,:],g*self.A_dict["A"+str(self.L-1)][:,1:,:],optimize='optimal')
            temp = temp1 + temp2
            loop1 = np.arange(self.L-2,(int(self.L/2))-1,-1)
            for i in loop1:
                temp = np.einsum('mbi,ikjl->mbkjl',self.A_dict["A"+str(i)],temp,optimize='optimal')
                temp1 = np.einsum('nbk,mbkjl->mnjl',self.A_dict["A"+str(i)][:,:1,:],temp[:,:1,:,:,:],optimize='optimal')
                temp2 = np.einsum('nbk,mbkjl->mnjl',g*self.A_dict["A"+str(i)][:,1:,:],g*temp[:,1:,:,:,:],optimize='optimal')
                temp = temp1 + temp2
            loop2 = np.arange((int(self.L/2))-1,-1,-1)
            for i in loop2:
                temp1 = np.einsum('ai,ikjl->akjl',self.A_dict["A"+str(i)][:,0,:],temp,optimize='optimal')
                temp = np.einsum('bk,akjl->abjl',self.A_dict["A"+str(i)][:,0,:],temp1,optimize='optimal')
            right_trace_sq = (1/2**(self.L/2))*temp.flatten()[0]
            self.renyi_right = -np.log2(right_trace_sq/(self.tr_TEBD**2))
                
        # Measure energy per site
        # Measure SzSz
        SxSy = np.zeros(self.L-1,dtype=np.complex128)
        SySx = np.zeros(self.L-1,dtype=np.complex128)
        SxSx = np.zeros(self.L-1,dtype=np.complex128)
        SySy = np.zeros(self.L-1,dtype=np.complex128)
        for ind in range(self.L-1):
            SxSy[ind] = self.tensordot_SxSy(ind)
            SySx[ind] = self.tensordot_SySx(ind)
            SxSx[ind] = self.tensordot_SxSx(ind)
            SySy[ind] = self.tensordot_SySy(ind)
    
        self.E_total_TEBD = 0
        for i in range(0,self.L-1):
            cij_en = (1/4)*(SxSx[i]+1j*SxSy[i]-1j*SySx[i]+SySy[i])
            self.E_persite[i] = cij_en + np.conjugate(cij_en)
            self.E_total_TEBD += self.E_persite[i]
            
        self.E_fourier = 0
        for i in range(self.L-1):
            self.E_fourier += np.e**(1j*(i+1)*self.k)*self.E_persite[i]
        self.E_fourier = -(1/self.L)*self.E_fourier
        
        for i in range(self.L):
            self.ni_persite[i] = (1/2)*(self.tensordot_I(i)-self.tensordot_Sz(i))
            

            
        temp_co = np.tensordot(self.left_trace[0],(1/2)*(self.A_dict["A0"][:,0,:]-g*g*self.A_dict["A0"][:,3,:]),axes=1)
        for i in range(1,self.L-1):
            temp_co = np.tensordot(temp_co,self.A_dict["A"+str(i)][:,0,:],axes=1)
        temp_co = np.tensordot(temp_co,(1/2)*(self.A_dict["A"+str(self.L-1)][:,0,:]-g*g*self.A_dict["A"+str(self.L-1)][:,3,:]),axes=1)
        self.ni_connect = temp_co.flatten()[0]
        self.ni_connect = self.ni_connect - self.ni_persite[0]*self.ni_persite[self.L-1]

    
        
            
    def measure_correlations(self):
        """
        Compute two-point correlators <c_i^\\dagger c_j> and related Fourier component.
        """
        self.ccdagger = np.zeros((self.L,self.L),dtype=np.complex128)
        self.cdaggerc = np.zeros((self.L,self.L),dtype=np.complex128)
        
        for i in range(self.L):
            self.cdaggerc[i][i] = (1/2)*(self.tensordot_I(i)-self.tensordot_Sz(i))
            self.ccdagger[i][i] = (1/2)*(self.tensordot_I(i)-self.tensordot_Sz(i))
            
            b_temp = np.tensordot(self.left_trace[i],(1/2)*(g*self.A_dict["A"+str(i)][:,1,:]+1j*g*self.A_dict["A"+str(i)][:,2,:]),axes=1)
            d_temp = np.tensordot(self.left_trace[i],(1/2)*(g*self.A_dict["A"+str(i)][:,1,:]-1j*g*self.A_dict["A"+str(i)][:,2,:]),axes=1)
        
            for j in range(i+1,self.L):
                bt1 = np.tensordot(b_temp,(1/2)*(g*self.A_dict["A"+str(j)][:,1,:]-1j*g*self.A_dict["A"+str(j)][:,2,:]),axes=1)
                self.ccdagger[i][j] = -1*np.tensordot(bt1,self.right_trace[j],axes=1).flatten()[0]
                b_temp = np.tensordot(b_temp,g*g*self.A_dict["A"+str(j)][:,3,:],axes=1)
            
                dt1 = np.tensordot(d_temp,(1/2)*(g*self.A_dict["A"+str(j)][:,1,:]+1j*g*self.A_dict["A"+str(j)][:,2,:]),axes=1)
                self.cdaggerc[i][j] = np.tensordot(dt1,self.right_trace[j],axes=1).flatten()[0]
                d_temp = np.tensordot(d_temp,g*g*self.A_dict["A"+str(j)][:,3,:],axes=1)
            
            for j in np.flip(np.arange(0,i,1)):
                self.ccdagger[i][j] = -1*self.cdaggerc[j][i]
                self.cdaggerc[i][j] = -1*self.ccdagger[j][i]

        
                
        self.ck_correl = 0
        for i in range(self.L):
            for j in range(self.L):
                #self.ck_correl += np.e**(self.k*(i-j))*self.ccdagger[i][j]
                self.ck_correl += np.e**(-1j*self.k*(i-j))*self.cdaggerc[i][j]
#         self.ck_correl = (1/self.L**2)*self.ck_correl

    

    def measure_Gij(self):
        """
        Compute a one-row of G_{ij} = <c_i^\\dagger c_j> starting from i = L/4 (legacy behavior).
        """
        self.Gij = []
        d_temp = np.tensordot(self.left_trace[int(self.L/4)],(1/2)*(g*self.A_dict["A"+str(int(self.L/4))][:,1,:]-1j*g*self.A_dict["A"+str(int(self.L/4))][:,2,:]),axes=1)

        for j in range(int(self.L/4)+1,self.L):
            dt1 = np.tensordot(d_temp,(1/2)*(g*self.A_dict["A"+str(j)][:,1,:]+1j*g*self.A_dict["A"+str(j)][:,2,:]),axes=1)
            self.Gij.append(np.tensordot(dt1,self.right_trace[j],axes=1).flatten()[0])
            d_temp = np.tensordot(d_temp,g*g*self.A_dict["A"+str(j)][:,3,:],axes=1)
           
    
    
    

    def build_left(self):
        """
        Build left environment contractions needed for fast expectation values.
        """
        temp = np.reshape(1.+0.*1j,(1,1))
        self.left_trace.append(temp)
        for i in range(1,self.L):
            temp = np.tensordot(temp,self.A_dict["A"+str(i-1)][:,0,:],axes=1)
            self.left_trace.append(temp)
        
    def build_right(self):
        """
        Build right environment contractions needed for fast expectation values.
        """
        temp = np.reshape(1.+0.*1j,(1,1))
        self.right_trace.append(temp)
        loop_arr = np.arange(self.L-2,-1,-1)
        for i in loop_arr:
            temp = np.tensordot(self.A_dict["A"+str(i+1)][:,0,:],temp,axes=1)
            self.right_trace.append(temp)
        self.right_trace.reverse()
        
    
    
    def tensordot_SxSy(self, ind):
        """
        Local contraction used to measure the Sx-Sy contribution on a bond.
        """
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,1,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,2,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_SySx(self, ind):
        """
        Local contraction used to measure the Sy-Sx contribution on a bond.
        """
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,2,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,1,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_SxSx(self, ind):
        """
        Local contraction used to measure the Sx-Sx contribution on a bond.
        """
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,1,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,1,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_SySy(self, ind):
        """
        Local contraction used to measure the Sy-Sy contribution on a bond.
        """
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,2,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,2,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_Sz(self, ind):
        """
        Local contraction used to measure the Sz expectation value at a site.
        """
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*g*self.A_dict["A"+str(ind)][:,3,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind],axes=1)
        return temp.flatten()[0]
    
    def tensordot_I(self, ind):
        """
        Local contraction used to measure the identity expectation value at a site.
        """
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,self.A_dict["A"+str(ind)][:,0,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind],axes=1)
        return temp.flatten()[0]
    
    


@dataclass(frozen=True)
class RunParams:
    L: int = 64
    chi: int = 32
    T: float = 20.0
    N: int = 250
    g: float = 1.5
    schrodinger_check: bool = False
    renyi_cuts: bool = False
    sqtrace: bool = False
    outdir: str = "runs"
    tag: str = ""


def _git_commit_hash() -> str:
    """Best-effort git commit hash (empty if unavailable)."""
    try:
        import subprocess  # local import to keep base deps minimal

        h = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return h
    except Exception:
        return ""


def _make_run_dir(base: Path, params: RunParams) -> Path:
    """
    Create a unique run directory name based on timestamp and key parameters.

    Args:
        base: Base output directory.
        params: Run parameters.

    Returns:
        Newly created run directory path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{params.tag}" if params.tag else ""
    name = f"{ts}{tag}_L{params.L}_chi{params.chi}_g{params.g}_T{params.T}_N{params.N}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _setup_logging(run_dir: Path) -> None:
    """
    Configure logging to both console and a file in the run directory.

    Args:
        run_dir: Run directory path where 'run.log' will be written.
    """
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def save_run(run_dir: Path, params: RunParams, results: dict) -> None:
    """Save params + results in a reproducible way."""
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    payload = {"params": asdict(params), "meta": meta}

    (run_dir / "params.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    # numpy arrays -> compressed npz
    np.savez_compressed(run_dir / "results.npz", **results)

    # Also store a tiny human-readable summary (energies/trace) if available
    summary_lines = []
    if "Et_TEBD" in results:
        Et = np.asarray(results["Et_TEBD"])
        summary_lines.append(f"Et_TEBD: shape={Et.shape}, real[min,max]=({np.real(Et).min():.6g},{np.real(Et).max():.6g})")
    if "tr_TB" in results:
        tr = np.asarray(results["tr_TB"])
        summary_lines.append(f"trace: shape={tr.shape}, real[min,max]=({np.real(tr).min():.6g},{np.real(tr).max():.6g})")
    if summary_lines:
        (run_dir / "summary.txt").write_text("\\n".join(summary_lines) + "\\n")


def parse_args(argv: list[str] | None = None) -> RunParams:
    """
    Parse command-line arguments into a :class:`RunParams`.

    This function is safe to call from both a terminal and a Jupyter notebook.

    Parameters
    ----------
    argv:
        If provided, parse arguments from this list (e.g. ``['--L', '64']``).
        If ``None``, parse from ``sys.argv[1:]`` and ignore any unknown arguments
        (e.g. Jupyter's ``--f=...``).

    Returns
    -------
    RunParams
        Parsed run parameters.
    """
    p = argparse.ArgumentParser(description="Run rTEBD for the fermion model and save results.")
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--chi", type=int, default=32)
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--N", type=int, default=250)
    p.add_argument("--g", type=float, default=1.5)
    p.add_argument("--schrodinger-check", action="store_true", help="Also evolve a Schrödinger wavefunction check (slow).")
    p.add_argument("--renyi-cuts", action="store_true", help="Compute Renyi entropies for left/right cuts (slow).")
    p.add_argument("--sqtrace", action="store_true", help="Compute squared trace / full Renyi (slow).")
    p.add_argument("--outdir", type=str, default="runs", help="Base output directory.")
    p.add_argument("--tag", type=str, default="", help="Optional tag appended to the run folder name.")
    if argv is None:
        argv = sys.argv[1:]
    a, _unknown = p.parse_known_args(argv)
    return RunParams(
        L=a.L,
        chi=a.chi,
        T=a.T,
        N=a.N,
        g=a.g,
        schrodinger_check=a.schrodinger_check,
        renyi_cuts=a.renyi_cuts,
        sqtrace=a.sqtrace,
        outdir=a.outdir,
        tag=a.tag,
    )


def main(argv: list[str] | None = None) -> None:
    """
    Run rTEBD for the fermion model and save results.

    Parameters
    ----------
    argv:
        Optional list of CLI-style arguments. Use this when calling from a
        notebook, e.g. ``main(['--L','32','--chi','16'])``.
        If ``None``, arguments are taken from ``sys.argv``.
    """
    params = parse_args(argv)
    base = Path(params.outdir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(base, params)
    _setup_logging(run_dir)

    logging.info("Starting rTEBD fermion run")
    logging.info("Run directory: %s", run_dir)
    logging.info("Params: %s", params)

    # ---- Preserve original global-parameter behavior (minimal-risk) ----
    global g, chi, sch_bool, _renyi, _sqtrace, L, T, N
    g = float(params.g)
    chi = int(params.chi)
    sch_bool = bool(params.schrodinger_check)
    _renyi = bool(params.renyi_cuts)
    _sqtrace = bool(params.sqtrace)
    L = int(params.L)
    T = float(params.T)
    N = int(params.N)

    # Preallocate outputs (same as original script)
    E_psite = np.zeros((L - 1, N), dtype=np.complex128)
    ni_psite = np.zeros((L, N), dtype=np.complex128)

    ni_conn = []
    tr_TB = []
    tr2_TB = []
    Et_TEBD = []
    Ef_TEBD = []
    ck_TEBD = []

    if _sqtrace:
        norm_trace_TEBD = []
        renyi_F = []
    if _renyi:
        renyi_L = []
        renyi_R = []

    if sch_bool:
        E_psite_sch = np.zeros((L - 1, N), dtype=np.complex128)
        ni_psite_sch = np.zeros((L, N), dtype=np.complex128)
        Et_sch = []
        renyi_L_test = []
        renyi_R_test = []
        Ef_sch = []

    mps_evolve = MPDO(L, chi, T, N)

    t_start = time.time()
    for i in range(N):
        step_start = time.time()
        mps_evolve.sweepU()
        if i == N - 1:
            mps_evolve.measure_Gij()
        step_end = time.time()

        logging.info("Step %d/%d done in %.3fs", i + 1, N, step_end - step_start)

        for j in range(L):
            ni_psite[j, i] = mps_evolve.ni_persite[j]
            if j != L - 1:
                E_psite[j, i] = mps_evolve.E_persite[j]

        Et_TEBD.append(mps_evolve.E_total_TEBD)
        ni_conn.append(mps_evolve.ni_connect)
        tr_TB.append(mps_evolve.tr_TEBD)
        Ef_TEBD.append(mps_evolve.E_fourier)
        ck_TEBD.append(mps_evolve.ck_correl)

        if _sqtrace:
            norm_trace_TEBD.append(mps_evolve.norm_trace)
            tr2_TB.append(mps_evolve.sq_trace)
            renyi_F.append(mps_evolve.renyi_full)

        if _renyi:
            renyi_L.append(mps_evolve.renyi_left)
            renyi_R.append(mps_evolve.renyi_right)

        if sch_bool:
            for j in range(L):
                ni_psite_sch[j, i] = mps_evolve.ni_persite_sch[j]
                if j != L - 1:
                    E_psite_sch[j, i] = mps_evolve.E_persite_sch[j]
            Et_sch.append(mps_evolve.E_total_sch)
            Ef_sch.append(mps_evolve.E_fourier_sch)
            renyi_L_test.append(mps_evolve.renyi_test_left)
            renyi_R_test.append(mps_evolve.renyi_test_right)

    elapsed = time.time() - t_start
    logging.info("Finished run in %.2fs (%.2fs/step average)", elapsed, elapsed / max(N, 1))

    results = {
        "E_psite": E_psite,
        "ni_psite": ni_psite,
        "Et_TEBD": np.asarray(Et_TEBD, dtype=np.complex128),
        "ni_conn": np.asarray(ni_conn, dtype=np.complex128),
        "tr_TB": np.asarray(tr_TB, dtype=np.complex128),
        "Ef_TEBD": np.asarray(Ef_TEBD, dtype=np.complex128),
        "ck_TEBD": np.asarray(ck_TEBD, dtype=np.complex128),
    }
    if hasattr(mps_evolve, "Gij"):
        results["Gij_final"] = np.asarray(mps_evolve.Gij, dtype=np.complex128)

    if _sqtrace:
        results["norm_trace_TEBD"] = np.asarray(norm_trace_TEBD, dtype=np.complex128)
        results["tr2_TB"] = np.asarray(tr2_TB, dtype=np.complex128)
        results["renyi_F"] = np.asarray(renyi_F, dtype=np.complex128)

    if _renyi:
        results["renyi_L"] = np.asarray(renyi_L, dtype=np.complex128)
        results["renyi_R"] = np.asarray(renyi_R, dtype=np.complex128)

    if sch_bool:
        results.update(
            {
                "E_psite_sch": E_psite_sch,
                "ni_psite_sch": ni_psite_sch,
                "Et_sch": np.asarray(Et_sch, dtype=np.complex128),
                "Ef_sch": np.asarray(Ef_sch, dtype=np.complex128),
                "renyi_L_test": np.asarray(renyi_L_test, dtype=np.complex128),
                "renyi_R_test": np.asarray(renyi_R_test, dtype=np.complex128),
            }
        )

    save_run(run_dir, params, results)
    logging.info("Saved results to %s", run_dir)


if __name__ == "__main__":
    main()
