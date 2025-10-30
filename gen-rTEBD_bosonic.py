
import numpy as np
import pandas as pd
from scipy.linalg import expm,svd
import math
import time
from os.path import exists


def vec_conj(a):
    return np.conjugate(a)

def vec_dot(a,b):
    # second vector is conjuagated
    return np.dot(a,vec_conj(b))

def dagger(M):
    return np.conjugate(np.transpose(M))

def mat_dot2(A,B):
    return np.dot(A,B)

def mat_dot4(A,B,C,D):
    return np.dot(A,np.dot(B,np.dot(C,D)))

def normalize(psi):
    return psi/np.sqrt(vec_dot(psi,psi))

def s_x():
    sx = np.matrix([[0,1.],[1.,0]])
    return sx

def s_y():
    sy = np.matrix([[0,-1j],[1j,0]])
    return sy

def s_z():
    sz = np.matrix([[1.,0],[0,-1.]])
    return sz

def spin_all():
    return [(1/2)*np.eye(2),(1/2)*s_x(),(1/2)*s_y(),(1/2)*s_z()]

def s_p():
    sp = np.matrix([[0,1.],[0,0]])
    return sp

def pauli_normal():
    return [np.eye(2),np.array(s_x()),np.array(s_y()),np.array(s_z())]

def pauli_tilde():
    return [np.eye(2),g*np.array(s_x()),g*np.array(s_y()),g*np.array(s_z())]

def pauli_bar():
    return [np.eye(2),(1/g)*np.array(s_x()),(1/g)*np.array(s_y()),(1/g)*np.array(s_z())]

def pauli_SzSz(L,ind):
    sz = np.matrix([[1,0],[0,-1]])
    tempz = 1
    for j in range(L):
        if j==ind:
            tempz = np.kron(tempz,sz)
        elif j == ind+1:
            tempz = np.kron(tempz,sz)
        else:
            tempz = np.kron(tempz,np.eye(2))
    return (1/4)*tempz

def transverse_ising_psite(L,J,hx,hz):
    H_psite = np.zeros((L-1,2**L,2**L),dtype=np.complex128)
    Sxx = []
    Syy = []
    Szz = []
    for i in range(L):
        Sxx.append((1/2)*pauli_matrices(L,i)[1])
        Syy.append((1/2)*pauli_matrices(L,i)[2])
        Szz.append((1/2)*pauli_matrices(L,i)[3])
    
    for i in range(0,L-1):
        if i == 0:
            H_psite[i] = (hx/2)*Sxx[i] +(hx/4)*Sxx[i+1] +J*np.dot(Szz[i],Szz[i+1]) +(hz/2)*Szz[i] +(hz/4)*Szz[i+1]
        elif i == L-2:
            H_psite[i] = (hx/4)*Sxx[i] +(hx/2)*Sxx[i+1] +J*np.dot(Szz[i],Szz[i+1]) +(hz/4)*Szz[i] +(hz/2)*Szz[i+1]
        else:
            H_psite[i] = (hx/4)*Sxx[i] +(hx/4)*Sxx[i+1] +J*np.dot(Szz[i],Szz[i+1]) +(hz/4)*Szz[i] +(hz/4)*Szz[i+1]
    return H_psite

def transverse_ising(J,hx,hz):
    # Returns local hamiltonian of all patches
    # No periodic bc
    sx = (1/2)*np.matrix([[0,1],[1,0]])
    sy = (1/2)*np.matrix([[0,-1j],[1j,0]])
    sz = (1/2)*np.matrix([[1,0],[0,-1]])
    
    hi = J*np.kron(sz,sz)+(hx/4)*(np.kron(np.eye(2),sx)+np.kron(sx,np.eye(2)))+(hz/4)*(np.kron(np.eye(2),sz)+np.kron(sz,np.eye(2)))
    if bound_diff == True:
        hi_start = J*np.kron(sz,sz)+(hx/4)*np.kron(np.eye(2),sx) + (hx/2)*np.kron(sx,np.eye(2))+(hz/4)*np.kron(np.eye(2),sz) + (hz/2)*np.kron(sz,np.eye(2))
        hi_end = J*np.kron(sz,sz)+(hx/2)*np.kron(np.eye(2),sx) + (hx/4)*np.kron(sx,np.eye(2))+(hz/2)*np.kron(np.eye(2),sz) + (hz/4)*np.kron(sz,np.eye(2))
    
    if bound_diff == False:
        return hi
    else:
        return hi,hi_start,hi_end

def U_mat(dt,U):
#     Hi = transverse_ising(J,hx,hz)
#     U = expm(-1j*dt*Hi)
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

def initial_psi_sch(L):
    gj = np.zeros(L)
    for i in range(L):
        if (i+1)%8 in [1,2,7,0]:
            gj[i] = -0.1
        elif (i+1)%8 in [3,4,5,6]:
            gj[i] = 0.1
    psi = 1
    for i in range(L):
        temp = np.dot((np.eye(2)+1j*(1+gj[i])*np.array(s_p())),np.array([0,1]))
        #temp = ((1-gj[i])*np.array([1,0]) + (1+gj[i])*np.array([0,1]))
        temp = normalize(temp)
        psi = np.kron(psi,temp)
    
    return psi

def initial_MPDO_dict(L):
    A_dict = {}
    gj = np.zeros(L)
    for i in range(L):
        if (i+1)%8 in [1,2,7,0]:
            gj[i] = -0.1
        elif (i+1)%8 in [3,4,5,6]:
            gj[i] = 0.1
    for i in range(L):
        A_temp = np.zeros(4,dtype=np.complex128)
        psi = ((1-gj[i])*np.array([1,0]) + (1+gj[i])*np.array([0,1]))
        #psi = np.dot((np.eye(2)+1j*(1+gj[i])*np.array(s_p())),np.array([0,1]))
        psi = normalize(psi)
        rho = np.outer(psi,np.conjugate(psi))
        #print(np.trace(rho))
        for j in range(4):
            A_temp[j] = np.trace(mat_dot2(pauli_bar()[j],rho))
        key = str("A"+str(i))
        A_dict[key] = np.reshape(A_temp,(1,4,1))
    return A_dict


class MPDO:
    # Hamiltonian parameters
    J = 1
    hx = 0.9045
    hz = 0.8090
    k = np.pi/4
    
    def __init__(self,L,chi,T,N):
        self.L = L
        self.chi = chi
        self.T = T
        self.N = N
        self.dt = self.T/self.N
        
        # Initialize U
        if bound_diff == False:
            Hi = transverse_ising(self.J,self.hx,self.hz)
            self.sU = expm(-1j*self.dt*Hi)
            self.U = U_mat(self.dt,self.sU)
        elif bound_diff == True:
            Hi,Hi_start,Hi_end = transverse_ising(self.J,self.hx,self.hz)
            self.sU = expm(-1j*self.dt*Hi)
            self.U = U_mat(self.dt,self.sU)
            self.sU_start = expm(-1j*self.dt*Hi_start)
            self.U_start = U_mat(self.dt,self.sU_start)
            self.sU_end = expm(-1j*self.dt*Hi_end)
            self.U_end = U_mat(self.dt,self.sU_end)
#         self.U = U_mat(self.L,self.J,self.hx,self.hz,self.dt)
#         self.sU = U_mat(self.L,self.J,self.hx,self.hz,self.dt)[0]
        
        # Initialize MPS
        self.A_dict = initial_MPDO_dict(self.L)
        self.lmbd_position = 0
        
        # Initialize Schrodinger wavefunction - all spins up
        if sch_bool == True:
            self.schrodinger_psi = initial_psi_sch(self.L)
            self.msr_schrodinger_mui = np.zeros((3,self.L),dtype=np.complex128)
            self.E_persite_sch = np.zeros(self.L-1,dtype=np.complex128)
            self.E_total_sch = 0
            self.renyi_test_left = 0
            self.renyi_test_right = 0
            self.E_fourier_sch = 0
            
        self.tr_TEBD = 0
        self.norm_trace = 0
        self.renyi_left = 0
        self.renyi_right = 0
        self.renyi_full = 0
        self.msr_mui = np.zeros((3,self.L),dtype=np.complex128)
        self.E_persite = np.zeros(self.L-1,dtype=np.complex128)
        self.E_total_TEBD = 0
        self.sq_trace = 0
        self.E_fourier = 0
        
    # Updates the Schrodinger wavefunction
    def applyU_schrodinger(self, ind, sU): # U is 4x4
        self.schrodinger_psi = np.reshape(self.schrodinger_psi, (2**ind,4,2**(self.L-2-ind)))
        self.schrodinger_psi = np.einsum('ij, ajb -> aib', sU, self.schrodinger_psi,optimize='optimal')
        self.schrodinger_psi = self.schrodinger_psi.flatten()
        
    def applyU(self,ind,dirc,U,sU,lm=False):
        
        # This part relocates lmbd to the right position
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
        I = np.reshape(np.eye(16),(4,4,4,4))
        self.applyU([ind,ind+1],'right',I,0,lm=True)
    
    # Function to move lmbd left
    def move_lmbd_left(self,ind):
        I = np.reshape(np.eye(16),(4,4,4,4))
        self.applyU([ind,ind+1],'left',I,0,lm=True)
        
    def sweepU(self):
    
        even_sites = [[i,i+1] for i in np.arange(0,self.L-1,2)]
        odd_sites = [[i,i+1] for i in np.arange(1,self.L-1,2)]
        odd_sites.reverse()
        
        for i in even_sites:
            if bound_diff == False:
                self.applyU(i,'right',self.U,self.sU)
            else:
                if i == [0,1]:
                    self.applyU(i,'right',self.U_start,self.sU_start)
                elif i == [self.L-2,self.L-1]:
                    self.applyU(i,'right',self.U_end,self.sU_end)
                else:
                    self.applyU(i,'right',self.U,self.sU)
            
        for i in odd_sites:
            self.applyU(i,'left',self.U,self.sU)
        
        if sch_bool == True:
            self.measure_schrodinger()
        self.measure_TEBD()
    
    # Relocates lmbd from lmbd_position to ind
    def lmbd_relocate(self,ind):
        step = ind - self.lmbd_position
        for i in range(np.abs(step)):
            if step > 0:
                self.move_lmbd_right(self.lmbd_position)
            elif step < 0:
                self.move_lmbd_left(self.lmbd_position-1)
                
    # Measures the expectation value using the Schrodinger wavefunction
    def measure_schrodinger(self):
        
        # Spin
        for m in range(1,4):
            for ind in range(self.L):
                #msr_sch = np.array((1/2)*pauli_matrices(self.L,ind)[m]) # fixx this
                msr_psi = np.reshape(self.schrodinger_psi,(2**ind,2,2**(self.L-1-ind)))
                #self.msr_schrodinger_mui[m-1][ind] = np.dot(np.conjugate(self.schrodinger_psi),np.dot(msr_sch,self.schrodinger_psi))
                self.msr_schrodinger_mui[m-1][ind] = np.einsum('aib,ij,ajb',np.conjugate(msr_psi),spin_all()[m],msr_psi)
        
        # Energy
#         self.E_total_sch = 0
#         for i in range(self.L-1):
#             msr_e = np.array(transverse_ising_psite(self.L,self.J,self.hx,self.hz)[i])
#             self.E_persite_sch[i] = np.dot(np.conjugate(self.schrodinger_psi),np.dot(msr_e,self.schrodinger_psi))
#             self.E_total_sch += self.E_persite_sch[i]
            
        # Different way of measuring the energy
        self.E_total_sch = 0
        Sz_ij = np.zeros(self.L-1,dtype=np.complex128)
        for ind in range(self.L-1):
            trunc_psi = np.reshape(self.schrodinger_psi,(2**ind,4,2**(self.L-2-ind)))
            Sz_ij[ind] = np.einsum('aib,ij,ajb',np.conjugate(trunc_psi),np.kron(spin_all()[3],spin_all()[3]),trunc_psi)
            #Sz_ij[ind] = np.dot(np.conjugate(self.schrodinger_psi),np.reshape(np.dot(pauli_SzSz(self.L,ind),self.schrodinger_psi),(2**L,1)))
            #print(Sz_ij)
        self.E_persite_sch[0] = (self.hx/2)*self.msr_schrodinger_mui[0][0] + (self.hx/4)*self.msr_schrodinger_mui[0][1] + self.J*Sz_ij[0]+(self.hz/2)*self.msr_schrodinger_mui[2][0] + (self.hz/4)*self.msr_schrodinger_mui[2][1]
        self.E_persite_sch[-1] = (self.hx/2)*self.msr_schrodinger_mui[0][-1] + (self.hx/4)*self.msr_schrodinger_mui[0][-2] + self.J*Sz_ij[-1] +(self.hz/2)*self.msr_schrodinger_mui[2][-1] + (self.hz/4)*self.msr_schrodinger_mui[2][-2]
        for i in range(1,self.L-2):
            self.E_persite_sch[i] = (self.hx/4)*self.msr_schrodinger_mui[0][i] + (self.hx/4)*self.msr_schrodinger_mui[0][i+1] + self.J*Sz_ij[i] +(self.hz/4)*self.msr_schrodinger_mui[2][i] + (self.hz/4)*self.msr_schrodinger_mui[2][i+1]
        for i in range(0,self.L-1):
            self.E_total_sch += self.E_persite_sch[i]
        
        self.E_fourier_sch = 0
        for i in range(self.L-1):
            self.E_fourier_sch += np.e**(1j*(i+1)*self.k)*self.E_persite_sch[i]
        self.E_fourier_sch = -(1/self.L)*self.E_fourier_sch
        
        
        # Total energy
#         H_tot = np.zeros((2**L,2**L),dtype=np.complex128)
#         for i in range(self.L-1):
#             H_tot += transverse_ising_psite(self.L,self.J,self.hx,self.hz)[i]
#         self.E_total_sch = np.dot(np.conjugate(self.schrodinger_psi),np.dot(H_tot,self.schrodinger_psi))
        
        # Renyi test
        psi_mat = np.reshape(self.schrodinger_psi,(int(2**(self.L/2)),int(2**(self.L/2))))
        rho_left = np.dot(dagger(psi_mat),psi_mat)
        rho_right = np.dot(psi_mat,dagger(psi_mat))
        tr_left_test = np.trace(np.dot(rho_left,rho_left))
        tr_right_test = np.trace(np.dot(rho_right,rho_right))
        total_trace = np.trace(rho_left)
        self.renyi_test_left = -np.log2(tr_left_test/(total_trace**2))
        self.renyi_test_right = -np.log2(tr_right_test/(total_trace**2))
                
    def measure_TEBD(self):
        
        self.left_trace = []
        self.right_trace = []
        self.build_left()
        self.build_right()
        
        for m in range(1,4):
            
            for ind in range(self.L):
                temp = self.left_trace[ind]
                temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,m,:],axes=1)
                temp = np.tensordot(temp,self.right_trace[ind],axes=1)
                self.msr_mui[m-1][ind] = (1/2)*temp.flatten()[0]
        
        # Measure trace
        temp = self.A_dict["A0"][:,0,:]
        for i in range(1,self.L):
            temp = np.tensordot(temp,self.A_dict["A"+str(i)][:,0,:],axes=1)
        temp = temp.flatten()[0]
        self.tr_TEBD = temp
        
#        if _sqtrace == True:
 #           temp1 = np.einsum('iaj,kal->ikjl',self.A_dict["A0"][:,:1,:],self.A_dict["A0"][:,:1,:],optimize='optimal')
  #          temp2 = np.einsum('iaj,kal->ikjl',(1/2)*g*self.A_dict["A0"][:,1:,:],(1/2)*g*self.A_dict["A0"][:,1:,:],optimize='optimal')
   #         temp = temp1 + temp2
    #        for i in range(1,self.L):
     #           temp = np.einsum('ikjl,jbm->ikmbl',temp,self.A_dict["A"+str(i)],optimize='optimal')
      #          temp1 = np.einsum('ikmbl,lbn->ikmn',temp[:,:,:,:1,:],self.A_dict["A"+str(i)][:,:1,:],optimize='optimal')
       #         temp2 = np.einsum('ikmbl,lbn->ikmn',g*temp[:,:,:,1:,:],g*self.A_dict["A"+str(i)][:,1:,:],optimize='optimal')
        #        temp = temp1 + temp2
         #   self.sq_trace = (1/2**self.L)*temp.flatten()[0]
          #  self.norm_trace = self.tr_TEBD/np.sqrt(self.sq_trace)
           # self.renyi_full = -np.log2(self.sq_trace/(self.tr_TEBD)**2)
        
#        if _renyi == True:
            # Renyi entropy left
 #           temp1 = np.einsum('iaj,kal->ikjl',self.A_dict["A0"][:,:1,:],self.A_dict["A0"][:,:1,:],optimize='optimal')
  #          temp2 = np.einsum('iaj,kal->ikjl',g*self.A_dict["A0"][:,1:,:],g*self.A_dict["A0"][:,1:,:],optimize='optimal')
   #         temp = temp1 + temp2
    #        for i in range(1,int(self.L/2)):
     #           temp = np.einsum('ikjl,jbm->ikmbl',temp,self.A_dict["A"+str(i)],optimize='optimal')
      #          temp1 = np.einsum('ikmbl,lbn->ikmn',temp[:,:,:,:1,:],self.A_dict["A"+str(i)][:,:1,:],optimize='optimal')
       #         temp2 = np.einsum('ikmbl,lbn->ikmn',g*temp[:,:,:,1:,:],g*self.A_dict["A"+str(i)][:,1:,:],optimize='optimal')
        #        temp = temp1 + temp2
         #   for i in range(int(self.L/2),self.L):
          #      temp1 = np.einsum('ikjl,jm->ikml',temp,self.A_dict["A"+str(i)][:,0,:],optimize='optimal')
           #     temp = np.einsum('ikml,ln->ikmn',temp1,self.A_dict["A"+str(i)][:,0,:],optimize='optimal')
           # left_trace_sq = (1/2**(self.L/2))*temp.flatten()[0]
           # self.renyi_left = -np.log2(left_trace_sq/(self.tr_TEBD**2))
        
            # Renyi entropy right
#            temp1 = np.einsum('iaj,kal->ikjl',self.A_dict["A"+str(self.L-1)][:,:1,:],self.A_dict["A"+str(self.L-1)][:,:1,:],optimize='optimal')
 #           temp2 = np.einsum('iaj,kal->ikjl',g*self.A_dict["A"+str(self.L-1)][:,1:,:],g*self.A_dict["A"+str(self.L-1)][:,1:,:],optimize='optimal')
  #          temp = temp1 + temp2
   #         loop1 = np.arange(self.L-2,(int(self.L/2))-1,-1)
    #        for i in loop1:
     #           temp = np.einsum('mbi,ikjl->mbkjl',self.A_dict["A"+str(i)],temp,optimize='optimal')
      #          temp1 = np.einsum('nbk,mbkjl->mnjl',self.A_dict["A"+str(i)][:,:1,:],temp[:,:1,:,:,:],optimize='optimal')
       #         temp2 = np.einsum('nbk,mbkjl->mnjl',g*self.A_dict["A"+str(i)][:,1:,:],g*temp[:,1:,:,:,:],optimize='optimal')
        #        temp = temp1 + temp2
         #   loop2 = np.arange((int(self.L/2))-1,-1,-1)
          #  for i in loop2:
           #     temp1 = np.einsum('ai,ikjl->akjl',self.A_dict["A"+str(i)][:,0,:],temp,optimize='optimal')
            #    temp = np.einsum('bk,akjl->abjl',self.A_dict["A"+str(i)][:,0,:],temp1,optimize='optimal')
           # right_trace_sq = (1/2**(self.L/2))*temp.flatten()[0]
           # self.renyi_right = -np.log2(right_trace_sq/(self.tr_TEBD**2))
                
        # Measure energy per site
        # Measure SzSz
        Sz_ij = np.zeros(self.L-1,dtype=np.complex128)
        for ind in range(self.L-1):
            Sz_ij[ind] = self.tensordot_SzSz(ind)
    
        self.E_total_TEBD = 0
        for i in range(0,self.L-1):
            if i == 0:
                self.E_persite[i] = (self.hx/2)*self.msr_mui[0][i] + (self.hx/4)*self.msr_mui[0][i+1] + self.J*Sz_ij[i] +(self.hz/2)*self.msr_mui[2][i] + (self.hz/4)*self.msr_mui[2][i+1]
            elif i == L-2:
                self.E_persite[i] = (self.hx/4)*self.msr_mui[0][i] + (self.hx/2)*self.msr_mui[0][i+1] + self.J*Sz_ij[i] +(self.hz/4)*self.msr_mui[2][i] + (self.hz/2)*self.msr_mui[2][i+1]
            else:
                self.E_persite[i] = (self.hx/4)*self.msr_mui[0][i] + (self.hx/4)*self.msr_mui[0][i+1] + self.J*Sz_ij[i] +(self.hz/4)*self.msr_mui[2][i] + (self.hz/4)*self.msr_mui[2][i+1]
            self.E_total_TEBD += self.E_persite[i]
            
        self.E_fourier = 0
        for i in range(self.L-1):
            self.E_fourier += np.e**(1j*(i+1)*self.k)*self.E_persite[i]
        self.E_fourier = -(1/self.L)*self.E_fourier
    
    def build_left(self):
        temp = np.reshape(1.+0.*1j,(1,1))
        self.left_trace.append(temp)
        for i in range(1,self.L):
            temp = np.tensordot(temp,self.A_dict["A"+str(i-1)][:,0,:],axes=1)
            self.left_trace.append(temp)
        
    def build_right(self):
        temp = np.reshape(1.+0.*1j,(1,1))
        self.right_trace.append(temp)
        loop_arr = np.arange(self.L-2,-1,-1)
        for i in loop_arr:
            temp = np.tensordot(self.A_dict["A"+str(i+1)][:,0,:],temp,axes=1)
            self.right_trace.append(temp)
        self.right_trace.reverse()
        
    def tensordot_A(self, ind):
        # ind is an array
        temp = self.A_dict["A0"][:,int(ind[0]),:]
        for i in range(1,self.L):
            temp = np.tensordot(temp,self.A_dict["A"+str(i)][:,int(ind[i]),:],axes=1)
            
        return temp.flatten()[0]
    
    def tensordot_SzSz(self, ind):
        # Sz at ind and ind+1
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,(1/2)*g*self.A_dict["A"+str(ind)][:,3,:],axes=1)
        temp = np.tensordot(temp,(1/2)*g*self.A_dict["A"+str(ind+1)][:,3,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    


g = #GG#
chi = #CHI#
bound_diff = True
sch_bool = False
_renyi = False
_sqtrace = False
L = #LL#
T = #TT#
N = #NN#

Sz_TEBD = np.zeros((L,N),dtype=np.complex128)
Sy_TEBD = np.zeros((L,N),dtype=np.complex128)
Sx_TEBD = np.zeros((L,N),dtype=np.complex128)
E_psite = np.zeros((L-1,N),dtype=np.complex128)
Sz_schrodinger = []
tr_TB = []
tr2_TB = []
Et_TEBD = []
Ef_TEBD = []
if _sqtrace == True:
    norm_trace_TEBD = []
    renyi_F = []
if _renyi == True:
    renyi_L = []
    renyi_R = []

if sch_bool == True:
    Sz_sch = np.zeros((L,N),dtype=np.complex128)
    Sy_sch = np.zeros((L,N),dtype=np.complex128)
    Sx_sch = np.zeros((L,N),dtype=np.complex128)
    E_psite_sch = np.zeros((L-1,N),dtype=np.complex128)
    Et_sch = []
    renyi_L_test = []
    renyi_R_test = []
    Ef_sch = []

mps_evolve = MPDO(L,chi,T,N)
if exists("py_print.txt"):
    f = open("py_print.txt","w")
    f.write('New run\n')
    f.close
else:
    f = open("py_print.txt","x")
    f.write('New run\n')
    f.close()
for i in range(N):

    t1 = time.time()
    mps_evolve.sweepU()
    t2 = time.time()
    f = open("py_print.txt","a")
    f.write('Time step = '+str(i)+', time taken = '+str(t2-t1)+' for each step\n')
    #print('Time step = '+str(i)+', time taken = '+str(t2-t1)+' for each step')
    f.close()
    for j in range(L):
        Sx_TEBD[j][i] = mps_evolve.msr_mui[0][j]
        Sy_TEBD[j][i] = mps_evolve.msr_mui[1][j]
        Sz_TEBD[j][i] = mps_evolve.msr_mui[2][j]
        if j != L-1:
            E_psite[j][i] = mps_evolve.E_persite[j]
    Et_TEBD.append(mps_evolve.E_total_TEBD)
    tr_TB.append(mps_evolve.tr_TEBD)
    Ef_TEBD.append(mps_evolve.E_fourier)
    if _sqtrace == True:
        norm_trace_TEBD.append(mps_evolve.norm_trace)
        tr2_TB.append(mps_evolve.sq_trace)
        renyi_F.append(mps_evolve.renyi_full)
    if _renyi == True:
        renyi_L.append(mps_evolve.renyi_left)
        renyi_R.append(mps_evolve.renyi_right)
    
    if sch_bool == True:
        for j in range(L):
            Sx_sch[j][i] = mps_evolve.msr_schrodinger_mui[0][j]
            Sy_sch[j][i] = mps_evolve.msr_schrodinger_mui[1][j]
            Sz_sch[j][i] = mps_evolve.msr_schrodinger_mui[2][j]
            if j != L-1:
                E_psite_sch[j][i] = mps_evolve.E_persite_sch[j]
        Et_sch.append(mps_evolve.E_total_sch)
        Ef_sch.append(mps_evolve.E_fourier_sch)
        renyi_L_test.append(mps_evolve.renyi_test_left)
        renyi_R_test.append(mps_evolve.renyi_test_right)
    
# Saving data
if sch_bool == True:
    final_array = np.zeros((N,8*L+10))
else:
    final_array = np.zeros((N,4*L+2))
col_names = []
for i in range(L):
    col_names.append('Sx'+str(i))
for i in range(L):
    col_names.append('Sy'+str(i))
for i in range(L):
    col_names.append('Sz'+str(i))
for i in range(L-1):
    col_names.append('E'+str(i))
col_names.append('E-total')
col_names.append('Trace')
#col_names.append('Trace^2')
#col_names.append('Norm_trace')
#col_names.append('RenyiF')
#col_names.append('RenyiL')
#col_names.append('RenyiR')
col_names.append('E_fourier')

if sch_bool == True:
    for i in range(L):
        col_names.append('Sx_sch'+str(i))
    for i in range(L):
        col_names.append('Sy_sch'+str(i))
    for i in range(L):
        col_names.append('Sz_sch'+str(i))
    for i in range(L-1):
        col_names.append('E_sch'+str(i))
    col_names.append('E_sch-total')
    col_names.append('Renyi_schL')
    col_names.append('Renyi_schR')
    col_names.append('E_fourier_sch')

df = pd.DataFrame(final_array,columns=np.array(col_names))

for i in range(L):
    temp_list_Sx = Sx_TEBD[i]
    df['Sx'+str(i)][:] = np.real(temp_list_Sx)

for i in range(L):
    temp_list_Sy = Sy_TEBD[i]
    df['Sy'+str(i)][:] = np.real(temp_list_Sy)

for i in range(L):
    temp_list_Sz = Sz_TEBD[i]
    df['Sz'+str(i)][:] = np.real(temp_list_Sz)
    
for i in range(L-1):
    temp_list_E = E_psite[i]
    df['E'+str(i)][:] = np.real(temp_list_E)
    
temp_list_Et = np.array(Et_TEBD)
df['E-total'][:] = np.real(temp_list_Et)

temp_list_trace = np.array(tr_TB)
df['Trace'][:] = np.real(temp_list_trace)

#temp_list_tr2 = np.array(tr2_TB)
#df['Trace^2'][:] = np.real(temp_list_tr2)

#temp_list_normt = np.array(norm_trace_TEBD)
#df['Norm_trace'][:] = np.real(temp_list_normt)

#temp_list_renyiF = np.array(renyi_F)
#df['RenyiF'][:] = np.real(temp_list_renyiF)

#temp_list_renyiL = np.array(renyi_L)
#df['RenyiL'][:] = np.real(temp_list_renyiL)

#temp_list_renyiF = np.array(renyi_R)
#df['RenyiR'][:] = np.real(temp_list_renyiF)

df['E_fourier'][:] = np.real(np.array(Ef_TEBD))

if sch_bool == True:
    for i in range(L):
        df['Sx_sch'+str(i)][:] = np.real(Sx_sch[i])
    for i in range(L):
        df['Sy_sch'+str(i)][:] = np.real(Sy_sch[i])
    for i in range(L):
        df['Sz_sch'+str(i)][:] = np.real(Sz_sch[i])
    for i in range(L-1):
        df['E_sch'+str(i)][:] = np.real(E_psite_sch[i])
    df['E_sch-total'][:] = np.real(np.array(Et_sch))
    df['Renyi_schL'][:] = np.real(np.array(renyi_L_test))
    df['Renyi_schR'][:] = np.real(np.array(renyi_R_test))
    df['E_fourier_sch'][:] = np.real(np.array(Ef_sch))
    
df.to_csv('MPDO_DMT_lowe_L'+str(L)+'_chi'+str(chi)+'_g'+str(g)+'_T'+str(T)+'_N'+str(N)+'.csv')


