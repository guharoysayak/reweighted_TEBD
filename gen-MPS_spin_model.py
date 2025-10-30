
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

def normalize(psi):
    return psi/np.sqrt(vec_dot(psi,psi))

def s_x():
    sx = np.matrix([[0,1],[1,0]])
    return sx

def s_y():
    sy = np.matrix([[0,-1j],[1j,0]])
    return sy

def s_z():
    sz = np.matrix([[1,0],[0,-1]])
    return sz

def spin_all():
    return [(1/2)*np.eye(2),(1/2)*s_x(),(1/2)*s_y(),(1/2)*s_z()]

def s_p():
    sp = np.matrix([[0,1.],[0,0]])
    return sp

def pauli_matrices(L,ind):
    # Simple Pauli matrices
    sx = np.matrix([[0,1],[1,0]])
    sy = np.matrix([[0,-1j],[1j,0]])
    sz = np.matrix([[1,0],[0,-1]])
    
    tempx = 1
    tempy = 1
    tempz = 1
    for j in range(L):
            
        if (j==ind):
            tempx = np.kron(tempx,sx)
            tempy = np.kron(tempy,sy)
            tempz = np.kron(tempz,sz)
            #tempx = np.kron(tx,np.eye(2))
        else:
            tempx = np.kron(tempx,np.eye(2))
            tempy = np.kron(tempy,np.eye(2))
            tempz = np.kron(tempz,np.eye(2))
        
    return [tempx,tempy,tempz]

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

'''def U_mat(L,J,h,dt):
    Hi = transverse_ising(J,h)
    U_all = expm(-1j*dt*Hi)
    return U_all'''

def initial_psi_sch(L):
    gj = np.zeros(L)
    for i in range(L):
        if (i+1)%8 in [1,2,7,0]:
            gj[i] = -0.1
        elif (i+1)%8 in [3,4,5,6]:
            gj[i] = 0.1
    psi = 1
    for i in range(L):
#         temp = np.dot((np.eye(2)+1j*(1+gj[i])*np.array(s_p())),np.array([0,1]))
        temp = ((1-gj[i])*np.array([1,0]) + (1+gj[i])*np.array([0,1]))
        temp = normalize(temp)
        psi = np.kron(psi,temp)
    
    return psi

def initial_psi(L):
    psi = np.zeros((L,2),dtype=np.complex128)
    gj = np.zeros(L)
    for i in range(L):
        if (i+1)%8 in [1,2,7,0]:
            gj[i] = -0.1
        elif (i+1)%8 in [3,4,5,6]:
            gj[i] = 0.1
    for i in range(L):
        #temp = np.dot((np.eye(2)+1j*(1+gj[i])*np.array(s_p())),np.array([0,1]))
        temp = ((1-gj[i])*np.array([1,0]) + (1+gj[i])*np.array([0,1]))
        temp = normalize(temp)
        psi[i] += temp
    return psi

def initial_MPS_dict(L):
    psi = initial_psi(L)
    A_dict = {}
    for i in range(L):
        key = str("A"+str(i))
        A_dict[key] = np.reshape(psi[i],(1,2,1))
    return A_dict


class MPS:
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
            self.U = expm(-1j*self.dt*Hi)
        else:
            Hi,Hi_start,Hi_end = transverse_ising(self.J,self.hx,self.hz)
            self.U = expm(-1j*self.dt*Hi)
            self.U_start = expm(-1j*self.dt*Hi_start)
            self.U_end = expm(-1j*self.dt*Hi_end)
        
        # Initialize MPS
        self.A_dict = initial_MPS_dict(self.L)
        self.lmbd_position = 0
        
        # Initialize Schrodinger wavefunction
        if sch_bool == True:
            self.schrodinger_psi = initial_psi_sch(self.L)
            self.msr_schrodinger_mui = np.zeros((3,self.L),dtype=np.complex128)
            self.E_persite_sch = np.zeros(self.L-1,dtype=np.complex128)
            self.E_total_sch = 0
            self.renyi_test_left = 0
            self.renyi_test_right = 0
            self.E_fourier_sch = 0
        
        # Initialize TEBD wavefunction
        #self.TEBD_psi = np.zeros(2**L,dtype=np.complex128)
        self.tr_TEBD = 0
        self.sq_trace = 0
        self.norm_trace = 0
        self.renyi_left = 0
        self.renyi_right = 0
        self.renyi_full = 0
        self.msr_mui = np.zeros((3,self.L),dtype=np.complex128)
        self.E_persite = np.zeros(self.L-1,dtype=np.complex128)
        self.E_total_TEBD = 0
        self.sq_trace = 0
        self.E_fourier = 0
        
        
    # Returns the position of lmbd based on the trace
    def lmbd_pos(self):
        for i in range(self.L):
            key = "A"+str(i)
            lmbd_trace = np.linalg.norm(np.einsum('aib,aib',self.A_dict[key],np.conjugate(self.A_dict[key])))
            if (math.isclose(lmbd_trace,1)):
                break
        return i
    
    # Check lmbd_position using the previous function
    def lmbd_check(self,ind):
        if (ind[0] == self.lmbd_pos() or ind[1] == self.lmbd_pos()):
            return True
        else:
            print('Warning: Lmbd not in position')
            return False
    
    # Updates the Schrodinger wavefunction
    def applyU_schrodinger(self, ind,U): # U is 4x4
        self.schrodinger_psi = np.reshape(self.schrodinger_psi, (2**ind,4,2**(self.L-2-ind)))
        self.schrodinger_psi = np.einsum('ij, ajb -> aib', U, self.schrodinger_psi,optimize='optimal')
        self.schrodinger_psi = self.schrodinger_psi.flatten()
    
    # Updates the MPS and runs applyU_schrodinger
    def applyU(self,ind,dirc,U,lm=False):
        
        # This part relocates lmbd to the right position
        if dirc == 'left':
            self.lmbd_relocate(ind[1])
        elif dirc == 'right':
            self.lmbd_relocate(ind[0])
        
        # lm checks if we want to apply U or move lmbd
        if lm == False:
            #self.lmbd_check(ind)
            if sch_bool == True:
                self.applyU_schrodinger(ind[0],U)
            U = np.reshape(U,(2,2,2,2))
            
        elif lm == True:
            U = np.reshape(np.eye(4),(2,2,2,2))
            
        A1 = self.A_dict["A"+str(ind[0])]
        A2 = self.A_dict["A"+str(ind[1])]
        chi1 = np.shape(A1)[0]
        chi2 = np.shape(A2)[2]
    
        #s1 = np.einsum('aib,bjc,ijkl->aklc',A1,A2,U)
        s1 = np.einsum('ijkl,akb,blc->aijc',U,A1,A2,optimize='optimal')
        s2 = np.reshape(s1,(2*chi1,2*chi2))
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
        chi12 = np.min([2*chi1,2*chi2])
        chi12_p = np.min([self.chi,chi12])
        lmbd = np.diag(lmbd)
    
        # Truncation step
        lmbd = lmbd[:chi12_p,:chi12_p]
        Lp = Lp[:,:chi12_p]
        R = R[:chi12_p,:]
        
        norm_lmbd = np.einsum('ab,ab',lmbd,np.conjugate(lmbd))
        lmbd = lmbd/np.sqrt(norm_lmbd)
    
        if (dirc == 'left'):
            A1 = np.reshape(np.dot(Lp,lmbd),(chi1,2,chi12_p))
            A2 = np.reshape(R,(chi12_p,2,chi2))
            self.lmbd_position = ind[0]
            
        elif (dirc == 'right'):
            A1 = np.reshape(Lp,(chi1,2,chi12_p))
            A2 = np.reshape(np.dot(lmbd,R),(chi12_p,2,chi2))
            self.lmbd_position = ind[1]
        
        self.A_dict["A"+str(ind[0])] = A1
        self.A_dict["A"+str(ind[1])] = A2
        
        # Checks the TEBD and Schrodinger wavefunctions match
#         if sch_bool == True:
#             if not self.check_schrodinger_psi():
#                 print('Warning: Wavefunctions do not match')
    
    # Function to move lmbd right
    def move_lmbd_right(self,ind):
        self.applyU([ind,ind+1],'right',1,lm=True)
    
    # Function to move lmbd left
    def move_lmbd_left(self,ind):
        self.applyU([ind,ind+1],'left',1,lm=True)
    
    # Sweeps over the entire system and updates the MPS and the Schrodinger wavefunction (1 time step)
    def sweepU(self):
    
        even_sites = [[i,i+1] for i in np.arange(0,self.L-1,2)]
        odd_sites = [[i,i+1] for i in np.arange(1,self.L-1,2)]
        odd_sites.reverse()
        
        for i in even_sites:
            if bound_diff == False:
                self.applyU(i,'right',self.U)
            else:
                if i == [0,1]:
                    self.applyU(i,'right',self.U_start)
                elif i == [self.L-2,self.L-1]:
                    self.applyU(i,'right',self.U_end)
                else:
                    self.applyU(i,'right',self.U)
            
        for i in odd_sites:
            self.applyU(i,'left',self.U)
        if sch_bool == True:   
            self.measure_schrodinger()
        self.measure_TEBD()
        
    def boustrophedon_sweep(self):
        sites = [[i,i+1] for i in np.arange(0,self.L-1,1)]
        
        for i in sites:
            
            if bound_diff == False:
                self.applyU(i,'right',self.U)
            else:
                if i == [0,1]:
                    self.applyU(i,'right',self.U_start)
                elif i == [self.L-2,self.L-1]:
                    self.applyU(i,'right',self.U_end)
                else:
                    self.applyU(i,'right',self.U)
#         if sch_bool == True:
#             self.measure_schrodinger()
#         self.measure_TEBD()
        
        sites.reverse()
        for i in sites:
            
            if bound_diff == False:
                self.applyU(i,'left',self.U)
            else:
                if i == [0,1]:
                    self.applyU(i,'left',self.U_start)
                elif i == [self.L-2,self.L-1]:
                    self.applyU(i,'left',self.U_end)
                else:
                    self.applyU(i,'left',self.U)
                    
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
        for m in range(1,4):
            for ind in range(self.L):
                #msr_sch = np.array((1/2)*pauli_matrices(self.L,ind)[m]) # fixx this
                msr_psi = np.reshape(self.schrodinger_psi,(2**ind,2,2**(self.L-1-ind)))
                #self.msr_schrodinger_mui[m-1][ind] = np.dot(np.conjugate(self.schrodinger_psi),np.dot(msr_sch,self.schrodinger_psi))
                self.msr_schrodinger_mui[m-1][ind] = np.einsum('aib,ij,ajb',np.conjugate(msr_psi),spin_all()[m],msr_psi)
        
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
        
        psi_mat = np.reshape(self.schrodinger_psi,(int(2**(self.L/2)),int(2**(self.L/2))))
        rho_left = np.dot(dagger(psi_mat),psi_mat)
        rho_right = np.dot(psi_mat,dagger(psi_mat))
        tr_left_test = np.trace(np.dot(rho_left,rho_left))
        tr_right_test = np.trace(np.dot(rho_right,rho_right))
        total_trace = np.trace(rho_left)
        self.renyi_test_left = -np.log2(tr_left_test/(total_trace**2))
        self.renyi_test_right = -np.log2(tr_right_test/(total_trace**2))
        
    # Measures the expectation value using the MPS
    def measure_TEBD(self):
        
        self.left_trace = []
        self.right_trace = []
        self.build_left()
        self.build_right()
        
        for m in range(1,4):
            for ind in range(self.L):
                self.lmbd_relocate(ind)
                key = "A"+str(self.lmbd_position)
                self.msr_mui[m-1][ind] = np.einsum('aib,ij,ajb',np.conjugate(self.A_dict[key]),spin_all()[m],self.A_dict[key])
        #self.sz_check = self.tensordot_Sz(2)
#         self.MPS_to_wf()
        self.E_total_TEBD = 0
        Sz_ij = np.zeros(self.L-1,dtype=np.complex128)
        for ind in range(self.L-1):
#             trunc_psi = np.reshape(self.TEBD_psi,(2**ind,4,2**(self.L-2-ind)))
#             Sz_ij[ind] = np.einsum('aib,ij,ajb',np.conjugate(trunc_psi),np.kron(spin_all()[3],spin_all()[3]),trunc_psi)
            Sz_ij[ind] = self.tensordot_SzSz(ind)
    
        self.E_persite[0] = (self.hx/2)*self.msr_mui[0][0] + (self.hx/4)*self.msr_mui[0][1] + self.J*Sz_ij[0]+(self.hz/2)*self.msr_mui[2][0] + (self.hz/4)*self.msr_mui[2][1]
        self.E_persite[-1] = (self.hx/2)*self.msr_mui[0][-1] + (self.hx/4)*self.msr_mui[0][-2] + self.J*Sz_ij[-1] +(self.hz/2)*self.msr_mui[2][-1] + (self.hz/4)*self.msr_mui[2][-2]
        for i in range(1,self.L-2):
            self.E_persite[i] = (self.hx/4)*self.msr_mui[0][i] + (self.hx/4)*self.msr_mui[0][i+1] + self.J*Sz_ij[i] +(self.hz/4)*self.msr_mui[2][i] + (self.hz/4)*self.msr_mui[2][i+1]
        for i in range(0,self.L-1):
            self.E_total_TEBD += self.E_persite[i]
            
        self.E_fourier = 0
        for i in range(self.L-1):
            self.E_fourier += np.e**(1j*(i+1)*self.k)*self.E_persite[i]
        self.E_fourier = -(1/self.L)*self.E_fourier
            
        # Measure Trace
        key_tr = "A"+str(self.lmbd_position)
        self.tr_TEBD = np.linalg.norm(np.einsum('aib,aib',self.A_dict[key_tr],np.conjugate(self.A_dict[key_tr])))
        
        # Sq trace
#         self.sq_trace = np.einsum('aib,cjd,cjd,aib',np.conjugate(self.A_dict[key_tr]),self.A_dict[key_tr],np.conjugate(self.A_dict[key_tr]),self.A_dict[key_tr],optimize='optimal')
#         self.sq_trace = np.linalg.norm(self.sq_trace)
#         self.norm_trace = self.tr_TEBD/np.sqrt(self.sq_trace)
        
#         # Renyi left
#         temp = np.einsum('aib,cjd,ejf,gih->acegbdfh',np.conjugate(self.A_dict["A0"]),self.A_dict["A0"],np.conjugate(self.A_dict["A0"]),self.A_dict["A0"],optimize='optimal')
#         for i in range(1,int(self.L/2)):
#             temp = np.einsum('acegbdfh,bil->acegildfh',temp,np.conjugate(self.A_dict["A"+str(i)]),optimize='optimal')
#             temp = np.einsum('acegildfh,djm->acegiljmfh',temp,self.A_dict["A"+str(i)],optimize='optimal')
#             temp = np.einsum('acegiljmfh,fjn->acegilmnh',temp,np.conjugate(self.A_dict["A"+str(i)]),optimize='optimal')
#             temp = np.einsum('acegilmnh,hio->aceglmno',temp,self.A_dict["A"+str(i)],optimize='optimal')
#         for i in range(int(self.L/2),self.L):
#             temp = np.einsum('acegbdfh,bil->acegildfh',temp,np.conjugate(self.A_dict["A"+str(i)]),optimize='optimal')
#             temp = np.einsum('acegildfh,dim->aceglmfh',temp,self.A_dict["A"+str(i)],optimize='optimal')
#             temp = np.einsum('aceglmfh,fin->aceglminh',temp,np.conjugate(self.A_dict["A"+str(i)]),optimize='optimal')
#             temp = np.einsum('aceglminh,hio->aceglmno',temp,self.A_dict["A"+str(i)],optimize='optimal')
#         left_trace_sq = temp.flatten()[0]
#         self.renyi_left = -np.log2(left_trace_sq/(self.tr_TEBD**2))
        
        
    # Returns the wavefunction from the MPS
    def MPS_to_wf(self):
        temp = self.A_dict['A0']
        for i in range(self.L-1):
            temp = np.tensordot(temp,mps_evolve.A_dict['A'+str(i+1)],axes=1)
        self.TEBD_psi = temp.flatten()
    
    # Function to check the TEBD and Schrodinger wavefunction
    def check_schrodinger_psi(self):
        self.MPS_to_wf()
        if np.allclose(self.TEBD_psi,self.schrodinger_psi):
            return True
        else:
            return False
        
    def build_left(self):
        temp = np.reshape(1.+0.*1j,(1,1,1,1))
        self.left_trace.append(temp)
        for i in range(1,self.L):
            temp = np.einsum('mnij,iak->mnkaj',temp,np.conjugate(self.A_dict["A"+str(i-1)]),optimize='optimal')
            temp = np.einsum('mnkaj,jal->mnkl',temp,self.A_dict["A"+str(i-1)],optimize='optimal')
            self.left_trace.append(temp)
        
    def build_right(self):
        temp = np.reshape(1.+0.*1j,(1,1,1,1))
        self.right_trace.append(temp)
        loop_arr = np.arange(self.L-2,-1,-1)
        for i in loop_arr:
            temp = np.einsum('ijmn,kai->kajmn',temp,np.conjugate(self.A_dict["A"+str(i+1)]),optimize='optimal')
            temp = np.einsum('kajmn,laj->klmn',temp,self.A_dict["A"+str(i+1)],optimize='optimal')
            self.right_trace.append(temp)
        self.right_trace.reverse()
        
    def tensordot_SzSz2(self,ind):
        temp = self.left_trace[ind]
        temp = np.einsum('mnij,iak,ab->mnkbj',temp,np.conjugate(self.A_dict["A"+str(ind)]),spin_all()[3],optimize='optimal')
        temp = np.einsum('mnkbj,jbl->mnkl',temp,self.A_dict["A"+str(ind)],optimize='optimal')
        temp = np.einsum('mnij,iak,ab->mnkbj',temp,np.conjugate(self.A_dict["A"+str(ind+1)]),spin_all()[3],optimize='optimal')
        temp = np.einsum('mnkbj,jbl->mnkl',temp,self.A_dict["A"+str(ind+1)],optimize='optimal')
        temp = np.einsum('ijkl,klmn->ijmn',temp,self.right_trace[ind+1],optimize='optimal')
        return temp.flatten()[0]
    
    def tensordot_SzSz(self,ind):
        self.lmbd_relocate(ind)
        temp = np.einsum('iaj,ab,ibl->jl',np.conjugate(self.A_dict["A"+str(ind)]),spin_all()[3],self.A_dict["A"+str(ind)],optimize='optimal')
        #self.lmbd_relocate(ind+1)
        temp = np.einsum('ij,iak,ab->kbj',temp,np.conjugate(self.A_dict["A"+str(ind+1)]),spin_all()[3],optimize='optimal')
        temp = np.einsum('kbj,jbk',temp,self.A_dict["A"+str(ind+1)],optimize='optimal')
        return temp.flatten()[0]
        
    def tensordot_Sz(self,ind):
        temp = self.left_trace[ind]
        temp = np.einsum('mnij,iak,ab->mnkbj',temp,np.conjugate(self.A_dict["A"+str(ind)]),spin_all()[3],optimize='optimal')
        temp = np.einsum('mnkbj,jbl->mnkl',temp,self.A_dict["A"+str(ind)],optimize='optimal')
        temp = np.einsum('ijkl,klmn->ijmn',temp,self.right_trace[ind],optimize='optimal')
        return temp.flatten()[0]


chi = #CHI#
sch_bool = False
bound_diff = True
L = #LL#
T = #TT#
N = #NN#

Sz_TEBD = np.zeros((L,N),dtype=np.complex128)
Sy_TEBD = np.zeros((L,N),dtype=np.complex128)
Sx_TEBD = np.zeros((L,N),dtype=np.complex128)
E_psite = np.zeros((L-1,N),dtype=np.complex128)
Sz_test = []
tr_TB = []
tr2_TB = []
Et_TEBD = []
Ef_TEBD = []
renyi_L = []
renyi_R = []
norm_trace_TEBD = []
if sch_bool == True:
    Sz_sch = np.zeros((L,N),dtype=np.complex128)
    Sy_sch = np.zeros((L,N),dtype=np.complex128)
    Sx_sch = np.zeros((L,N),dtype=np.complex128)
    E_psite_sch = np.zeros((L-1,N),dtype=np.complex128)
    Et_sch = []
    renyi_L_test = []
    renyi_R_test = []
    Ef_sch = []

mps_evolve = MPS(L,chi,T,N)
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
    print('Time step = '+str(i)+', time taken = '+str(t2-t1)+' for each step')
    f.close()
    for j in range(L):
        Sx_TEBD[j][i] = mps_evolve.msr_mui[0][j]
        Sy_TEBD[j][i] = mps_evolve.msr_mui[1][j]
        Sz_TEBD[j][i] = mps_evolve.msr_mui[2][j]
        if j != L-1:
            E_psite[j][i] = mps_evolve.E_persite[j]
    #Sz_test.append(mps_evolve.sz_check)
    Et_TEBD.append(mps_evolve.E_total_TEBD)
    Ef_TEBD.append(mps_evolve.E_fourier)
    tr_TB.append(mps_evolve.tr_TEBD)
    tr2_TB.append(mps_evolve.sq_trace)
    norm_trace_TEBD.append(mps_evolve.norm_trace)
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
    final_array = np.zeros((N,8*L+8))
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
# col_names.append('Trace^2')
# col_names.append('Norm_trace')
# col_names.append('RenyiL')
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

# temp_list_tr2 = np.array(tr2_TB)
# df['Trace^2'][:] = np.real(temp_list_tr2)

# temp_list_normt = np.array(norm_trace_TEBD)
# df['Norm_trace'][:] = np.real(temp_list_normt)

# temp_list_renyiL = np.array(renyi_L)
# df['RenyiL'][:] = np.real(temp_list_renyiL)

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
    
df.to_csv('MPS_DMT_lowe_L'+str(L)+'_chi'+str(chi)+'_T'+str(T)+'_N'+str(N)+'.csv')



