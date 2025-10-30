
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
    
def non_int_fermi(J):
    hi = J*np.array([[0,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,0]])
    return hi

def init_fermi_psi_sch(L):
    t1 = np.array([1,0])
    t2 = np.array([0,1])
    psi = 1
    for i in range(int(L/2)):
        psi = np.kron(psi,t2)
        psi = np.kron(psi,t1)
    return psi

def init_fermi_psi(L):
    t1 = np.array([1,0])
    t2 = np.array([0,1])
    psi = np.zeros((L,2),dtype=np.complex128)
    for i in range(L):
        if (i+1)%8 in [1,2,7,0]:
            psi[i] += t2
        elif (i+1)%8 in [3,4,5,6]:
            psi[i] += t1
            
    return psi

def init_MPS_dict_fermi(L):
    psi = init_fermi_psi(L)
    A_dict = {}
    for i in range(L):
        key = str("A"+str(i))
        A_dict[key] = np.reshape(psi[i],(1,2,1))
    return A_dict

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
        temp = np.dot((np.eye(2)+1j*(1+gj[i])*np.array(s_p())),np.array([0,1]))
        #temp = ((1-gj[i])*np.array([1,0]) + (1+gj[i])*np.array([0,1]))
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
        temp = np.dot((np.eye(2)+1j*(1+gj[i])*np.array(s_p())),np.array([0,1]))
        #temp = ((1-gj[i])*np.array([1,0]) + (1+gj[i])*np.array([0,1]))
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

def n_op():
    return np.array([[0,0],[0,1]])

def ccd():
    return np.array([[1,0],[0,0]])

def c_op():
    return np.array([[0,1],[0,0]])

def cd_op():
    return np.array([[0,0],[1,0]])


class MPS:
    # Hamiltonian parameters
    J = 1
    k = np.pi/4
    
    def __init__(self,L,chi,T,N):
        self.L = L
        self.chi = chi
        self.T = T
        self.N = N
        self.dt = self.T/self.N
        
        # Initialize U
        
        Hi = non_int_fermi(self.J)
        self.U = expm(-1j*self.dt*Hi)
        
        
        # Initialize MPS
        self.A_dict = init_MPS_dict_fermi(self.L)
        self.lmbd_position = 0
        
        # Initialize Schrodinger wavefunction
        if sch_bool == True:
            self.schrodinger_psi = init_fermi_psi_sch(self.L)
            self.E_persite_sch = np.zeros(self.L-1,dtype=np.complex128)
            self.ni_persite_sch = np.zeros(self.L,dtype=np.complex128)
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
        self.E_persite = np.zeros(self.L-1,dtype=np.complex128)
        self.ni_persite = np.zeros(self.L,dtype=np.complex128)
        self.ni_connect = 0
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
        if lm == False:
            if dirc == 'left':
                self.lmbd_relocate(ind[1])
            elif dirc == 'right':
                self.lmbd_relocate(ind[0])
        
        # lm checks if we want to apply U or move lmbd
        if lm == False:
            self.lmbd_check(ind)
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
        if sch_bool == True:
            if not self.check_schrodinger_psi():
                print('Warning: Wavefunctions do not match')
    
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
        
        self.E_total_sch = 0
        c_ij = np.zeros(self.L-1,dtype=np.complex128)
        for ind in range(self.L-1):
            trunc_psi = np.reshape(self.schrodinger_psi,(2**ind,4,2**(self.L-2-ind)))
            c_ij[ind] = np.einsum('aib,ij,ajb',np.conjugate(trunc_psi),non_int_fermi(self.J),trunc_psi,optimize='optimal')
            self.E_persite_sch[ind] = c_ij[ind]
            self.E_total_sch += self.E_persite_sch[ind]
            
        for i in range(self.L):
            trunc_psi = np.reshape(self.schrodinger_psi,(2**i,2,2**(self.L-1-i)))
            self.ni_persite_sch[i] = np.einsum('aib,ij,ajb',np.conjugate(trunc_psi),n_op(),trunc_psi,optimize='optimal')
        
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
        
        self.E_total_TEBD = 0
        for ind in range(self.L-1):
            self.lmbd_relocate(ind)
            key1 = "A"+str(ind)
            key2 = "A"+str(ind+1)
            cij_tmp1 = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict[key1]),cd_op(),self.A_dict[key1],optimize='optimal')
            cij_tmp1 = np.einsum('bc,bid->dic',cij_tmp1,np.conjugate(self.A_dict[key2]),optimize='optimal')
            cij_tmp1 = np.einsum('dic,ij,cjd',cij_tmp1,c_op(),self.A_dict[key2],optimize='optimal')
            cij_tmp2 = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict[key1]),c_op(),self.A_dict[key1],optimize='optimal')
            cij_tmp2 = np.einsum('bc,bid->dic',cij_tmp2,np.conjugate(self.A_dict[key2]),optimize='optimal')
            cij_tmp2 = np.einsum('dic,ij,cjd',cij_tmp2,cd_op(),self.A_dict[key2],optimize='optimal')
            self.E_persite[ind] = cij_tmp1 + cij_tmp2
            self.E_total_TEBD += self.E_persite[ind]
            
        self.E_fourier = 0
        for i in range(self.L-1):
            self.E_fourier += np.e**(1j*(i+1)*self.k)*self.E_persite[i]
        self.E_fourier = -(1/self.L)*self.E_fourier
        
        for i in range(self.L):
            self.lmbd_relocate(i)
            self.ni_persite[i] = np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),n_op(),self.A_dict["A"+str(i)],optimize='optimal')
        
#         temp_co = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A0"]),n_op(),self.A_dict["A0"],optimize='optimal')
#         for i in range(1,self.L-1):
#             temp_co1 = np.einsum('bc,bid->dic',temp_co,np.conjugate(self.A_dict["A"+str(i)]),optimize='optimal')
#             temp_co = np.einsum('dic,ij,cje->de',temp_co1,s_z(),self.A_dict["A"+str(i)],optimize='optimal')
#         temp_co1 = np.einsum('bc,bid->dic',temp_co,np.conjugate(self.A_dict["A"+str(self.L-1)]),optimize='optimal')
#         temp_co = np.einsum('dic,ij,cjd',temp_co1,n_op(),self.A_dict["A"+str(self.L-1)],optimize='optimal')
#         self.n1nL = temp_co - self.ni_persite[0]*self.ni_persite[self.L-1]
        
#         temp_co3 = np.einsum('aib,aic->bc',np.conjugate(self.A_dict["A0"]),self.A_dict["A0"],optimize='optimal')
#         temp_co3_1 = np.einsum('bc,bid->dic',temp_co3,np.conjugate(self.A_dict["A1"]),optimize='optimal')
#         temp_co3 = np.einsum('dic,cie->de',temp_co3_1,self.A_dict["A1"],optimize='optimal')
#         temp_co3_1 = np.einsum('bc,bid->dic',temp_co3,np.conjugate(self.A_dict["A2"]),optimize='optimal')
#         temp_co3 = np.einsum('dic,ij,cje->de',temp_co3_1,n_op(),self.A_dict["A2"],optimize='optimal')
#         for i in range(3,self.L-1):
#             temp_co3_1 = np.einsum('bc,bid->dic',temp_co3,np.conjugate(self.A_dict["A"+str(i)]),optimize='optimal')
#             temp_co3 = np.einsum('dic,cie->de',temp_co3_1,self.A_dict["A"+str(i)],optimize='optimal')
#         temp_co3_1 = np.einsum('bc,bid->dic',temp_co3,np.conjugate(self.A_dict["A"+str(self.L-1)]),optimize='optimal')
#         temp_co3 = np.einsum('dic,ij,cjd',temp_co3_1,n_op(),self.A_dict["A"+str(self.L-1)],optimize='optimal')
#         self.ni_connect = temp_co3 #- self.ni_persite[2]*self.ni_persite[self.L-1]
        
        self.lmbd_relocate(int(self.L/2)-8)
        temp_co = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(int(self.L/2)-8)]),n_op(),self.A_dict["A"+str(int(self.L/2)-8)],optimize='optimal')
        for i in range(int(self.L/2)-7,int(self.L/2)+10):
            temp_co1 = np.einsum('bc,bid->dic',temp_co,np.conjugate(self.A_dict["A"+str(i)]),optimize='optimal')
            temp_co = np.einsum('dic,cie->de',temp_co1,self.A_dict["A"+str(i)],optimize='optimal')
        temp_co1 = np.einsum('bc,bid->dic',temp_co,np.conjugate(self.A_dict["A"+str(int(self.L/2)+10)]),optimize='optimal')
        temp_co = np.einsum('dic,ij,cjd',temp_co1,n_op(),self.A_dict["A"+str(int(self.L/2)+10)],optimize='optimal')
        self.ni_connect = temp_co - self.ni_persite[int(self.L/2)-8]*self.ni_persite[int(self.L/2)+10]
        
        # Measure Trace
        key_tr = "A"+str(self.lmbd_position)
        self.tr_TEBD = np.linalg.norm(np.einsum('aib,aib',self.A_dict[key_tr],np.conjugate(self.A_dict[key_tr])))
        
        # Sq trace
        self.sq_trace = np.einsum('aib,cjd,cjd,aib',np.conjugate(self.A_dict[key_tr]),self.A_dict[key_tr],np.conjugate(self.A_dict[key_tr]),self.A_dict[key_tr],optimize='optimal')
        self.sq_trace = np.linalg.norm(self.sq_trace)
        self.norm_trace = self.tr_TEBD/np.sqrt(self.sq_trace)

    
    def measure_correlations(self):
        
        self.ccdagger = np.zeros((self.L,self.L),dtype=np.complex128)
        self.cdaggerc = np.zeros((self.L,self.L),dtype=np.complex128)
        for i in range(self.L):
            self.lmbd_relocate(i)
            self.ccdagger[i][i] = np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),ccd(),self.A_dict["A"+str(i)],optimize='optimal')
            self.cdaggerc[i][i] = np.einsum('aib,ij,ajb',np.conjugate(self.A_dict["A"+str(i)]),n_op(),self.A_dict["A"+str(i)],optimize='optimal')
            
            b_temp = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(i)]),c_op(),self.A_dict["A"+str(i)],optimize='optimal')
            d_temp = np.einsum('aib,ij,ajc->bc',np.conjugate(self.A_dict["A"+str(i)]),cd_op(),self.A_dict["A"+str(i)],optimize='optimal')
            
            for j in range(i+1,self.L):
                #self.lmbd_relocate(j)
                b_temp1 = np.einsum('bc,bid->dic',b_temp,np.conjugate(self.A_dict["A"+str(j)]),optimize='optimal')
                self.ccdagger[i][j] = -1*np.einsum('dic,ij,cjd',b_temp1,cd_op(),self.A_dict["A"+str(j)],optimize='optimal')
                b_temp = np.einsum('dic,ij,cje->de',b_temp1,s_z(),self.A_dict["A"+str(j)],optimize='optimal')
                
                d_temp1 = np.einsum('bc,bid->dic',d_temp,np.conjugate(self.A_dict["A"+str(j)]),optimize='optimal')
                self.cdaggerc[i][j] = np.einsum('dic,ij,cjd',d_temp1,c_op(),self.A_dict["A"+str(j)],optimize='optimal')
                d_temp = np.einsum('dic,ij,cje->de',d_temp1,s_z(),self.A_dict["A"+str(j)],optimize='optimal')
                
            b_temp = np.einsum('aib,ij,cjb->ac',np.conjugate(self.A_dict["A"+str(i)]),c_op(),self.A_dict["A"+str(i)],optimize='optimal')
            d_temp = np.einsum('aib,ij,cjb->ac',np.conjugate(self.A_dict["A"+str(i)]),cd_op(),self.A_dict["A"+str(i)],optimize='optimal')
            
            for j in np.flip(np.arange(0,i,1)):
                
                self.ccdagger[i][j] = -1*self.cdaggerc[j][i]
                self.cdaggerc[i][j] = -1*self.ccdagger[j][i]
                
        self.ck_correl = 0
        for i in range(self.L):
            for j in range(self.L):
                #self.ck_correl += np.e**(self.k*(i-j))*self.ccdagger[i][j]
                self.ck_correl += np.e**(-1j*self.k*(i-j))*self.cdaggerc[i][j]
        self.ck_correl = (1/self.L**2)*self.ck_correl
        
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
        

chi = #CHI#
sch_bool = False
bound_diff = False
L = #LL#
T = #TT#
N = #NN#

E_psite = np.zeros((L-1,N),dtype=np.complex128)
ni_psite = np.zeros((L,N),dtype=np.complex128)
n3_nL = []
tr_TB = []
tr2_TB = []
Et_TEBD = []
Ef_TEBD = []
if sch_bool == True:
    E_psite_sch = np.zeros((L-1,N),dtype=np.complex128)
    ni_psite_sch = np.zeros((L,N),dtype=np.complex128)
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
        ni_psite[j][i] = mps_evolve.ni_persite[j]
        if j != L-1:
            E_psite[j][i] = mps_evolve.E_persite[j]
            
    
    #Sz_test.append(mps_evolve.sz_check)
    n3_nL.append(mps_evolve.ni_connect)
    Et_TEBD.append(mps_evolve.E_total_TEBD)
    Ef_TEBD.append(mps_evolve.E_fourier)
    tr_TB.append(mps_evolve.tr_TEBD)
    tr2_TB.append(mps_evolve.sq_trace)
    if sch_bool == True:
        for j in range(L):
            ni_psite_sch[j][i] = mps_evolve.ni_persite_sch[j]
            if j != L-1:
                E_psite_sch[j][i] = mps_evolve.E_persite_sch[j]
        Et_sch.append(mps_evolve.E_total_sch)
        Ef_sch.append(mps_evolve.E_fourier_sch)
        renyi_L_test.append(mps_evolve.renyi_test_left)
        renyi_R_test.append(mps_evolve.renyi_test_right)
        
# Saving data
if sch_bool == True:
    final_array = np.zeros((N,4*L+8))
else:
    final_array = np.zeros((N,2*L+3))
col_names = []
for i in range(L-1):
    col_names.append('E'+str(i))
for i in range(L):
    col_names.append('ni'+str(i))
col_names.append('E-total')
col_names.append('Trace')
col_names.append('E_fourier')
col_names.append('ninj_cor')

if sch_bool == True:
    for i in range(L-1):
        col_names.append('E_sch'+str(i))
    for i in range(L):
        col_names.append('ni_sch'+str(i))
    col_names.append('E_sch-total')
    col_names.append('Renyi_schL')
    col_names.append('Renyi_schR')
    col_names.append('E_fourier_sch')

    
df = pd.DataFrame(final_array,columns=np.array(col_names))

for i in range(L-1):
    temp_list_E = E_psite[i]
    df['E'+str(i)][:] = np.real(temp_list_E)
    
for i in range(L):
    temp_list_ni = ni_psite[i]
    df['ni'+str(i)][:] = np.real(temp_list_ni)
    
temp_list_Et = np.array(Et_TEBD)
df['E-total'][:] = np.real(temp_list_Et)

temp_list_trace = np.array(tr_TB)
df['Trace'][:] = np.real(temp_list_trace)

df['E_fourier'][:] = np.real(np.array(Ef_TEBD))

df['ninj_cor'][:] = np.real(np.array(n3_nL))

if sch_bool == True:
    for i in range(L-1):
        df['E_sch'+str(i)][:] = np.real(E_psite_sch[i])
    for i in range(L):
        df['ni_sch'+str(i)][:] = np.real(ni_psite_sch[i])
    df['E_sch-total'][:] = np.real(np.array(Et_sch))
    df['Renyi_schL'][:] = np.real(np.array(renyi_L_test))
    df['Renyi_schR'][:] = np.real(np.array(renyi_R_test))
    df['E_fourier_sch'][:] = np.real(np.array(Ef_sch))
    
df.to_csv('MPS_fermi_co_L'+str(L)+'_chi'+str(chi)+'_T'+str(T)+'_N'+str(N)+'.csv')

