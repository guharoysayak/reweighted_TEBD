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
    return [np.eye(2),g*np.array(s_x()),g*np.array(s_y()),g*g*np.array(s_z())]

def pauli_bar():
    return [np.eye(2),(1/g)*np.array(s_x()),(1/g)*np.array(s_y()),(1/g*g)*np.array(s_z())]

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

def init_MPDO_dict_fermi(L):
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
    return np.array([[0,0],[0,1]])

class MPDO:
    # Hamiltonian parameters
    J = 1
    k = np.pi/4
    
    def __init__(self,L,chi,T,N):
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
        
    # Updates the Schrodinger wavefunction
    def applyU_schrodinger(self, ind, sU): # U is 4x4
        self.schrodinger_psi = np.reshape(self.schrodinger_psi, (2**ind,4,2**(self.L-2-ind)))
        self.schrodinger_psi = np.einsum('ij, ajb -> aib', sU, self.schrodinger_psi,optimize='optimal')
        self.schrodinger_psi = self.schrodinger_psi.flatten()
        
    def applyU(self,ind,dirc,U,sU,lm=False):
        
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
            
#         self.cd0c2 = 0
#         tmp_trunc = np.reshape(self.schrodinger_psi,(2**0,2,2**2,2,2**(self.L-2-2)))
#         self.cd0c2 = np.einsum('aibjc,ik,jl,akblc',np.conjugate(tmp_trunc),cd_op(),c_op(),tmp_trunc)
#         print(self.cd0c2)

#         self.cd0c2 = np.dot(np.conjugate(self.schrodinger_psi),np.dot(sch_exact_cd0c2(),self.schrodinger_psi))
#         self.cd0c2 = 0
    
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
            
        self.ni_connect = (1/4)*(self.tr_TEBD - self.tensordot_n1nL())
            
    def measure_correlations(self):
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
                b_temp = np.tensordot(b_temp,g*self.A_dict["A"+str(j)][:,3,:],axes=1)
            
                dt1 = np.tensordot(d_temp,(1/2)*(g*self.A_dict["A"+str(j)][:,1,:]+1j*g*self.A_dict["A"+str(j)][:,2,:]),axes=1)
                self.cdaggerc[i][j] = np.tensordot(dt1,self.right_trace[j],axes=1).flatten()[0]
                d_temp = np.tensordot(d_temp,g*self.A_dict["A"+str(j)][:,3,:],axes=1)
            
            for j in np.flip(np.arange(0,i,1)):
                self.ccdagger[i][j] = -1*self.cdaggerc[j][i]
                self.cdaggerc[i][j] = -1*self.ccdagger[j][i]
                
        self.ck_correl = 0
        for i in range(self.L):
            for j in range(self.L):
                #self.ck_correl += np.e**(self.k*(i-j))*self.ccdagger[i][j]
                self.ck_correl += np.e**(-1j*self.k*(i-j))*self.cdaggerc[i][j]
        self.ck_correl = (1/self.L**2)*self.ck_correl
    
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
        temp = np.tensordot(temp,(1/2)*g*g*self.A_dict["A"+str(ind)][:,3,:],axes=1)
        temp = np.tensordot(temp,(1/2)*g*g*self.A_dict["A"+str(ind+1)][:,3,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_SxSy(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,1,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,2,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_SySx(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,2,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,1,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_SxSx(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,1,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,1,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_SySy(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind)][:,2,:],axes=1)
        temp = np.tensordot(temp,g*self.A_dict["A"+str(ind+1)][:,2,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind+1],axes=1)
        return temp.flatten()[0]
    
    def tensordot_Sz(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,g*g*self.A_dict["A"+str(ind)][:,3,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind],axes=1)
        return temp.flatten()[0]
    
    def tensordot_I(self, ind):
        temp = self.left_trace[ind]
        temp = np.tensordot(temp,self.A_dict["A"+str(ind)][:,0,:],axes=1)
        temp = np.tensordot(temp,self.right_trace[ind],axes=1)
        return temp.flatten()[0]
    
    def tensordot_n1nL(self):
        temp1 = self.left_trace[0]
        temp1 = np.tensordot(temp1,g*g*self.A_dict["A0"][:,3,:],axes=1)
        for i in range(1,self.L-1):
            temp1 = np.tensordot(temp1,self.A_dict["A"+str(i)][:,0,:],axes=1)
        temp1 = np.tensordot(temp1,g*g*self.A_dict["A"+str(self.L-1)][:,3,:],axes=1)
        
        return temp1.flatten()[0]
        
        #temp2 = np.tensordot(self.left_trace[0],self.right_trace[0],axes=1)
        
        #return (1/4)*(temp2.flatten()[0] - temp1.flatten()[0])


# Main code with parameters

g = 1.5
chi = 1000
bound_diff = False
sch_bool = False
_renyi = False
_sqtrace = False
L = 8
T = 50
N = 625

E_psite = np.zeros((L-1,N),dtype=np.complex128)
ni_psite = np.zeros((L,N),dtype=np.complex128)
ni_conn = []
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
    E_psite_sch = np.zeros((L-1,N),dtype=np.complex128)
    ni_psite_sch = np.zeros((L,N),dtype=np.complex128)
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
    print('Time step = '+str(i)+', time taken = '+str(t2-t1)+' for each step')
    f.close()
    for j in range(L):
        ni_psite[j][i] = mps_evolve.ni_persite[j]
        if j != L-1:
            E_psite[j][i] = mps_evolve.E_persite[j]
    Et_TEBD.append(mps_evolve.E_total_TEBD)
    tr_TB.append(mps_evolve.tr_TEBD)
    Ef_TEBD.append(mps_evolve.E_fourier)
    ni_conn.append(mps_evolve.ni_connect)
    if _sqtrace == True:
        norm_trace_TEBD.append(mps_evolve.norm_trace)
        tr2_TB.append(mps_evolve.sq_trace)
        renyi_F.append(mps_evolve.renyi_full)
    if _renyi == True:
        renyi_L.append(mps_evolve.renyi_left)
        renyi_R.append(mps_evolve.renyi_right)
    
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
    final_array = np.zeros((N,2*L+10))
else:
    final_array = np.zeros((N,2*L+2))
col_names = []
for i in range(L-1):
    col_names.append('E'+str(i))
for i in range(L):
    col_names.append('ni'+str(i))
col_names.append('E-total')
col_names.append('Trace')
col_names.append('E_fourier')

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

if sch_bool == True:
    for i in range(L-1):
        df['E_sch'+str(i)][:] = np.real(E_psite_sch[i])
    for i in range(L):
        df['ni_sch'+str(i)][:] = np.real(ni_psite_sch[i])
    df['E_sch-total'][:] = np.real(np.array(Et_sch))
    df['Renyi_schL'][:] = np.real(np.array(renyi_L_test))
    df['Renyi_schR'][:] = np.real(np.array(renyi_R_test))
    df['E_fourier_sch'][:] = np.real(np.array(Ef_sch))
    
#df.to_csv('MPDO_fermi_hydp_L'+str(L)+'_chi'+str(chi)+'_g'+str(g)+'_T'+str(T)+'_N'+str(N)+'.csv')