import numpy as np
import copy
import itertools
from scipy import linalg as sLA
from numpy import linalg as LA

from Gates import *
from functions import *

class Qpsi():
    def __init__(self, N = 2):
        self.N = N
        self.coefs = None
        self.rho_A = None
        self.rho_B = None
        self.tensor = None
        
    def build_random_state(self):
        self.coefs = np.random.rand(2**self.N) + 1j*np.random.rand(2**self.N)
        self.coefs = self.coefs/np.sqrt(np.sum(self.coefs*self.coefs.conj()))
        self.tensor = np.reshape(self.coefs, [2]*int(np.log2(len(self.coefs))))
        
    def build_zero_state(self):
        coefs = np.zeros(2**self.N) + 1j*np.zeros(2**self.N)
        coefs[0] = 1
        self.coefs = coefs
        self.tensor = np.reshape(self.coefs, [2]*int(np.log2(len(self.coefs))))
        
    def build_H_state(self):
        coefs = np.ones(2**self.N) + 1j*np.ones(2**self.N)
        self.coefs = self.coefs/np.sqrt(np.sum(self.coefs*self.coefs.conj()))
        self.tensor = np.reshape(self.coefs, [2]*int(np.log2(len(self.coefs))))
        
    def set_coefs(self, coefs):
        if len(coefs) == 2**self.N:
            self.coefs = np.array(coefs)
            
        self.tensor = np.reshape(self.coefs, [2]*int(np.log2(len(self.coefs))))
            
        
    # разбиение на два регистра кубитов:
    
    def split_system(self, na, nb):
        
#         d = int(2**len(A_axis))
#         k = int(2**len(B_axis))
        d = int(2**na)
        k = int(2**nb)
        
        coefs = np.reshape(self.coefs, [1,d,k,1])
        rho = np.tensordot(coefs, np.conjugate(coefs), [0,3])
        rho = np.reshape(rho, (d,k,d,k))
        self.rho_A = np.trace(rho, axis1=1, axis2=3)
        self.rho_B = np.trace(rho, axis1=0, axis2=2)
        
    def get_A_coefs(self):
        s, v = LA.eig(self.rho_A)
        A_vec = v[:, 0]
        return A_vec
    
    def get_B_coefs(self):
        s, v = LA.eig(self.rho_B)
        B_vec = v[:, 0]
        return B_vec
            
    def get_coefs(self):
        return self.coefs
    
    def apply_U(self, U, axis):
#         axis = self.N - 1 - np.array(axis)
        self.coefs = gate_md(U, self.coefs.copy(), axis)
#         self.coefs = apply_U(U, self.coefs.copy(), axis)
        self.tensor = np.reshape(self.coefs, [2]*int(np.log2(len(self.coefs))))
        
    def apply_long_Toffoli(self, axis):
        self.coefs = apply_long_Toffoli(self.coefs.copy(), axis = axis, N = len(axis)-1)
        self.tensor = np.reshape(self.coefs, [2]*int(np.log2(len(self.coefs))))
      
        
        
        