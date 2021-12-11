import numpy as np
import copy
import itertools
from scipy import linalg as sLA
from numpy import linalg as LA
from copy import deepcopy
from scipy.stats import unitary_group


from functions import *



class Unitary(object):
    
    def __init__(self, n):
        self.n = n
        self.U = None
        self.eigvec = None
        self.eigval = None
        self.phi = None
        
    def build_matrix(self, U = None, random = False):
        if random == True:
            self.U = unitary_group.rvs(2**self.n)
        elif random == False:
            self.U = np.array(U)
            
        s, v = LA.eig(self.U)
        self.eigval = s[0] 
        self.eigvec = v[:, 0]
        
        self.phi = np.real(np.log(self.eigval)/(2*np.pi*1j))
            
    def get_U(self, power = 0):
        if power == 0:
            return self.U
        return LA.matrix_power(self.U, power)
    
    
    def get_vec(self):
        return self.eigvec
    
    def get_val(self):
        return self.eigval
    
    def get_phi(self):
        return self.phi
    
    
def build_CU(U):
    CU = np.array([[0,0],
                   [0,1]])
    CU = np.kron(CU,U)
    for i in range(int(len(CU)/2)):
        CU[i][i] = 1
        
    return CU
