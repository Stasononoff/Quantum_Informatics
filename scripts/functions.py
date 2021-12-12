import numpy as np
import copy
import itertools
from scipy import linalg as sLA
from numpy import linalg as LA

from Gates import *
from Unitary import *


# возвращает все комбинации элементов листа без учёта перестановок
def get_combinations(val):
    comb_list = []
    for j in range(1,len(val)+1):
        com_set = itertools.combinations(val, j)
        for i in com_set: 
            comb_list.append(i)
    return comb_list


# применяет унитарное преобразование U над кубитами под индексами axis в состоянии с коэффициентами coefs.
# U - матрица унитарного преобразования
# coefs - вектор состояния (коээфициенты)
# axis - индексы выбранных кубитов

def apply_U(U, coefs, axis):
    axis = list(reversed(axis)) # Нужно для оценки фазы
    if ((len(U) != 2**len(axis)) | (len(coefs) < len(U)) | (2**max(axis) > len(coefs))):
        print('Некорректно задано преобразование')
        return 0
    a = coefs.copy()
    b = [0]*len(a)
    
    ax = axis[0]
    
    N = len(U)  # размерность преобразования 
    K = len(coefs) # размерность состояния системы
    
    axis_comb = get_combinations(axis)
    old_ind_set = {K+1}
    
    for i in range(K):
        
        zero_index = (K-1)
            
        for ax in axis:
            zero_index = zero_index^(1<<ax)
            
        zero_index = i&zero_index
        
        if zero_index in old_ind_set:
            continue
            
        old_ind_set.add(zero_index)

                
        index = zero_index
        
        b[index] += a[zero_index]*U[0][0]
        
        
        u1 = 0
        for ax_list in axis_comb:
            u1 += 1
            m = 0
            for ax in ax_list:
                m += 1<<ax

            b[index] += a[zero_index^m]*U[0][u1]
            

        u0 = 0
        for ax_list in axis_comb:
            u0 += 1
            r = 0
            for ax in ax_list:
                r += 1<<ax
            
            index = zero_index^r
            old_ind_set.add(index)
            
            b[index] += a[zero_index]*U[u0][0]
            
            
            
            u1 = 0
            for ax1_list in axis_comb:
                u1 += 1
                m = 0
                for ax1 in ax1_list:
                    m += 1<<ax1
                b[index] += a[zero_index^m]*U[u0][u1]
            
        
    return b




# m-кубитное преобразование
def gate_md(Um, state, nums): 
    state = np.array(state)
    dim = state.shape[0]
    nums = np.flip(nums)
    n = int(np.log2(dim))
    m = nums.shape[0]
    b = np.zeros(dim, dtype=complex)
    masks = np.zeros(m, dtype=int)
    masks_m = np.zeros(m, dtype=int)
    for i in range(m):
        masks_m[i] = (1 << m-1-i)
    for i in range(m):
        masks[i] = (1 << n-1-nums[i])
    mask = masks.sum()
    
    for ind in range(dim):
        first = 0
        for i in range(m):
            first += (int((ind & masks[i]) != 0)<<i)
        for x in range(2**m):
            second = 0
            state_ind = ind - (ind&mask)
            for i in range(m):
                second += (int((x & masks_m[i]) != 0) << i)
                state_ind += (int((x & masks_m[i]) != 0) << n-1-nums[i])
            b[ind] += state[state_ind]*Um[first, second]
    return b




def apply_Toffoli(coefs, axis):
    coefs = gate_md(H(), coefs, [axis[2]])
    coefs = gate_md(CX(), coefs, [axis[1],axis[2]])
    
    coefs = gate_md(T().conj(), coefs, [axis[2]])
    coefs = gate_md(CX(), coefs, [axis[0],axis[2]])
    
    coefs = gate_md(T(), coefs, [axis[2]])
    coefs = gate_md(CX(), coefs, [axis[1],axis[2]])
    
    coefs = gate_md(T().conj(), coefs, [axis[2]])
    coefs = gate_md(CX(), coefs, [axis[0],axis[2]])
    
    coefs = gate_md(T(), coefs, [axis[1]])
    coefs = gate_md(T(), coefs, [axis[2]])
    
    coefs = gate_md(CX(), coefs, [axis[0],axis[1]])
    coefs = gate_md(H(), coefs, [axis[2]])
    
    coefs = gate_md(T(), coefs, [axis[0]])
    coefs = gate_md(T().conj(), coefs, [axis[1]])
    
    coefs = gate_md(CX(), coefs, [axis[0],axis[1]])
    
    return coefs

# def apply_Toffoli(coefs, axis):
#     coefs = apply_U(U = H(), coefs = coefs, axis = [axis[2]])
#     coefs = apply_U(U = CX(), coefs = coefs, axis = [axis[1],axis[2]])
    
#     coefs = apply_U(U = T().conj(), coefs = coefs, axis = [axis[2]])
#     coefs = apply_U(U = CX(), coefs = coefs, axis = [axis[0],axis[2]])
    
#     coefs = apply_U(U = T(), coefs = coefs, axis = [axis[2]])
#     coefs = apply_U(U = CX(), coefs = coefs, axis = [axis[1],axis[2]])
    
#     coefs = apply_U(U = T().conj(), coefs = coefs, axis = [axis[2]])
#     coefs = apply_U(U = CX(), coefs = coefs, axis = [axis[0],axis[2]])
    
#     coefs = apply_U(U = T(), coefs = coefs, axis = [axis[1]])
#     coefs = apply_U(U = T(), coefs = coefs, axis = [axis[2]])
    
#     coefs = apply_U(U = CX(), coefs = coefs, axis = [axis[0],axis[1]])
#     coefs = apply_U(U = H(), coefs = coefs, axis = [axis[2]])
    
#     coefs = apply_U(U = T(), coefs = coefs, axis = [axis[0]])
#     coefs = apply_U(U = T().conj(), coefs = coefs, axis = [axis[1]])
    
#     coefs = apply_U(U = CX(), coefs = coefs, axis = [axis[0],axis[1]])
    
#     return coefs


    
    
def expand_state(coefs, n_qubits):
    
    
    c = coefs.copy()
    # print(len(c), n_qubits)
    exp_coefs = np.zeros(2**n_qubits)
    exp_coefs[:len(c)] = c
    
    return exp_coefs

def split_system(coefs, na, nb):
    

    d = int(2**na)
    k = int(2**nb)
    
    coefs = np.reshape(coefs, [1,d,k,1])
    rho = np.tensordot(coefs, np.conjugate(coefs), [0,3])
    rho = np.reshape(rho, (d,k,d,k))
    rho_A = np.trace(rho, axis1=1, axis2=3)
    rho_B = np.trace(rho, axis1=0, axis2=2)

    s, v_a = LA.eig(rho_A)
    A_vec = v_a[:, 0]

    s, v_b = LA.eig(rho_B)
    B_vec = v_b[:, 0]

    return A_vec, B_vec
    




def reduce_state(coefs, N):
    c = coefs.copy()
    return c[:2**N]   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def apply_long_Toffoli(coefs, wires):

    n = len(wires)-2

    if len(wires) == 1:
        coefs_new = gate_md(X(), copy.deepcopy(coefs), wires)
        return coefs_new
    if len(wires) == 2:
        coefs_new = gate_md(CX(), copy.deepcopy(coefs), wires)
        return coefs_new
    if len(wires) == 3:
        coefs_new = apply_Toffoli(copy.deepcopy(coefs), wires)
        return coefs_new
        
    nq = int(np.log2(len(coefs)))
    
    coefs_new = expand_state(copy.deepcopy(coefs), nq+n)

    
    cw = np.array(wires).copy() + n
    ww = np.array(list(range(n)))

    
    coefs_new = apply_Toffoli(coefs_new, axis = [cw[0],cw[1],ww[0]])

    
    for i in range(1,n):
        coefs_new = apply_Toffoli(coefs_new, axis = [cw[i+1],ww[i-1],ww[i]])
        
        
    coefs_new = gate_md(CX(), coefs_new, [ww[-1],cw[-1]])
   
    
    for i in reversed(range(1,n)):
        coefs_new = apply_Toffoli(coefs_new, axis = [cw[i+1],ww[i-1],ww[i]])
      
        
    coefs_new = apply_Toffoli(coefs_new, axis = [cw[0],cw[1],ww[0]])

    coefs = reduce_state(coefs_new, nq)
    # _, coefs = split_system(coefs_new, n, nq)
    
    
    return coefs


    
    
def Fourier_transfer(state, t, inv = False, noisy = False, eps = 1e-6):
    
    if inv == True:
        for i in range(0, t - 1, 1):
            state.apply_U(H(), axis = [i])
            if noisy == True:
                eps = np.random.normal(0, e)
                state.apply_U(R_matrix(eps*np.pi, np.pi/2, eps), axis = [i])
            for j in range(2, t - i + 1, 1):
                state.apply_U(CR(j), axis = [i, i + j - 1])
            state.apply_U(H(), [t - 1])
        for i in range(int(t / 2)):
            state.apply_U(SWAP(), axis = [i, t - i - 1])

    else:
        
        for i in range(int(t / 2)):
            state.apply_U(SWAP(), axis = [i, t - i - 1])
        state.apply_U(H(), axis = [t - 1])
        if noisy == True:
                eps = np.random.normal(0, e)
                state.apply_U(R_matrix(eps*np.pi, np.pi/2, eps), axis = [i])
        for i in range(t - 2, -1, -1):
            for j in range(t - i, 1, -1):
                state.apply_U(CR(j).T.conjugate(), axis = [i, i + j - 1])
            state.apply_U(H(), [i])
    
                         
    

    

def phase_estimation(state, t, n, u_target):
    for i in range(0, t, 1):
        state.apply_U(H(), axis = [i])
        
    for i in range(t - 1, -1, -1):
        for j in range(2 ** (t - i - 1)):
#             U_pow = LA.matrix_power(U, n = 2**i)
            CU = build_CU(u_target)
            state.apply_U(CU, axis = [i] + list(np.arange(t, t + n, 1)))
    Fourier_transfer(state, t)


