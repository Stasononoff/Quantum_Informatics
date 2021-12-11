import numpy as np

def expand(matrix, wire, N):
    new_matrix = matrix
    for i in range(N):
        if i>wire:
            new_matrix = np.kron([[1,0],
                                  [0,1]], new_matrix)
        elif i<wire:
            new_matrix = np.kron(new_matrix, [[1,0],
                                  [0,1]])
            
    return np.array(new_matrix)
            
def expandNq(matrix, wires, N):
    new_matrix = matrix
    for i in range(N):
        if i>max(wires):
            new_matrix = np.kron([[1,0],
                                  [0,1]], new_matrix)
        elif i<min(wires):
            new_matrix = np.kron(new_matrix, [[1,0],
                                  [0,1]])
            
    return np.array(new_matrix)




def I(wire = 0, N = 1):
    return expand(np.array([[1, 0],
                     [0, 1]], dtype=complex), wire, N)


def X(wire = 0, N = 1):
    return expand(np.array([[0, 1],
                     [1, 0]], dtype=complex), wire, N)


def Y(wire = 0, N = 1):
    return expand(np.array([[0, -1j],
                     [1j, 0]], dtype=complex), wire, N)


def Z(wire = 0, N = 1):
    return expand(np.array([[1, 0],
                     [0, -1]], dtype=complex), wire, N)


def H(wire = 0, N = 1):
    return expand((1 / np.sqrt(2)) * np.array([[1, 1],
                                        [1, -1]], dtype=complex), wire, N)

# def ZERO_GATE(wire = 0, N = 1):
#     return expand((np.array([[1, 0],
#                              [0, 0]], dtype=complex), wire, N)


def Rn(alpha, phi, theta, wire = 0, N = 1):
    nx = np.sin(alpha) * np.cos(phi)
    ny = np.sin(alpha) * np.sin(phi)
    nz = np.cos(alpha)
    return expand(np.cos(theta / 2) * I() - 1j * np.sin(theta / 2) * (nx * X() + ny * Y() + nz * Z()), wire, N)


def Rn_random(wire = 0, N = 1):
    alpha = np.pi * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    theta = 2 * np.pi * np.random.rand()
    return expand(Rn(alpha, phi, theta), wire, N)


def CR(k, wires = [0,1], N = 2):
    return expandNq(np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, np.exp(1j*2*np.pi/(2**k))]], dtype=complex), wires, N)


# def R_K_C(k):
#     return np.array([[1, 0, 0, 0],
#                      [0, 1, 0, 0],
#                      [0, 0, 1, 0],
#                      [0, 0, 0, np.exp(2 * np.pi * 1j / (2 ** k))]], dtype=complex)



def CX(wires = [0,1], N = 2):
    return expandNq(np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex), wires, N)


def CY(wires = [0,1], N = 2):
    return expandNq(np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, -1j],
                     [0, 0, 1j, 0]], dtype=complex), wires, N)


def CZ(wires = [0,1], N = 2):
    return expandNq(np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex), wires, N)


def SWAP(wires = [0,1], N = 2):
    return expandNq(np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex), wires, N)


def TOFFOLI(wires = [0,1,2], N = 3):
    return expandNq(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex), wires, N)

def T(wire = 0, N = 1):
    return expand(np.array([[1, 0],
                     [0, np.exp(1j*np.pi/4)]], dtype=complex), wire, N)

# Можно использовать как матрицу ошибок:
def R_matrix(delta =0, theta = 0, phi = 0):
    R = np.zeros((2,2)).astype('complex')
    R[0][0] = np.cos(delta/2) - 1j*np.cos(theta)*np.sin(delta/2)
    R[1][0] = -1j*np.sin(theta)*np.sin(delta/2)*np.exp(1j*phi)
    R[0][1] = -1j*np.sin(theta)*np.sin(delta/2)*np.exp(-1j*phi)
    R[1][1] = np.cos(delta/2) + 1j*np.cos(theta)*np.sin(delta/2)
    return R


