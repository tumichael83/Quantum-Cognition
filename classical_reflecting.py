# arrays
import numpy as np 
# matrix exponent, and more
from scipy.linalg import expm, sinm, cosm

def classical_reflecting(n, drift, diffusion, t, initial):
    #create hamiltonian
    h_dimension = n
    
    #drift diagonal
    a = np.zeros((1,h_dimension))
    for i in range(0, h_dimension):
        # placeholder value for drift
        a[0,i] = drift - (h_dimension / 2) + i
    
    #diffusion diagonals
    b = np.zeros(1, h_dimension - 1)
    for i in range(0, h_dimension - 1):
        b[0,i] = diffusion
    
    # add diagonals to matrix
    H = np.diagflat(a) + np.diagflat(b,1) + np.diagflat(b,-1)

    # compute the unitary
    U = expm(-(1j)*H)

    # target state matrices
    measure_A = np.zeros((h_dimension, h_dimension))
    for i in range(h_dimension/2):
        M = np.zeros((h_dimension, h_dimension))
        M[i,i] = 1
        measure_A += M

    measure_B = np.zeros((h_dimension, h_dimension))
    for i in range(h_dimension/2, h_dimension):
        M = np.zeros((h_dimension, h_dimension))
        M[i,i] = 1
        measure_B += M

    # calculate each prob according to the formula
    probA = abs(np.linalg.norm(measure_A @ np.linalg.matrix_power(U, t) @ initial))
    probB = abs(np.linalg.norm(measure_B @ np.linalg.matrix_power(U, t) @ initial))

    print(probA, probB)

    return [probA, probB]



