# any kind of matrix stuff
import numpy as np
# matrix exponent
from scipy.linalg import expm
# basic circuit and transpile
from qiskit import Aer, QuantumCircuit, execute, transpile
# backend
from qiskit.providers.basicaer import QasmSimulatorPy
# visualization
from qiskit.visualization import *
# the complete package
from qiskit import terra
#
from qiskit.quantum_info.operators import Operator

def quantum_reflecting(n, drift, diffusion, t, initial):
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

    # obtain actual gate
    unitary_operator = Operator(U, input_dims = (h_dimension), output_dims = (h_dimension))

    randwalk = QuantumCircuit (n,n)

    # lst of qubits
    qlist = []
    for i in range(n):
        qlist.append(i)

    for i in range(n):
        randwalk.h() # hadamard each qubit

    randwalk.measure(qlist,qlist)

    #running the job on QASM simulator
    backend = Aer.get_backend('qasm_simulator')
    job = execute(randwalk, backend, shots=10000).result() # get to call the shots
    return job.get_counts()