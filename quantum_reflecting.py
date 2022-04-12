# any kind of matrix stuff
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
# matrix exponent
from scipy.linalg import expm
# basic circuit and transpile
from qiskit import Aer, QuantumCircuit, execute, transpile
# backend
from qiskit.providers.basicaer import QasmSimulatorPy
# visualization
from qiskit.visualization import *
#
from qiskit.quantum_info.operators import Operator

def quantum_reflecting(n, drift, diffusion, t):
    #create hamiltonian
    h_dimension = 2**n
    
    #drift diagonal
    a = np.zeros((1,h_dimension))
    for i in range(0, h_dimension):
        # placeholder value for drift
        a[0,i] = drift - (h_dimension / 2) + i
    
    #diffusion diagonals
    b = np.zeros((1, h_dimension - 1))
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

    # hadamard each qubit
    for i in range(n):
        randwalk.h(i) 
        
    # add unitaries
    for i in range(t):
        randwalk.append(unitary_operator, qlist) # add unitary

    randwalk.measure(qlist,qlist)

    #randwalk.draw(output='mpl')

    #running the job on QASM simulator
    backend = Aer.get_backend('qasm_simulator')
    job = execute(randwalk, backend, shots=10000).result() # get to call the shots
    return job.get_counts()

def quantum_sim_qasm(qubits, drift, diffusion, t):
    #compute num rows required
    nrows = ceil(t/3)
    fig, ax = plt.subplots(nrows, 3, figsize = (t,10))
    fig.suptitle("QASM Simulation of Reflecting Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    for i in range(0,t):
        plot_histogram(quantum_reflecting(qubits,drift,diffusion,i),color='midnightblue', ax=ax[i])
        ax[i].title.set_text("QRW timestep " +str(i))
        plt.ylim([0,1])
        fig.tight_layout()
    
    plt.savefig("./walk implementations/quantum graphs/my timestep=" + str(t), format='png')
    plt.show()