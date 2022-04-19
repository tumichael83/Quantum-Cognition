# any kind of matrix stuff
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
# matrix exponent
from scipy.linalg import expm
# Aer is simultor, QC is circuit, execute executes, transpile is to adapt to real machines
from qiskit import Aer, QuantumCircuit, execute, transpile
# visualization
from qiskit.visualization import *
#
from qiskit.quantum_info.operators import Operator

def gen_quantum_randwalk(qubits, drift, diffusion, t):
    #create hamiltonian
    h_dimension = 2**qubits

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

    randwalk = QuantumCircuit (qubits,qubits)

    # lst of qubits
    qlist = []
    for i in range(qubits):
        qlist.append(i)

    # hadamard each qubit
    for i in range(qubits):
        randwalk.h(i)

    # add unitaries
    for i in range(t):
        randwalk.append(unitary_operator, qlist) # add unitary

    randwalk.measure(qlist,qlist)

    return randwalk

# TODO: make this take backend as input
def sim_qasm(circuit):
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=10000).result() # get to call the shots
    return job.get_counts()


#TODO: make this take a backend as input
def graph_quantum_sim(qubits, drift, diffusion, t):
    #compute num rows required
    nrows = ceil(t/3)
    fig, ax = plt.subplots(nrows, 3, figsize = (t,10))
    fig.suptitle("QASM Simulation of Reflecting Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    for i in range(0,t):
        randwalk = gen_quantum_randwalk(qubits,drift,diffusion,i)
        plot_histogram(sim_qasm(randwalk),color='midnightblue', ax=ax[i])
        ax[i].title.set_text("QRW timestep " +str(i))
        ax[i].set_ylim([0,1])
        fig.tight_layout()

    # the biggest quantum circuit
    randwalk.draw('mpl', filename="./walk implementations/reflecting boundaries/quantum graphs/figure.png")

    # the distrigutions
    plt.savefig("./walk implementations/reflecting boundaries/quantum graphs/timestep=" + str(t), format='png')
    plt.show()
