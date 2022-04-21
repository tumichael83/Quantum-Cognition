# any kind of matrix stuff
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
# matrix exponent
from scipy.linalg import expm
# Aer is simultor, QC is circuit, execute executes, transpile is to adapt to real machines
from qiskit import Aer, QuantumCircuit, execute
# visualization
from qiskit.visualization import *
# Unitary operator
from qiskit.quantum_info.operators import Operator
# no idea what this does
from qiskit.result import marginal_counts

# I'm only going to use a 2 qubit walk on this for now
def gen_quantum_randwalk(state_qubits, drift, diffusion, t):
    #create hamiltonian
    h_dimension = 2**state_qubits

    total_qubits = state_qubits + 1 # hardcoded 1 for 1 ancilla

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

    # measure ancilla after every unitary apart from last (t-1), then measure state qubits
    randwalk = QuantumCircuit (total_qubits, t*(state_qubits-1) + state_qubits)

    # lst of state_qubits
    qlist = []
    for i in range(state_qubits):
        qlist.append(i)

    # hadamard each state qubit
    for i in range(state_qubits):
        randwalk.h(i)

    # unitaries and ancilla before final measurement
    for i in range(t - 1):
        randwalk.append(unitary_operator, qlist) # add unitary

        ## currently this is just one ancilla qubit!!!!!!
        ## state_qubits  is the index of the ancilla qubit
        ## we reset the ancilla state_qubits number of times to clear it
        ## before we measure

        ancilla = state_qubits;

        # measure each pair of qubits
        for k in range(state_qubits-1):
            randwalk.reset([ancilla]*(2))
            randwalk.cx(k, ancilla)
            randwalk.cx(k+1, ancilla)
            randwalk.measure(ancilla, i*state_qubits + k)


    # final unitary (without ancilla)
    randwalk.append(unitary_operator, qlist)

    # the list expansion on the right is just to measure the bits in
    # qindex onto the very last cbits
    randwalk.measure(qlist, [(t + qbIndex) for qbIndex in qlist])

    return randwalk

# TODO: make this take backend as input
def sim_qasm(circuit, t):
    backend = Aer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots=10000).result() # get to call the shots
    return job.get_counts()

#TODO: make this take a backend as input
def graph_quantum_sim(qubits, drift, diffusion, t):
    #compute num rows required
    nrows = ceil(t/3)
    fig, ax = plt.subplots(nrows, 3, figsize = (16,10))
    fig.suptitle("QASM Simulation of Absorbing Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    for i in range(0,t):
        print("current timestep= " + str(i))
        randwalk = gen_quantum_randwalk(qubits,drift,diffusion,i)
        plot_histogram(sim_qasm(randwalk),color='midnightblue', ax=ax[i])
        ax[i].title.set_text("QRW timestep " +str(i))
        ax[i].set_ylim([0,1])
        fig.tight_layout()

    # the biggest quantum circuit
    randwalk.draw('mpl', filename="./absorbing boundaries/quantum graphs/figure.png")

    # the distrigutions
    plt.savefig("./absorbing boundaries/quantum graphs/timestep=" + str(t), format='png')
    plt.show()
