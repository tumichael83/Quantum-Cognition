# any kind of matrix stuff
from math import ceil, sqrt
from matplotlib import pyplot as plt
import numpy as np
# matrix exponent
from scipy.linalg import expm
# Aer is simultor, QC is circuit, execute executes, transpile is to adapt to real machines
from qiskit import Aer, QuantumCircuit, execute, transpile, assemble
# visualization
from qiskit.visualization import *
# Unitary operator
from qiskit.quantum_info.operators import Operator

from qiskit import IBMQ
#set up backend
IBMQ.load_account()
provider = IBMQ.get_provider(group='yale-uni-1')
#mybackend = provider.get_backend('ibmq_manila')
mybackend = Aer.get_backend('qasm_simulator')
config = mybackend.configuration()


#num shots to use when running
numshots = 8192

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
    randwalk = QuantumCircuit(total_qubits, (t-1)*(state_qubits-1) + state_qubits)

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
        ## we reset the ancilla 2 times to clear it, 
        ## before we measure

        ancilla = state_qubits;

        # measure each pair of qubits
        for k in range(state_qubits-1):
            randwalk.reset([ancilla]*(1))
            randwalk.cx(k, ancilla)
            randwalk.cx(k+1, ancilla)
            randwalk.measure(ancilla, i*(state_qubits-1) + k)


    # final unitary (without ancilla)
    randwalk.append(unitary_operator, qlist)

    # the list expansion on the right is just to measure the bits in
    # qindex onto the very last cbits
    randwalk.measure(qlist, [((t-1)*(state_qubits-1) + qbIndex) for qbIndex in qlist])

    return randwalk

# TODO: make this take backend as input
def run_backend(circuit):
    trans_c = transpile(circuit, basis_gates=config.basis_gates)

    print('assembling for '+config.backend_name+'...')
    qobj = assemble(trans_c, backend=mybackend,shots=numshots)

    print('running on '+config.backend_name+'...')
    job = mybackend.run(qobj) # get to call the shots
    result = job.result()

    print(result.get_counts())

    return result.get_counts()

def select_counts(circuit, qubits):
    result_dict = run_backend(circuit)
    print("selecting valid runs...")

    good_runs = {}

    for k in result_dict.keys():
        if '0' not in k[qubits:]:
            good_runs[k[:qubits]] = result_dict[k]/8192

    print(good_runs)
    return good_runs


#TODO: make this take a backend as input
def graph_quantum_sim(qubits, drift, diffusion, t):
    #compute num rows required
    numsubplots = t+1
    fig, ax = plt.subplots(ceil(sqrt(numsubplots)), ceil(sqrt(numsubplots)), figsize = (16,10))
    fig.suptitle(config.backend_name + "Simulation of Absorbing Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    for i in range(t+1):
        randwalk = gen_quantum_randwalk(qubits, drift, diffusion, i)
        plot_histogram(select_counts(randwalk, qubits), color='cyan', ax=ax[i])
        ax[i].title.set_text('QRW timestep '+str(i))
        ax[i].set_ylim([0,1])
        fig.tight_layout()

    # the distrigutions
    plt.savefig("./absorbing boundaries/quantum graphs/timestep=" + str(t), format='png')
    plt.show()
