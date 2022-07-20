# any kind of matrix stuff
from math import ceil, sqrt
from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import expm # matrix exponent

# circuit stuff
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit import transpile, assemble              # adapts stuff for backend

from qiskit.visualization import *

#timing info
import datetime

#simulators and systems
from qiskit import Aer, IBMQ
#set up backend
IBMQ.load_account()
provider = IBMQ.get_provider(group='yale-uni-1')
mybackend = provider.get_backend('ibmq_santiago')
#mybackend = Aer.get_backend('qasm_simulator')
config = mybackend.configuration()

yale_backends = ['ibmq_armonk',
'ibmq_santiago',
'ibmq_bogota',
'ibmq_lima',
'ibmq_belem',
'ibmq_quito',
'simulator_statevector',
'simulator_mps',
'simulator_extended_stabilizer',
'simulator_stabilizer',
'ibmq_jakarta',
'ibmq_manila',
'ibm_lagos',
'ibm_perth']

import pprint

# I'm only going to use a 2 qubit walk on this for now
def gen_quantum_randwalk(state_qubits, drift, diffusion, t):
    #create hamiltonian and unitary
    h_dimension = 2**state_qubits

    total_qubits = state_qubits + 1 # hardcoded 1 for 1 ancilla

    a = np.zeros((1,h_dimension))
    for i in range(0, h_dimension):
        # placeholder value for drift
        a[0,i] = drift*i

    b = np.zeros((1, h_dimension - 1))
    for i in range(0, h_dimension - 1):
        b[0,i] = diffusion

    H = np.diagflat(a) + np.diagflat(b,1) + np.diagflat(b,-1)

    U = expm(-(1j)*H)

    unitary_operator = Operator(U, input_dims = (h_dimension), output_dims = (h_dimension))


    #create quantum circuit
    if t==0:
        randwalk = QuantumCircuit(total_qubits, state_qubits)
        qlist = [i for i in range(state_qubits)]
        for i in range(state_qubits):
            randwalk.h(i)
        randwalk.measure(qlist, qlist)
        return randwalk

    # measure ancilla after every unitary apart from last (t-1), then measure state qubits
    randwalk = QuantumCircuit(total_qubits, (t-1)*(state_qubits-1) + state_qubits)

    # lst of state_qubits
    qlist = [i for i in range(state_qubits)]

    # hadamard each state qubit
    for i in range(state_qubits):
        randwalk.h(i)

    for i in range(t-1):
        ## currently this is just one ancilla qubit!!!!!!
        ## state_qubits  is the index of the ancilla qubit
        ## we reset the ancilla to clear it, 
        ## before we measure

        randwalk.append(unitary_operator, qlist) # add unitary

        ancilla = state_qubits;

        # measure each pair of qubits
        for k in range(state_qubits-1):
            randwalk.reset([ancilla]*(1))
            randwalk.cx(k, ancilla)
            randwalk.cx(k+1, ancilla)
            randwalk.measure(ancilla, i*(state_qubits-1)+k)

    randwalk.append(unitary_operator, qlist)

    # the list expansion on the right is just to measure the bits in
    # qindex onto the very last cbits
    randwalk.measure(qlist, [((t-1)*(state_qubits-1) + qbIndex) for qbIndex in qlist])

    return randwalk

# collect the time information from the result object
# https://quantumcomputing.stackexchange.com/questions/3901/comparing-run-times-on-ibm-quantum-experience
def run_backend(qc_list, mybackend_name):

    if mybackend_name in yale_backends:
        mybackend = provider.get_backend(mybackend_name)
    else:
        mybackend = Aer.get_backend(mybackend_name)
        print(mybackend_name)

    config = mybackend.configuration()
    print('transpiling for '+config.backend_name+'...')
    qc_list = transpile(qc_list ,backend=mybackend, basis_gates=config.basis_gates)

    print('assembling for '+config.backend_name+'...')
    numshots = min(config.max_shots,100000)
    qobj = assemble(qc_list , backend=mybackend,shots=numshots)

    print('running on '+config.backend_name+'...')
    job = mybackend.run(qobj) # get to call the shots

    return job # the list of circuits is submitted with the index being the # of timesteps

def select_counts(job, qubits, t):
    #myresults = run_backend(circuit)
    qc_counts = job.result().get_counts()
    print("selecting valid runs...")

    # no longer worrying about dictionaries and keys, now everything is just going to be in order
    all_timesteps = []
    for counts in qc_counts:

        # the format turns each i into a binary string with <qubits> bits
        final_states_counts = dict.fromkeys([format(i, '0'+str(qubits)+'b') for i in range(2**qubits)], 0)

        for k in sorted(counts.keys()):
            final_state = k[:qubits]

            final_states_counts[final_state] += counts[k] # treat it as valid, then check if its not

            start = qubits
            step = qubits-1
            end = start + step*t

            #print(str(k) + ': ' + str(counts[k]))
            while start < end:
                if '0'*(qubits-1) in k[start:start+step]:
                    final_states_counts[final_state] -= counts[k]
                    #print(' '*start+'^')
                    break
                start+=step

        all_timesteps.append(final_states_counts)
        print(sum(final_states_counts.values()))

    return all_timesteps


def graph_quantum_sim(qubits, drift, diffusion, t, mybackend_name, job, save_dest):

    if mybackend_name in yale_backends:
        mybackend = provider.get_backend(mybackend_name)
    else:
        mybackend = Aer.get_backend(mybackend_name)
    config = mybackend.configuration()
    numshots = min(config.max_shots, 100000)

    f = open(save_dest+'/'+config.backend_name+'-timestep='+str(t)+'-results.txt', 'w')

    # text file header
    f.write('absorbing boundaries\n')
    f.write(config.backend_name+'\n')
    f.write('qubits = ' + str(qubits) + '\n')
    f.write('drift = ' + str(drift) + '\n')
    f.write('diffusion = '+str(diffusion)+'\n')
    if mybackend_name in yale_backends:
        f.write('JobID = '+str(job.job_id())+'\n')

    #compute num rows required
    numsubplots = t+1
    fig, ax = plt.subplots(ceil(sqrt(numsubplots)), ceil(sqrt(numsubplots)), figsize = (16,10))
    fig.suptitle(config.backend_name + " Absorbing Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    runtimes = []

    results = select_counts(job, qubits, t)

    for i, step in enumerate(results):

        print("graphing step="+str(i))

        # probabilities
        states = list(step.keys())
        states.sort()
        vals = [step[s]/numshots for s in states]

        # graph
        bar_plot = ax[i].bar(states, vals)
        for x,  bar in enumerate(bar_plot):
            ax[i].text(bar.get_x() + bar.get_width() / 2, bar.get_y()+bar.get_height(), str(round(vals[x], 3)), ha="center", va="bottom")
            plt.tight_layout()

        ax[i].set_ylim([0,1])
        ax[i].title.set_text("t=" +str(i)+ ' (shots='+str(sum(step.values()))+'/'+str(numshots)+')')
        plt.tight_layout()

        f.write('\n-----timestep '+str(i)+':'+str(sum(step.values()))+'-----\n')
        for s in range(2**qubits):
            f.write(str(s)+':\t'+str(vals[s])+'\n')

        #if mybackend_name in yale_backends:
            #runtimes.append(job.time_per_step()['COMPLETED'] - job.time_per_step()['RUNNING'])


    # the distributions
    plt.savefig(save_dest+'/'+config.backend_name+" timestep=" + str(t), format='png')
    plt.show()

    runtime = sum(runtimes, datetime.timedelta())
    f.write('\ntotal runningtime: '+str(runtime))
    f.close()

    return
