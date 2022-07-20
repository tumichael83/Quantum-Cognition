import quantum_absorbing as qa
from qiskit.providers.ibmq import least_busy

backend = 'qasm_simulator'
qc_list = [qa.gen_quantum_randwalk(3,1,.33,t) for t in range(3+1)]

#job = qa.provider.get_backend('qasm-simulator').jobs(1)[0]
job = qa.run_backend(qc_list, 'qasm_simulator')

job.wait_for_final_state()

qa.graph_quantum_sim(3, 1, .33, 3, backend, job, '/Users/michaeltu/Desktop/Independent CS Stuff/Quantum/walk implementations/absorbing boundaries/comparisons/circuit length')
