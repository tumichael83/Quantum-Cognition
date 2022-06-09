import quantum_absorbing as qa
from qiskit.providers.ibmq import least_busy

backend = least_busy(map(qa.provider.get_backend, ['ibmq_santiago','ibmq_bogota','ibmq_lima','ibmq_belem','ibmq_quito','ibmq_jakarta','ibmq_manila','ibm_lagos','ibm_perth'])).configuration().backend_name

#backend = 'qasm_simulator'
qc_list = [qa.gen_quantum_randwalk(2,1,.33,t) for t in range(15+1)]

job = qa.provider.get_backend('ibmq_jakarta').jobs(1)[0]

job.wait_for_final_state()

qa.graph_quantum_sim(2, 1, .33, 15, backend, job, '/Users/michaeltu/Desktop/Independent CS Stuff/Quantum/walk implementations/absorbing boundaries/comparisons/circuit length')
