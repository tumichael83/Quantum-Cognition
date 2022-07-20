import quantum_reflecting as qr
import time


joblist = qr.submit_jobs(2,2,4,3,'qasm_simulator')

joblist[-1].wait_for_final_state()
qr.graph_quantum_sim(2,2,4,3, 'qasm_simulator', joblist)
