import quantum_reflecting as qr
# for transpile
from qiskit.test.mock import FakeBoeblingen

#drift 1 diffusion 0.33
qr.graph_quantum_sim(2,1,0.333,15)