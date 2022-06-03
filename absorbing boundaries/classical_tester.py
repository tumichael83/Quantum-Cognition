import classical_absorbing as ca


# 2 qubits/4 states, 1 ancilla, multiple lengths

timesteps = [35]

for t in timesteps:
    ca.graph_per_step_prob(4,1,.33,t,'/Users/michaeltu/Desktop/Independent CS Stuff/Quantum/walk implementations/absorbing boundaries/comparisons/circuit length')
    ca.graph_state_probs(4,1,.33,t, '/Users/michaeltu/Desktop/Independent CS Stuff/Quantum/walk implementations/absorbing boundaries/comparisons/circuit length')
