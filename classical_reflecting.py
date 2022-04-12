# arrays
import numpy as np 
# matrix exponent, and more
from scipy.linalg import expm
# to graph things
from matplotlib import pyplot as plt

def classical_reflecting(n, drift, diffusion, t, target):
    #create hamiltonian
    h_dimension = n
    
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

    # measurement matrix for target state
    measure = np.zeros((h_dimension, h_dimension))
    measure[target, target] = 1

    # initial, evenly distributed state vector (for 4 states only)
    initial = [1/2, 1/2, 1/2, 1/2]

    # calculate each prob according to the formula
    prob = abs(np.linalg.norm(measure @ np.linalg.matrix_power(U, t) @ initial))

    print("Probability of state " + str(target) + ": " + str(prob))

    return prob

def classical_sim(n, drift, diffusion, t):
    #setup
    for t in range (0,t):
        prob_list = []
        for i in range(0,n):
            prob_list.append(classical_reflecting(n, drift, diffusion, t, i))
        print(prob_list)
        states_list = ["|00>", "|01>", "|10>", "|11>"]
        bar_plot = plt.bar(states_list, prob_list)

        # display, see csg290 doc from Raghav for details
        for bar in bar_plot:
            yval = bar.get_height()
            plt.text(bar.get_x()+0.2,yval+.01, round(yval, 3))
        plt.savefig("./walk implementations/classical graphs/timestep=" + str(t), format='png')
        plt.show()