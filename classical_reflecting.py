# arrays
from math import ceil
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
    prob = abs(np.linalg.norm(measure @ np.linalg.matrix_power(U, t) @ initial))**2

    print("Probability of state " + str(target) + ": " + str(prob))

    return prob

def classical_sim(n, drift, diffusion, t):
    #compute num rows required
    nrows = ceil(t/3)
    fig, ax = plt.subplots(nrows, 3, figsize = (t,10))
    fig.suptitle("QASM Simulation of Reflecting Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()
    states_list = ["|00>", "|01>", "|10>", "|11>"]

    for i in range(0,t):
        #get probs
        prob_list = []
        for j in range(0,n):
            prob_list.append(round(classical_reflecting(n, drift, diffusion, i, j), 3))

        # add bars on each small graph
        bar_plot = ax[i].bar(states_list, prob_list)
        for x, bar in enumerate(bar_plot): 
            ax[i].text(bar.get_x() + bar.get_width() / 2, bar.get_y()+bar.get_height(), str(prob_list[x]), ha="center", va="top")
            yval = bar.get_height()
            plt.text(bar.get_x(),bar.get_y()+.01, round(yval, 3))

        # fix each small graph
        ax[i].title.set_text("QRW timestep " +str(i))
        plt.ylim([0,1])
        fig.tight_layout()
    
    plt.savefig("timestep=" + str(t), format='png')
    plt.show()