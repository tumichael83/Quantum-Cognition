# arrays
from math import ceil, sqrt, log
import numpy as np 
# matrix exponent, and more
from scipy.linalg import expm
# to graph things
from matplotlib import pyplot as plt

#TODO: file that contains hamiltonian function and simulation functions

def classical_absorbing(num_states, drift, diffusion, t, target):
    # num_states does include terminal states
    h_dimension = num_states
    
    #drift diagonal
    a = np.zeros((1,h_dimension))
    for i in range(0, h_dimension):
        # drift value as specified in gunnar's document
        a[0,i] = i*drift
    
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
    
    # initial, evenly distributed state vector 
    initial = [(1/sqrt(num_states)) for i in range (0, num_states)]

    # no decision matrix
    N = np.zeros((h_dimension, h_dimension))
    for i in range(1, h_dimension-1):
        N[i,i] = 1
    print(N)

    # iterate, discarding terminal states
    if (t > 0):
        prev_steps = np.linalg.matrix_power(N @ U, t-1)
    else:
        prev_steps = np.linalg.matrix_power(U, -1)

    # calculate each prob according to the formula
    prob = abs(np.linalg.norm(measure @ U @ prev_steps @ initial))**2

    print("Probability of state " + str(target) + ": " + str(prob))

    return prob

def classical_sim(n, drift, diffusion, t):
    #compute num rows required
    nrows = ceil(t/3)
    fig, ax = plt.subplots(nrows, 3, figsize = (16,10))
    fig.suptitle("Classical Simulation of Absorbing Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    bin_len = str(ceil(log(n, 2))) # how many 0s to pad to
    states_list = [format(state, '0'+bin_len+'b') for state in range(0,n)]

    for i in range(0,t):
        #get probs
        prob_list = []
        for j in range(0,n):
            prob_list.append(round(classical_absorbing(n, drift, diffusion, i, j), 3))

        # add bars on each small graph
        bar_plot = ax[i].bar(states_list, prob_list)
        for x, bar in enumerate(bar_plot): 
            ax[i].text(bar.get_x() + bar.get_width() / 2, bar.get_y()+bar.get_height(), str(prob_list[x]), ha="center", va="bottom")
            #plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.tight_layout()
        
        # fix each small graph
        ax[i].set_ylim([0,1])
        ax[i].title.set_text("QRW timestep " +str(i))
        plt.tight_layout()
    
    plt.savefig("./absorbing boundaries/classical graphs/timestep=" + str(t), format='png')
    plt.show()

# TODO: Response probability