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
    # TODO: check that this does fine with complex matrices
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
    #print(N)

    # iterate, discarding terminal state
    if (t > 0):
        prev_steps = np.linalg.matrix_power(N @ U, t-1)
    else:
        prev_steps = np.linalg.matrix_power(U, -1)

    # calculate each prob according to the formula
    prob = abs(np.linalg.norm(measure @ U @ prev_steps @ initial))**2

    #print("Probability of state " + str(target) + ": " + str(prob))

    return prob

# returns all the results for everything up to this state
def eval_walk(num_states, drift, diffusion, t):
    results = []

    for i in range(t+1):
        step = []
        for s in range(num_states):
            step.append(classical_absorbing(num_states, drift, diffusion, i, target=s))
        results.append(step)
    print(results)
    return results

def graph_state_probs(n, drift, diffusion, t, results, save_dest):
    #compute num rows required
    numsubplots = t+1
    fig, ax = plt.subplots(ceil(sqrt(numsubplots)), ceil(sqrt(numsubplots)), figsize = (16,10))
    fig.suptitle("Probability of Each State Prior to Absorption in Absorbing Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    bin_len = str(ceil(log(n, 2))) # how many 0s to pad to
    states_list = [format(state, '0'+bin_len+'b') for state in range(0,n)]

    f = open(save_dest+'/classical-sim-timestep='+str(t)+'-results.txt', 'w')

    # text file header
    f.write('absorbing boundaries\n')
    f.write('classical simulation\n')
    f.write('states = ' + str(n) + '\n')
    f.write('drift = ' + str(drift) + '\n')
    f.write('diffusion = '+str(diffusion)+'\n')

    for i in range(numsubplots):
        print("adding step="+str(i))
        f.write('\n-----timestep '+str(i)+'('+str(round(sum(results[i]),4))+')-----\n')

        # add bars on each small graph
        bar_plot = ax[i].bar(states_list, results[i])
        for x, bar in enumerate(bar_plot):
            ax[i].text(bar.get_x() + bar.get_width() / 2, bar.get_y()+bar.get_height(), str(round(results[i][x], 3)), ha="center", va="bottom")
            #plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.tight_layout()
            f.write(str(x)+':\t'+str(round(results[i][x], 3))+'\n')


        # fix each small graph
        ax[i].set_ylim([0,1])
        ax[i].title.set_text("t=" +str(i)+' '+str(sum(results[i])))
        plt.tight_layout()

    plt.savefig(save_dest+"/pre-absorbing-states-timestep=" + str(t), format='png')
    plt.show()
    f.close()

# TODO: Response probability
def graph_per_step_prob(n, drift, diffusion, t, save_dest):
    #compute num rows required
    numsubplots = t+1
    fig, ax = plt.subplots(ceil(sqrt(numsubplots)), ceil(sqrt(numsubplots)), figsize = (16,10))
    fig.suptitle("Probability of response at each timestep of Absorbing Boundaries QRW with drift=" +str(drift) + " & diffusion=" + str(diffusion))
    ax = ax.flatten()

    bin_len = str(ceil(log(n, 2))) # how many 0s to pad to
    states_list = [format(state, '0'+bin_len+'b') for state in [0, (n-1)]]

    for i in range(1, numsubplots):
        print("adding step="+str(i))
        #get probs
        prob_list = []

        # only add terminal states
        prob_list.append(round(classical_absorbing(n, drift, diffusion, i, 0), 3))
        prob_list.append(round(classical_absorbing(n, drift, diffusion, i, n-1), 3))

        # add bars on each small graph
        bar_plot = ax[i].bar(states_list, prob_list)
        for x, bar in enumerate(bar_plot):
            ax[i].text(bar.get_x() + bar.get_width() / 2, bar.get_y()+bar.get_height(), str(prob_list[x]), ha="center", va="bottom")
            #plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right')
            plt.tight_layout()

        # fix each small graph
        ax[i].set_ylim([0,0.3])
        ax[i].title.set_text("QRW timestep " +str(i))
        plt.tight_layout()

    plt.savefig(save_dest+"/per-step-decisions-timestep=" + str(t), format='png')
    plt.show()
