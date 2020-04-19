import numpy as np
from numba import jit, autojit
import pandas as pd
from tqdm import tqdm


# ================================================================================
#                               Initialize
# ================================================================================
@jit(nopython=True)
def simulate(edgelist, node_risk, T, N, alpha, beta, ensemble, outbreak_location):
    if outbreak_location == -1:
        outbreak_location = np.random.randint(N)
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False

    infection_time = np.zeros(N, dtype=float)
    recovery_time = np.zeros(N, dtype=float)

    num_infected = 1
    num_recovered = 0
    lst_infected[0] += 1 #initial condition is not added to node risk

    eIndex = 0
    t, u, v = edgelist[eIndex] # UNWEIGHTED!
    num_contacts = len(edgelist)
    # ---------------------------------------------------------------
    for time in xrange(T):
        if num_infected == 0:
            break

        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if np.random.random() < alpha: # IGNORING WEIGHTS!
                    infection_time[v] = t
                    isInfected[v] = True #ERROR: v -> newly infected
                    isSusceptible[v] = False
                    node_risk[v] += 1
                    num_infected += 1
            eIndex += 1
            if eIndex < num_contacts:
                t, u, v = edgelist[eIndex] # UNWEIGHTED!<
            else:
                break

        for n in xrange(N): #improve for speed: iterate only over isInfected == True
            if isInfected[n]:
                if np.random.random() < beta:
                    recovery_time[n] = t
                    isInfected[n] = False
                    isRecovered[n] = True
                    num_infected -= 1
                    num_recovered += 1

    return num_recovered, num_infected


def simulation_SIR(edgelist, T, alpha, beta, ensemble, outbreak_location, init_prob=1.):
    N = len( np.unique(edgelist[:,1:3]) )
    lst_infected = np.zeros(T + 1, dtype=np.int64) #SHIFTED
    lst_recovered = np.zeros(T + 1, dtype=np.int64) #SHIFTED
    lst_attack = np.empty(ensemble, dtype=np.int64)
    node_risk = np.zeros(N, dtype=np.int64)
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    for ii in tqdm(xrange(ensemble)):
        num_recovered, num_infected = simulate(edgelist, lst_infected, lst_recovered, node_risk, T, N, alpha, beta, ensemble, outbreak_location)
        lst_attack[ii] = num_recovered + num_infected

    return lst_infected, lst_recovered, lst_attack, node_risk

if __name__ == "__main__":
    # ================================================================================
    #                               Test Run
    # ================================================================================
    fname = "/home/andreasko/Schreibtisch/Simulation/HT09_edgelist_full_tuv.npy"
    alpha             = 0.02
    beta              = 0.0002
    ensemble          = 100
    init_prob         = 1.
    outbreak_location = 107
    
    edgelist = np.load(fname)
    nodes = set(edgelist[:,1]) | set(edgelist[:,2])
    N = len(nodes)
    T = np.max(edgelist[:,0]) + 1
    assert nodes == set(xrange(N))
    del nodes
    print "Number of nodes:", N
    print "Time runs from", edgelist[0,0], "to", edgelist[-1,0], "in", len(np.unique(edgelist[:,0])), "time steps"

    lst_infected, lst_recovered, lst_attack, node_risk = simulation_SIR(edgelist, 
                                                                            T,
                                                                            alpha, 
                                                                            beta,
                                                                            ensemble,
                                                                            outbreak_location,
                                                                            init_prob)
    np.savez("test", 
            infected=lst_infected,
            recovered=lst_recovered,
            attacked=lst_attack,
            alpha=alpha,
            beta=beta,
            ensemble=ensemble,
            data=fname)

