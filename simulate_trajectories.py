from __future__ import print_function
import pdb
import numpy as np
from numba import jit, autojit
from simulate_single_realization import simulate_single_trajectory, simulate_single_target_node, simulate_single_target_node_static, simulate_single_trajectory_multiple_outbreak_locations
from tqdm import tqdm_notebook as tqdm

@jit(nopython=True, nogil=True, cache=False)
def update_error(new_trajectory, iteration, stop_flag, old_trajectory, idx, tol, deque_length, error_deque, error_list, N):
    new_error = np.sqrt( np.square( new_trajectory - old_trajectory ).sum() / N )
    error_deque[idx] = new_error
    idx = (idx + 1) % deque_length
    
    max_error = error_deque.max()
    error_list[ int(iteration) ] = max_error
    if max_error < tol:
        stop_flag = True
    
    old_trajectory[:] = new_trajectory
    return stop_flag, (old_trajectory, idx, tol, deque_length, error_deque, error_list, N)

#@jit(int64, float64[:], float64[:](int32, int32)), nopython=True)
@jit(nopython=True, nogil=True, cache=False)
def simulation(edgelist, directed, num_nodes, alpha, beta, init_prob=1, iter_max=100, iter_min=0, tol=1e-3, verbose=False):
    #outbreak_location = int(outbreak_location)
    #if outbreak_location == -1:
    #    outbreak_locations = np.arange(num_nodes, dtype=np.int64)
    #else: 
    #    outbreak_locations = np.array([outbreak_location], dtype=np.int64)
    
    Tmax = edgelist[-1,0] + 2 #
    infected = np.zeros(Tmax, dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    recovered = np.zeros(Tmax, dtype=np.float64)
    
    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max, dtype=np.float64) * -1. #save the error trajectory here
    old_trajectory = np.zeros(Tmax, dtype=np.float64)
    error_stats = old_trajectory, 0, tol, deque_length, error_deque, error_list, len(old_trajectory) #old_value, idx, tol, deque_length, error_deque, error_list
    
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    stop_flag = False
    j = 1.
    for it in xrange(iter_max):
        #norm = 0
        for outbreak_location in xrange(num_nodes):
            inf_single_run, rec_single_run = simulate_single_trajectory( edgelist, directed, num_nodes, alpha, beta, outbreak_location, Tmax )
            
            infected  *= (j-1.) / j
            infected  += inf_single_run / j
            recovered *= (j-1.) / j
            recovered += rec_single_run / j
            j += 1.
            # Check convergence
            stop_flag, error_stats = update_error(infected, it, stop_flag, *error_stats)

        if stop_flag:
            if it > iter_min:
                break
        
    return it, infected, recovered, error_list

# ================================================================================
#                               Return full distributions
# ================================================================================
@jit(nopython=True, nogil=True, cache=False)
def distribution(edgelist, directed, num_nodes, alpha, beta, init_prob=1, iter_max=100, iter_min=0, tol=1e-3, verbose=False, threshold=0.):
    #outbreak_location = int(outbreak_location)
    #if outbreak_location == -1:
    #    outbreak_locations = np.arange(num_nodes, dtype=np.int64)
    #else: 
    #    outbreak_locations = np.array([outbreak_location], dtype=np.int64)
    
    Tmax = edgelist[-1,0] + 2 #
    infected = np.zeros(Tmax, dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    recovered = np.zeros(Tmax, dtype=np.float64)
    
    infected_array = np.zeros((num_nodes, Tmax), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    recovered_array = np.zeros((num_nodes, Tmax), dtype=np.float64)

    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max, dtype=np.float64) * -1. #save the error trajectory here
    old_trajectory = np.zeros(Tmax, dtype=np.float64)
    error_stats = old_trajectory, 0, tol, deque_length, error_deque, error_list, len(old_trajectory) #old_value, idx, tol, deque_length, error_deque, error_list
    
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    stop_flag = False
    cnt = 1.

    for it in xrange(iter_max): #tqdm(xrange(iter_max)):
        #norm = 0
        for outbreak_location in xrange(num_nodes):
            inf_single_run, rec_single_run = simulate_single_trajectory( edgelist, directed, num_nodes, alpha, beta, outbreak_location, Tmax-1 )

            for time in xrange(Tmax):
                infected_array[int(inf_single_run[time]), time] += 1
                recovered_array[int(inf_single_run[time]), time] += 1

            if inf_single_run[-1] + rec_single_run[-1] >= threshold:
                infected  *= (cnt-1.) / cnt
                infected  += inf_single_run / cnt #type conversion: type(cnt) = float
                recovered *= (cnt-1.) / cnt
                recovered += rec_single_run / cnt
                cnt += 1.

                # Check convergence
                stop_flag, error_stats = update_error(infected, it, stop_flag, *error_stats)

                if stop_flag:
                    if it > iter_min:
                        break
    '''
    results = {
        "iterations" : it,
        "mean_infected" : infected,
        "mean_recovered" : recovered,
        "mean_infected_above_threshold" : infected_threshold,
        "mean_recovered_above_threshold" : recovered_threshold,
        "infected_distribution" : infected_array,
        "recovered_distribution" : recovered_array,
        "errors" : error_list
    }
    '''
    return it, infected, recovered, infected_array, recovered_array, error_list


# ================================================================================
#                 multiple outbreak locations, full distributions
# ================================================================================
#@jit(nopython=True, nogil=True, cache=False)
def multiple_outbreak_locations_distribution(edgelist,
                                             directed,
                                             num_nodes, 
                                             alpha, 
                                             beta, 
                                             init_infection_prob=1,
                                             Tmax = -1,
                                             iter_max=100, 
                                             iter_min=0, 
                                             tol=1e-3, 
                                             verbose=False, 
                                             threshold=0.):

    #outbreak_location = int(outbreak_location)
    #if outbreak_location == -1:
    #    outbreak_locations = np.arange(num_nodes, dtype=np.int64)
    #else: 
    #    outbreak_locations = np.array([outbreak_location], dtype=np.int64)
    
    if Tmax == -1:
        Tmax = edgelist[-1,0] + 2 #add initial state as additional time step
    else:
        Tmax = int(Tmax) + 2 #add initial state as additional time step
    
    infected_mean = np.zeros(Tmax, dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack_mean = np.zeros(Tmax, dtype=np.float64)
    infected_distribution = np.zeros((num_nodes, Tmax), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack_distribution = np.zeros((num_nodes, Tmax), dtype=np.float64)
    results = (infected_mean,
               attack_mean,
               infected_distribution,
               attack_distribution)

    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max, dtype=np.float64) * -1. #save the error trajectory here
    old_trajectory = np.zeros(Tmax, dtype=np.float64)
    error_stats = old_trajectory, 0, tol, deque_length, error_deque, error_list, len(old_trajectory) #old_value, idx, tol, deque_length, error_deque, error_list
    
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    stop_flag = False
    for it in tqdm(range(1, iter_max+1)): #tqdm(xrange(iter_max)):
        #norm = 0
        outbreak_locations = np.random.random(num_nodes) < init_infection_prob

        results = simulate_single_trajectory_multiple_outbreak_locations( 
            edgelist, 
            directed, 
            num_nodes, 
            alpha, 
            beta, 
            outbreak_locations, 
            Tmax - 1,
            it,
            *results)

        # Check convergence
        stop_flag, error_stats = update_error(infected_mean, it, stop_flag, *error_stats)

        if stop_flag:
            if it > iter_min:
                break
    '''
    results = {
        "iterations" : it,
        "mean_infected" : infected,
        "mean_recovered" : recovered,
        "mean_infected_above_threshold" : infected_threshold,
        "mean_recovered_above_threshold" : recovered_threshold,
        "infected_distribution" : infected_array,
        "recovered_distribution" : recovered_array,
        "errors" : error_list
    }
    '''
    return it, infected_mean, attack_mean, infected_distribution, attack_distribution, error_list



# ================================================================================
#                               fixed source, full distributions
# ================================================================================
#@jit(nopython=True, nogil=True, cache=False)
def fixed_source_full_distribution(edgelist, directed, num_nodes, outbreak_location, alpha, beta, Tmax = -1, init_prob=1, iter_max=100, iter_min=0, tol=1e-3, verbose=False, threshold=0., ignore_dead=False):

    if Tmax == -1:
        Tmax = edgelist[-1,0] + 2 #add initial state as additional time step
    else:
        Tmax = int(Tmax) + 2 #add initial state as additional time step

    infected = np.zeros(Tmax, dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack = np.zeros(Tmax, dtype=np.float64)
    
    infected_array = np.zeros((num_nodes, Tmax), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack_array = np.zeros((num_nodes, Tmax), dtype=np.float64)

    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max, dtype=np.float64) * -1. #save the error trajectory here
    old_trajectory = np.zeros(Tmax, dtype=np.float64)
    error_stats = old_trajectory, 0, tol, deque_length, error_deque, error_list, len(old_trajectory) #old_value, idx, tol, deque_length, error_deque, error_list
    
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    stop_flag = False
    cnt = 1.

    for it in tqdm(range(iter_max)): #tqdm(xrange(iter_max)):
        #norm = 0
        inf_single_run, rec_single_run = simulate_single_trajectory( edgelist, directed, num_nodes, alpha, beta, outbreak_location, Tmax - 1 )

        if inf_single_run[-1] + rec_single_run[-1] >= threshold:
            if (not ignore_dead) or (ignore_dead and inf_single_run[-1] > 0):
                for time in xrange(Tmax):
                    infected_array[int(inf_single_run[time]), time] += 1
                    attack_array[int(inf_single_run[time] + rec_single_run[time]), time] += 1

                infected  *= (cnt-1.) / cnt
                infected  += inf_single_run / cnt #type conversion: type(cntrec) = float
                attack *= (cnt-1.) / cnt
                attack += ( rec_single_run + inf_single_run ) / cnt

                #pdb.set_trace() 
                # Check convergence
                stop_flag, error_stats = update_error(infected, cnt, stop_flag, *error_stats)

                if stop_flag:
                    if it > iter_min:
                        break
                cnt += 1.
    
    return cnt, infected, attack, infected_array, attack_array, error_list


# ================================================================================
#                               Fixed outbreak location
# ================================================================================
#@jit(nopython=True, nogil=True)
def fixed_source_all_trajectories(edgelist, directed, num_nodes, source, alpha, beta, ensemble=100, threshold=0., max_iter=1000000, Tmax=-1):
    if threshold == 0:
        max_iter = ensemble
    
    if Tmax == -1:
        Tmax = edgelist[-1,0] + 2 #add initial state as additional time step
    else:
        Tmax = int(Tmax) + 2 #add initial state as additional time step

    infected = np.zeros((ensemble, Tmax), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack = np.zeros((ensemble, Tmax), dtype=np.float64)
    mean_infected = np.zeros(Tmax, dtype=np.float64)

    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(max_iter, dtype=np.float64) * -1. #save the error trajectory here
    old_trajectory = np.zeros(Tmax, dtype=np.float64)
    error_stats = old_trajectory, 0, 0, deque_length, error_deque, error_list, len(old_trajectory) #old_value, idx, tol, deque_length, error_deque, error_list
    stop_flag = False
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    cnt = 1
    for it in tqdm(xrange(max_iter)):
        inf_single_run, rec_single_run = simulate_single_trajectory( edgelist, directed, num_nodes, alpha, beta, source, Tmax - 1  )
        
        if inf_single_run[-1] + rec_single_run[-1] > threshold:
            infected[cnt, :] = inf_single_run
            attack[cnt, :] = rec_single_run + inf_single_run
            cnt += 1

            mean_infected  *= (cnt-1.) / cnt
            mean_infected  += inf_single_run / cnt
            stop_flag, error_stats = update_error(mean_infected, cnt, stop_flag, *error_stats)
            
            if cnt == ensemble:
                break
        
    return infected, attack, cnt


'''
# ================================================================================
#                               Fixed outbreak location
# ================================================================================
@jit(nopython=True, nogil=True)
def simulation_fixed_source(edgelist, directed, num_nodes, source, alpha, beta, init_prob=1, iter_max=100, iter_min=0, tol=1e-3, verbose=False):

    Tmax = edgelist[-1,0] + 2 #
    infected = np.zeros(Tmax, dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    recovered = np.zeros(Tmax, dtype=np.float64)
    
    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max, dtype=np.float64) * -1. #save the error trajectory here
    old_trajectory = np.zeros(Tmax, dtype=np.float64)
    error_stats = old_trajectory, 0, tol, deque_length, error_deque, error_list #old_value, idx, tol, deque_length, error_deque, error_list
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    stop_flag = False
    j = 1.
    for it in xrange(iter_max):
        #norm = 0
        inf_single_run, rec_single_run = simulate_single_trajectory( edgelist, directed, num_nodes, alpha, beta, source, Tmax -1)
        
        infected  *= (j-1.) / j
        infected  += inf_single_run / j
        recovered *= (j-1.) / j
        recovered += rec_single_run / j
        j += 1.
        # Check convergence
        stop_flag, error_stats = update_error(infected, it, stop_flag, *error_stats)

        if stop_flag:
            if it > iter_min:
                break
        
    return it, infected, recovered, error_list
'''


# ================================================================================
#                               fixed source and target node
# ================================================================================
#@jit(nopython=True, nogil=True)
def simulation_fixed_source_target(edgelist, directed, num_nodes, source, alpha, beta, init_prob=1, iter_max=100, iter_min=0, tol=1e-3):

    Tmax = edgelist[-1,0] + 2 #
    infected = np.zeros((num_nodes, Tmax), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    recovered = np.zeros((num_nodes, Tmax), dtype=np.float64)
    
    #deque_length = 1000 #save the relative error in a list of constant size 
    #error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    #error_list = np.ones(iter_max, dtype=, infected, recoverednp.float64) * -1. #save the error trajectory here
    #old_trajectory = np.zeros((num_nodes, Tmax), dtype=np.float64)
    #error_stats = old_trajectory, 0, tol, deque_length, error_deque, error_list #old_value, idx, tol, deque_length, error_deque, error_list
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    #stop_flag = False
    for it in tqdm( range(1, iter_max + 1) ):
        infected  *= (it-1.) / it
        recovered *= (it-1.) / it
        infected, recovered = simulate_single_target_node(edgelist, directed, num_nodes, alpha, beta, source, it, infected, recovered)
        # Check convergence
        #stop_flag, error_stats = update_error(infected, it, stop_flag, *error_stats)
        #
        #if stop_flag:
        #    if it > iter_min:
        #        break
        
    return infected, recovered#, error_list



# ================================================================================
#                   fixed source, fixed target, full distributions
# ================================================================================
#@jit(nopython=False, nogil=False, cache=False)
def fixed_source_and_target(edgelist, 
                            directed, 
                            Nnodes, 
                            outbreak_location,
                            alpha, 
                            beta,
                            Tmax=-1, iter_max=100, iter_min=0, tol=1e-3, verbose=False):

    if Tmax == -1:
        Tmax = edgelist[-1,0] + 1 #add initial state as additional time step
    else:
        Tmax = int(Tmax) + 1 #add initial state as additional time step
    
    infected = np.zeros(Tmax+1, dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack = np.zeros(Tmax+1, dtype=np.float64)
    
    infected_array = np.zeros((Nnodes, Tmax+1), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack_array = np.zeros((Nnodes, Tmax+1), dtype=np.float64)
    results = (infected,
               attack,
               infected_array,
               attack_array)

    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max, dtype=np.float64) * -1. #save the error trajectory here
    old_trajectory = np.zeros(Tmax+1, dtype=np.float64)
    error_stats = old_trajectory, 0, tol, deque_length, error_deque, error_list, len(old_trajectory) #old_value, idx, tol, deque_length, error_deque, error_list
    
    stop_flag = False
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    
    for cnt in tqdm(range(1,iter_max+1)):
        #norm = 0infected_array
        results = simulate_single_target_node( edgelist, directed, Nnodes, alpha, beta, outbreak_location, Tmax, cnt, *results)

        # Check convergence
        stop_flag, error_stats = update_error(infected, cnt, stop_flag, *error_stats)

        if stop_flag:
            if cnt > iter_min:
                break

    
    return infected, attack, infected_array, attack_array, error_list, cnt


# ================================================================================
#                               Static Network
# ================================================================================
#@jit(nopython=True, nogil=True)
def simulation_fixed_source_target_static(edgelist, directed, num_nodes, source, alpha, beta, Tmax, init_prob=1, iter_max=100, iter_min=0, tol=1e-3, verbose=False):

    infected = np.zeros((num_nodes, Tmax+1), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    recovered = np.zeros((num_nodes, Tmax+1), dtype=np.float64)
    
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    
    for it in tqdm( range(1, iter_max + 1) ):
        infected  *= (it-1.) / it
        recovered *= (it-1.) / it
        infected, recovered = simulate_single_target_node_static(edgelist, directed, num_nodes, alpha, beta, Tmax, source, it, infected, recovered)
        
    return infected, recovered#, error_list

if __name__ == "__main__":
    # ================================================================================
    #                               Init. Simulation
    # ================================================================================
    alpha = 0.5
    beta = 0.5
    edgelist = np.array([[0,1],[1,0]])
    N = 2
    directed = True
    source = 0
    target = 1
    Tmax = 10

    results = simulation_fixed_source_target_static(edgelist, 
                            directed,
                            N, 
                            source,
                            alpha,
                            beta,
                            Tmax,
                            init_prob=1.,
                            iter_max=10000,
                            iter_min=0,
                            tol=1e-3,
                            verbose = True)
    
    print(results)