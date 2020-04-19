from __future__ import print_function
import numpy as np
from numba import jit, autojit
from simulate_single_realization import simulate_single_attack, simulate_single_attack_weighted, simulate_single_attack_multiple_outbreak_locations, simulate_single_attack_multiple_outbreak_locations_weighted
from tqdm import tqdm_notebook as tqdm

# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
@jit(nopython=True)
def update(newValue, count, mean, M2):
    count += 1 
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return count, mean, M2

# retrieve the mean, variance and sample variance from an aggregate
@jit(nopython=True)
def finalize(count, mean, M2): 
    return mean, M2/count

@jit(nopython=True)
def update_error(new_values, iteration, stop_flag, old_value, idx, tol, deque_length, error_deque, error_list):
    if new_values[0] > 0 and new_values[1] > 0:
        new_error_sus = np.abs( new_values[0] - old_value[0] ) / new_values[0] #check convergence on rel. deviation of 
        new_error_att = np.abs( new_values[1] - old_value[1] ) / new_values[1] #the mean value
        new_error = max( new_error_att, new_error_sus )

        error_deque[idx] = new_error
        idx = (idx + 1) % deque_length
    

    max_error = error_deque.max()
    error_list[iteration] = max_error
    if max_error < tol:
        stop_flag = True
    
    return stop_flag, (new_values, idx, tol, deque_length, error_deque, error_list)
    


@jit(nopython=True)
def update_error_vulnerability_list(new_values, iteration, old_values, idx, tol, deque_length, error_deque, error_list):
    new_error = np.abs( new_values - old_values ).max()
    
    error_deque[idx] = new_error
    idx = (idx + 1) % deque_length
    
    max_error = error_deque.max()
    error_list[iteration] = max_error
    if max_error < tol:
        stop_flag = True
    else:
        stop_flag = False
    
    old_values[:] = new_values
    return stop_flag, (old_values, idx, tol, deque_length, error_deque, error_list)

#@jit(nopython=True)
def simulation(edgelist, 
               directed, 
               num_nodes, 
               alpha, 
               beta, 
               weighted = False,
               verbose = True,
               init_prob=1, 
               iter_max=10000, 
               iter_min=0,
               tol=1e-3,
               deque_length = 10,
               Tmax = None):
    
    if Tmax is None:
        Tmax = edgelist[-1,0] + 1 #last time step if not provided
        # if Tmax > edgelist[-1,0] + 1, then no contacts are assumed within the interval
    node_risk = np.zeros(num_nodes, dtype=np.float64) #Number of nodes attacked by a given outbreak location
    node_vulnerability = np.zeros(num_nodes, dtype=np.float64) #Number of times that a given node has been infected
    attack_array = np.zeros(num_nodes, dtype=np.float64)
    

    attack_stats = 0., 0., 0. #count, mean, squared deviation (M2)
    susceptibility_stats = 0., 0., 0.

    deque_length = deque_length #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max) * -1. #save the error trajectory here
    error_stats = (0, 0), 0, tol, deque_length, error_deque, error_list #old_values (susceptibility, attack), idx, tol, deque_length, error_deque, error_list
    
    if weighted:
        simulate = simulate_single_attack_weighted
    else:
        simulate = simulate_single_attack
    
    if verbose:
        ensemble_iter = tqdm( range(iter_max) )
    else:
        ensemble_iter = xrange(iter_max)
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    stop_flag = False
    
    for it in ensemble_iter:
        #norm = 0
        attack = 0.
        for outbreak_location in xrange(num_nodes):
            
            attack, wasInfected = simulate( edgelist, directed, num_nodes, alpha, beta, outbreak_location, Tmax)
            
            attack_array[int(attack)-1] += 1.
            node_vulnerability += wasInfected.astype(np.float64)
            node_risk[outbreak_location] += float(attack) - 1. #Minus initial node.
            
            attack_stats = update(attack, *attack_stats)
            cnt, mean, M2 = attack_stats
            if cnt > 1 and mean > 0:
                susceptibility_stats = update( np.sqrt( M2 / (cnt-1) ) / mean , *susceptibility_stats )

            #print(outbreak_location, attack / (outbreak_location + 1.) )

        
        
        # Check convergence
        stop_flag, error_stats = update_error(
            (susceptibility_stats[1], attack_stats[1] ), #check convergence on mean
            it,
            stop_flag,
            *error_stats)
            
        if stop_flag:
            if it > iter_min:
                break
        
    attack_array /= float( (it+1)  * num_nodes )
    node_vulnerability /= float( (it+1) * (num_nodes-1) )
    node_risk /= float( (it+1) * (num_nodes-1) )
    #error_list = error_stats[-1][ error_stats[-1] > 0 ]
    attack_mean, attack_var = finalize( *attack_stats )
    sus_mean, sus_var = finalize( *susceptibility_stats )

    results = (it+1, 
              attack_array, 
              node_risk, 
              node_vulnerability, 
              (attack_mean, np.sqrt(attack_var)), 
              (sus_mean, np.sqrt(sus_var)),
               error_list )
    return results


# ================================================================================
#                               Multiple Sources
# ================================================================================

#@jit(nopython=True)
def simulation_multiple_outbreak_locations(edgelist,
                                           directed,
                                           num_nodes, 
                                           alpha, 
                                           beta,
                                           verbose = True, 
                                           weighted = False, 
                                           init_prob = 0.1,
                                           iter_max = 10000, 
                                           iter_min = 0, 
                                           tol=1e-3,
                                           deque_length = 10,
                                           Tmax = None):
    
    if Tmax is None:
        Tmax = edgelist[-1,0] + 1 #last time step if not provided
        # if Tmax > edgelist[-1,0] + 1, then no contacts are assumed within the interval
    node_risk = np.zeros(num_nodes, dtype=np.float64) #Number of nodes attacked by a given outbreak location
    node_vulnerability = np.zeros(num_nodes, dtype=np.float64) #Number of times that a given node has been infected
    attack_array = np.zeros(num_nodes, dtype=np.float64)
    

    attack_stats = 0., 0., 0. #count, mean, squared deviation (M2)
    susceptibility_stats = 0., 0., 0.

    deque_length = 10 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max) * -1. #save the error trajectory here
    error_stats = (0, 0), 0, tol, deque_length, error_deque, error_list #old_values (susceptibility, attack), idx, tol, deque_length, error_deque, error_list
    
    if weighted:
        simulate = simulate_single_attack_multiple_outbreak_locations_weighted
    else:
        simulate = simulate_single_attack_multiple_outbreak_locations
    
    if verbose:
        ensemble_iter = tqdm( range(iter_max) )
    else:
        ensemble_iter = xrange(iter_max)
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    stop_flag = False
    
    #if init_prob < 1.:
    #    def set_outbreak_locations(): return np.random.random(num_nodes)  < init_prob
    #else:
    #    def set_outbreak_locations():
    #        outbreak_locations = np.zeros(num_nodes, dtype=np.bool_)
    #        outbreak_locations[ np.random.randint(0, num_nodes, init_prob) ] = True
    #        return outbreak_locations
    for it in ensemble_iter:
        #norm = 0
        outbreak_locations = np.zeros(num_nodes, dtype=np.bool_)
        #outbreak_locations[ np.random.randint(0, num_nodes, init_prob) ] = True
        outbreak_locations = np.random.random(num_nodes)  < init_prob # ALTERNATIVE

        attack, wasInfected = simulate( edgelist, directed, num_nodes, alpha, beta, outbreak_locations, Tmax)
        
        attack_stats = update(attack, *attack_stats)
        attack_array[int(attack)-1] += 1.
        node_vulnerability += wasInfected.astype(np.float64)
        node_risk[outbreak_locations] += float(attack) - 1. #Minus initial node.
        
        attack_mean, attack_var = finalize( *attack_stats )
        susceptibility = attack_var / attack_mean if attack_mean > 0. else 0.
        susceptibility_stats = update(susceptibility , *susceptibility_stats )
        
        # Check convergence
        stop_flag, error_stats = update_error(
            (susceptibility_stats[1], attack_stats[1] ),
            it,
            stop_flag,
            *error_stats)
            
        if stop_flag:
            if it > iter_min:
                break
        
    attack_array /= float(it+1)
    node_vulnerability /= float(it+1)
    node_risk /= float(it+1)
    #error_list = error_stats[-1][ error_stats[-1] > 0 ]
    attack_mean, attack_var = finalize( *attack_stats )
    sus_mean, sus_var = finalize( *susceptibility_stats )

    results = (it, 
              attack_array, 
              node_risk, 
              node_vulnerability, 
              (attack_mean, np.sqrt(attack_var)), 
              (sus_mean, np.sqrt(sus_var)),
               error_list )
    return results



#@jit(nopython=True)
def fixed_source_attack_rate(edgelist, 
                             directed, 
                             num_nodes, 
                             alpha, 
                             beta, 
                             outbreak_location, 
                             weighted = False,
                             verbose = True,
                             threshold=0., 
                             init_prob=1, 
                             iter_max=10000, 
                             iter_min=0, 
                             tol=1e-3,
                             Tmax = None):
    
    if Tmax is None:
        Tmax = edgelist[-1,0] + 1 #last time step if not provided
        # if Tmax > edgelist[-1,0] + 1, then no contacts are assumed within the interval
    node_vulnerability = np.zeros(num_nodes, dtype=np.float64) #Number of times that a given node has been infected
    attack_array = np.zeros(num_nodes, dtype=np.float64)
    
    deque_length = 100 #save the relative error in a list of constant size 
    error_deque = np.ones(deque_length, dtype=np.float64) # a new element replaces the oldest in a deque
    error_list = np.ones(iter_max) * -1. #save the error trajectory here
    old_vulnerability = np.zeros(num_nodes, dtype=np.float64)
    error_stats = old_vulnerability, 0, tol, deque_length, error_deque, error_list #old_values, idx, tol, deque_length, error_deque, error_list
    
    if weighted:
        simulate = simulate_single_attack_multiple_outbreak_locations_weighted
    else:
        simulate = simulate_single_attack_multiple_outbreak_locations
    
    if verbose:
        ensemble_iter = tqdm( range(iter_max) )
    else:
        ensemble_iter = xrange(iter_max)
    
    # ================================================================================
    #                               Run Simulation
    # ================================================================================
    
    cnt = 1
    for it in ensemble_iter:
        #norm = 0
        attack, wasInfected = simulate_single_attack( edgelist, directed, num_nodes, alpha, beta, outbreak_location, Tmax)
        
        if attack >= threshold:
            attack_array[int(attack)-1] += 1.
            node_vulnerability *= (cnt-1.) / cnt
            node_vulnerability += wasInfected.astype(np.float64) / cnt
            
            # Check convergence
            stop_flag, error_stats = update_error_vulnerability_list(
                node_vulnerability,
                cnt,
                *error_stats)
                
            if stop_flag:
                if it > iter_min:
                    print("stop flag: {}".format(stop_flag))
                    print("error : {}".format(error_stats[4]))
                    break
            cnt += 1
    
    return cnt, attack_array, node_vulnerability, error_list



if __name__ == "__main__":
    # ================================================================================
    #                               Init. Simulation
    # ================================================================================
    alpha = 1.
    beta = 1./28.
    #edgelist = np.array([[0, 0, 1, 1],[1, 1, 0, 1]])
    edgelist = np.load("/home/andreas/TU/Message_Passing/HIT/Threshold/Original/Bundesland/12/"+"edgelist_testing_tuvw.npy")
    N = 715#3570
    directed=True
    
    results = simulation(edgelist, 
                            directed,
                            N, 
                            alpha,
                            beta,
                            weighted = True,
                            verbose = False,
                            init_prob = 1,
                            iter_max = 1000,
                            iter_min = 0,
                            tol = 1e-2)
    
    results_dct = dict(
        ensemble = results[0],
        attack_rate = results[1],
        node_risk = results[2],
        node_vulnerability = results[3],
        attack_stats = results[4],
        susceptibility_stats = results[5],
        error = results[6])
    
    for key, item in results_dct.iteritems():
        print("{}: {}".format(key, item))