from __future__ import print_function
import numpy as np
from numba import jit, autojit

'''
# ================================================================================
#                               Unweighted (OLD)
# ================================================================================
@jit(nopython=True, cache=True)
def simulate_single_trajectory(edgelist, N, alpha, beta, outbreak_location, init_prob=1., max_iterations=1):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=1. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
        outbreak_time (int)     : Infection time of the initial node, i.e. the "outbreak_location".
                                  Contacts before and at(!) the initial infection time are ignored. 
                                  Outbreak_time=0 by default, i.e. one time step before the first 
                                  contact.
        single_run (bool)       : simulation either stops after all infected nodes recover
                                  assuming periodic boundary conditions or, after one full 
                                  period, i.e. if "outbreak_time" == 0 the simulation stops
                                  at the end of the edgelist.
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    Tmax = edgelist[-1,0] + 1

    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)
    
    infected  = np.zeros(Tmax+1, dtype=np.int64) #assuming that times start with 0! ?????????
    recovered = np.zeros(Tmax+1, dtype=np.int64)

    num_infected = 1
    num_recovered = 0
    
    # Find initial contact
    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    iterations = 0
    #while outbreak_time > edgelist[eIndex,0]:#ATTENTION: Depending on the definition choose ">="
    #    eIndex += 1
    t, u, v = edgelist[eIndex]
    continue_flag = True
    time = 0 #outbreak_time -> ATTENTION: ignores outbreak time
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    while num_infected > 0 and continue_flag:
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if not is_newly_infected[u]: #no secondary infections within one time step
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1

            eIndex += 1
            iteration, eIndex = divmod(eIndex, max_index)
            if iteration >= max_iterations:
                continue_flag = False
                break # continue with recovery process and then finish
            else:
                t, u, v = edgelist[eIndex]
        
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        # -----------------------------------------------------------
        # Update infection list
        #assert num_infected >= 0 and num_infected <= N, "Value error"
        infected[time+1] = num_infected
        recovered[time+1] = num_recovered

        time += 1
        #if single_run: #ATTENTION: no periodic boundary
        #    if eIndex == outbreak_time: #One full cycle completed
        #        break
    
    infected[time:Tmax+1] = num_infected
    recovered[time:Tmax+1] = num_recovered
    #assert np.all(infected <= N), "ValueError at line 107"
    #assert np.all(infected >= 0), "ValueError at line 108"
    #edgelist[:,0] -= edgelist[0,0] #reset edgelist to the original state  #ATTENTION: no periodic boundary
    wasInfected = ~ isSusceptible #indicates all infected or recovered nodes 
    wasInfected[outbreak_location] = False #exclude outbreak location (optionally)
    return num_recovered + num_infected, wasInfected, infected, recovered
'''





# ================================================================================
#                               unweighted (NEW)
# ================================================================================
@jit(nopython=True, nogil=True, cache=True)
def simulate_single_trajectory(edgelist, directed, N, alpha, beta, outbreak_location, Tmax):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
    
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    #Tmax = edgelist[-1,0] + 1 #assuming that times start with 1 or higher! t = 0 is reserved for the initially infected node ?????????
    
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)

    infected = np.zeros(Tmax+1, dtype=np.float64) # len(infected) = Tmax+1: initial condition stored in 0
    recovered = np.zeros(Tmax+1, dtype=np.float64)
    infected[0] = 1

    num_infected = 1
    num_recovered = 0
    
    # Find initial contact
    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    t, u, v = edgelist[eIndex] #edgelist: time, source, target
    T = edgelist[-1,0] + 1
    iterations = 0
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    for time in xrange(Tmax):
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if not is_newly_infected[u]: #no secondary infections within one time step
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
            
            if not directed:
                if isInfected[v] and isSusceptible[u]:
                    if not is_newly_infected[v]: #no secondary infections within one time step
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1

            add, eIndex = divmod(eIndex + 1, max_index)
            iterations += add
            t, u, v = edgelist[eIndex]
            t += iterations*T
            if eIndex == 0: #only important for static limit, ie. t == const for all eIndex
                break #steps out of the "while time == t" loop
        # -----------------------------------------------------------
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        simulate_single_attack
        # -----------------------------------------------------------
        # Update infection list
        infected[time+1] = num_infected
        recovered[time+1] = num_recovered

    return infected, recovered

# ================================================================================
#                      multiple outbreak origins, unweighted
# ================================================================================
@jit(nopython=True, nogil=True, cache=False)
def simulate_single_trajectory_multiple_outbreak_locations(edgelist, 
                                                           directed, 
                                                           N, 
                                                           alpha, 
                                                           beta, 
                                                           outbreak_locations, 
                                                           Tmax, 
                                                           cnt,
                                                           infected_mean,
                                                           attack_mean,
                                                           infected_distribution,
                                                           attack_distribution):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
    
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    #Tmax = edgelist[-1,0] + 1 #assuming that times start with 1 or higher! t = 0 is reserved for the initially infected node ?????????
    
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_locations] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_locations] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)

    num_infected = int( outbreak_locations.sum() )
    num_recovered = 0

    infected_distribution  *= (cnt-1.) / cnt
    attack_distribution *= (cnt-1.) / cnt
    infected_mean  *= (cnt-1.) / cnt
    attack_mean *= (cnt-1.) / cnt

    infected_mean[0] += float( num_infected ) / cnt
    attack_mean[0] += float( num_infected + num_recovered ) / cnt
    infected_distribution[num_infected, 0]  += 1. / cnt #type conversion: type(cntrec) = float
    attack_distribution[num_infected + num_recovered, 0]  += 1. / cnt

    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    t, u, v = edgelist[eIndex] #edgelist: time, source, target
    T = edgelist[-1,0] + 1
    iterations = 0
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    for time in xrange(Tmax):
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if not is_newly_infected[u]: #no secondary infections within one time step
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
            
            if not directed:
                if isInfected[v] and isSusceptible[u]:
                    if not is_newly_infected[v]: #no secondary infections within one time step
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1

            add, eIndex = divmod(eIndex + 1, max_index)
            iterations += add
            t, u, v = edgelist[eIndex]
            t += iterations*T
            if eIndex == 0: #only important for static limit, ie. t == const for all eIndex
                break #steps out of the "while time == t" loop
        # -----------------------------------------------------------
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        
        infected_mean[time + 1] += float( num_infected ) / cnt
        attack_mean[time + 1] += float( num_infected + num_recovered ) / cnt
        infected_distribution[num_infected, time + 1]  += 1. / cnt #type conversion: type(cntrec) = float
        attack_distribution[num_infected + num_recovered, time + 1]  += 1. / cnt

    return infected_mean, attack_mean, infected_distribution, attack_distribution




# ================================================================================
#                               unweighted (NEW)
# ================================================================================
@jit(nopython=True, nogil=True, cache=False)
def simulate_single_target_node(edgelist, directed, N, alpha, beta, outbreak_location, Tmax, cnt, infected, attack, infected_array, attack_array):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
    
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)

    infected  *= (cnt-1.) / cnt
    attack *= (cnt-1.) / cnt
    infected_array  *= (cnt-1.) / cnt
    attack_array *= (cnt-1.) / cnt
    infected[0] += 1. / cnt
    attack[0] += 1. / cnt
    infected_array[outbreak_location,0]  += 1. / cnt
    attack_array[outbreak_location,0]  += 1. / cnt

    num_infected = 1
    num_recovered = 0
    
    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    t, u, v = edgelist[eIndex] #edgelist: time, source, target
    T = edgelist[-1,0] + 1
    iterations = 0
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    for time in xrange(Tmax):
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if not is_newly_infected[u]: #no secondary infections within one time step
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
            
            if not directed:
                if isInfected[v] and isSusceptible[u]:
                    if not is_newly_infected[v]: #no secondary infections within one time step
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1

            add, eIndex = divmod(eIndex + 1, max_index)
            iterations += add
            t, u, v = edgelist[eIndex]
            t += iterations*T
            if eIndex == 0: #only important for static limit, ie. t == const for all eIndex
                break #steps out of the "while time == t" loop
        # -----------------------------------------------------------
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        


        infected[time+1]  += float(num_infected) / cnt #type conversion: type(cntrec) = float
        attack[time+1] += float( num_infected + num_recovered ) / cnt
        infected_array[:, time+1]  += isInfected.astype( np.float64 ) / cnt #type conversion: type(cntrec) = float
        attack_array[:, time+1] += ( isRecovered.astype( np.float64 ) + isInfected.astype( np.float64) ) / cnt

    return infected, attack, infected_array, attack_array


# ================================================================================
#                               unweighted (NEW)
# ================================================================================
@jit(nopython=True, nogil=True, cache=True)
def simulate_single_target_node_static(edgelist, directed, N, alpha, beta, Tmax, outbreak_location, it, infected, recovered):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
    
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    edgelist = edgelist.astype(np.int64)
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)

    #infected = np.zeros((N, Tmax+1), dtype=np.float64) # len(infected) = Tmax+1: initial condition stored in 0
    #recovered = np.zeros((N, Tmax+1), dtype=1. * np.float64)
    infected[outbreak_location, 0] += 1./it
    num_infected = 1
    num_susceptible = N-1
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    max_index = len(edgelist)
    for time in xrange(Tmax):
        # -----------------------------------------------------------
        # Infection transmission
        for e in xrange(max_index):
            u, v = edgelist[e]
            if isInfected[u] and isSusceptible[v]:
                if not is_newly_infected[u]: #no secondary infections within one time step
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
                        num_susceptible -= 1
            
            if not directed:
                if isInfected[v] and isSusceptible[u]:
                    if not is_newly_infected[v]: #no secondary infections within one time step
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1
                            num_susceptible -= 1
                            
        # -----------------------------------------------------------
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
        
        # -----------------------------------------------------------
        # Update infection list
        infected[:,time+1] += 1. * isInfected / it
        recovered[:,time+1] += 1. * isRecovered / it

        if num_infected == 0:
            break

    for t in xrange(time+2, Tmax):
        infected[:, t] = 1. * isInfected / it
        recovered[:, t] = 1. * isRecovered / it
    return infected, recovered




# ================================================================================
#                               attack rate only (unweighted)
# ================================================================================
@jit(nopython=True, nogil=True, cache=True)
def simulate_single_attack(edgelist, directed, N, alpha, beta, outbreak_location, Tmax):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
        outbreak_time (int)     : Infection time of the initial node, i.e. the "outbreak_location".
                                  Contacts before and at(!) the initial infection time are ignored. 
                                  Outbreak_time=0 by default, i.e. one time step before the first 
                                  contact.
        single_run (bool)       : simulation either stops after all infected nodes recover
                                  assuming periodic boundary conditions or, after one full 
                                  period, i.e. if "outbreak_time" == 0 the simulation stops
                                  at the end of the edgelist.
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    #Tmax = edgelist[-1,0] + 1 #assuming that times start with 0!!!!
    MAX_ITERATIONS = 1000
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)

    num_infected = 1
    num_recovered = 0
    num_susceptible = N - 1
    
    # Find initial contact
    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    t, u, v = edgelist[eIndex] #edgelist: time, source, target
    time = 0 #outbreak_time -> ATTENTION: ignores outbreak time
    #run = 0
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    it = 0
    while num_infected > 0 and num_susceptible > 0 and it < MAX_ITERATIONS:
        ####
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if not is_newly_infected[u]: #no secondary infections within one time step
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
                        num_susceptible -= 1
            
            if not directed:
                if isInfected[v] and isSusceptible[u]:
                    if not is_newly_infected[v]: #no secondary infections within one time step
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1

            eIndex = (eIndex + 1) % max_index
            t, u, v = edgelist[eIndex]
            if eIndex == 0: #only important for static limit, ie. t == const for all eIndex
                it += 1
                break
        
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        time = (time + 1) % Tmax
    
    wasInfected = ~ isSusceptible #indicates all infected or recovered nodes 
    wasInfected[outbreak_location] = False #exclude outbreak location (optionally)
    return num_recovered + num_infected, wasInfected


# ================================================================================
#                               attack rate only, !WEIGHTED!
# ================================================================================
@jit(nopython=True, nogil=True, cache=True)
def simulate_single_attack_weighted(edgelist, directed, N, alpha, beta, outbreak_location, Tmax):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target, weight
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
        outbreak_time (int)     : Infection time of the initial node, i.e. the "outbreak_location".
                                  Contacts before and at(!) the initial infection time are ignored. 
                                  Outbreak_time=0 by default, i.e. one time step before the first 
                                  contact.
        single_run (bool)       : simulation either stops after all infected nodes recover
                                  assuming periodic boundary conditions or, after one full 
                                  period, i.e. if "outbreak_time" == 0 the simulation stops
                                  at the end of the edgelist.
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    MAX_ITERATIONS = 1000
    #Tmax = edgelist[-1,0] + 1 #assuming zero indexing for time!!!!
    
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)

    num_infected = 1
    num_recovered = 0
    num_susceptible = N - 1
    
    # Find initial contact
    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    t, u, v, w = edgelist[eIndex] #edgelist: time, source, target
    time = 0 #outbreak_time -> ATTENTION: ignores outbreak time
    #run = 0
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    it = 0
    while num_infected > 0 and num_susceptible > 0 and it < MAX_ITERATIONS:
        ####
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and not is_newly_infected[u] and isSusceptible[v]:
                for cnt in xrange(w):
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
                        num_susceptible -= 1
                        break
            
            if not directed:
                if isInfected[v] and not is_newly_infected[v] and isSusceptible[u]:
                    for cnt in xrange(w):
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1
                            break

            eIndex = (eIndex + 1) % max_index
            t, u, v, w = edgelist[eIndex]
            if eIndex == 0: #only important for static limit, ie. t == const for all eIndex
                it += 1
                break
        
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        time = (time + 1) % Tmax
    
    assert it < MAX_ITERATIONS, "maximum number of iteration exceeded at outbreak location."# Num. infectiond: {}. Num. susceptible: {}".format(outbreak_location, num_infected, num_susceptible)
    wasInfected = ~ isSusceptible #indicates all infected or recovered nodes 
    wasInfected[outbreak_location] = False #exclude outbreak location (optionally)
    return num_recovered + num_infected, wasInfected

# ================================================================================
#                               attack rate only, multiple sources
# ================================================================================
@jit(nopython=True, nogil=True, cache=False)
def simulate_single_attack_multiple_outbreak_locations(edgelist, directed, N, alpha, beta, outbreak_locations, Tmax):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
        outbreak_time (int)     : Infection time of the initial node, i.e. the "outbreak_location".
                                  Contacts before and at(!) the initial infection time are ignored. 
                                  Outbreak_time=0 by default, i.e. one time step before the first 
                                  contact.
        single_run (bool)       : simulation either stops after all infected nodes recover
                                  assuming periodic boundary conditions or, after one full 
                                  period, i.e. if "outbreak_time" == 0 the simulation stops
                                  at the end of the edgelist.
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    #Tmax = edgelist[-1,0] + 1 #assuming that times start with 0!!!!
    
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_locations] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_locations] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)
    
    num_infected = int( outbreak_locations.sum() )
    num_recovered = 0
    num_susceptible = N - num_infected

    # Find initial contact
    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    t, u, v = edgelist[eIndex] #edgelist: time, source, target
    time = 0 #outbreak_time -> ATTENTION: ignores outbreak time
    #run = 0
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    
    while num_infected > 0 and num_susceptible > 0:
        ####
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if not is_newly_infected[u]: #no secondary infections within one time step
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
                        num_susceptible -= 1
            
            if not directed:
                if isInfected[v] and isSusceptible[u]:
                    if not is_newly_infected[v]: #no secondary infections within one time step
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1

            eIndex = (eIndex + 1) % max_index
            t, u, v = edgelist[eIndex]
            if eIndex == 0: #only important for static limit, ie. t == const for all eIndex
                break
        
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        time = (time + 1) % Tmax
    
    wasInfected = ~ isSusceptible #indicates all infected or recovered nodes 
    wasInfected[outbreak_locations] = False #exclude outbreak location (optionally)
    return num_recovered + num_infected, wasInfected

# ================================================================================
#                   attack rate only, multiple sources, !WEIGHTED!
# ================================================================================
@jit(nopython=True, nogil=True, cache=False)
def simulate_single_attack_multiple_outbreak_locations_weighted(edgelist, directed, N, alpha, beta, outbreak_locations, Tmax):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target, weight
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=0. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
        outbreak_time (int)     : Infection time of the initial node, i.e. the "outbreak_location".
                                  Contacts before and at(!) the initial infection time are ignored. 
                                  Outbreak_time=0 by default, i.e. one time step before the first 
                                  contact.
        single_run (bool)       : simulation either stops after all infected nodes recover
                                  assuming periodic boundary conditions or, after one full 
                                  period, i.e. if "outbreak_time" == 0 the simulation stops
                                  at the end of the edgelist.
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    #Tmax = edgelist[-1,0] + 1 #assuming that times start with 0!!!!
    
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_locations] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_locations] = False
    is_newly_infected = np.zeros(N, dtype=np.bool_)
    
    num_infected = int( outbreak_locations.sum() )
    num_recovered = 0
    num_susceptible = N - num_infected

    # Find initial contact
    eIndex = 0 # edgelist index
    max_index = len(edgelist)
    t, u, v, w = edgelist[eIndex] #edgelist: time, source, target
    time = 0 #outbreak_time -> ATTENTION: ignores outbreak time
    #run = 0
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    
    while num_infected > 0 and num_susceptible > 0:
        ####
        # -----------------------------------------------------------
        # Infection transmission
        while time == t:
            if isInfected[u] and not is_newly_infected[u] and isSusceptible[v]:
                for cnt in xrange(w):
                    if np.random.random() < alpha:
                        isSusceptible[v] = False
                        isInfected[v] = True
                        is_newly_infected[v] = True # v can only appear once in the list
                        num_infected += 1
                        num_susceptible -= 1
                        break
            
            if not directed:
                if isInfected[v] and not is_newly_infected[v] and isSusceptible[u]:
                    for cnt in xrange(w):
                        if np.random.random() < alpha:
                            isSusceptible[u] = False
                            isInfected[u] = True
                            is_newly_infected[u] = True # v can only appear once in the list
                            num_infected += 1
                            break

            eIndex = (eIndex + 1) % max_index
            t, u, v, w = edgelist[eIndex]
            if eIndex == 0: #only important for static limit, ie. t == const for all eIndex
                break
        
        # Recovery (applies only to previously infected nodes)
        for n in xrange(N): #make a copy and modify the original inside the loop
            if is_newly_infected[n]:
                is_newly_infected[n] = False
            else:
                if isInfected[n]:
                    if np.random.random() < beta:
                        isInfected[n] = False
                        isRecovered[n] = True
                        num_infected -= 1
                        num_recovered += 1
        time = (time + 1) % Tmax
    
    wasInfected = ~ isSusceptible #indicates all infected or recovered nodes 
    wasInfected[outbreak_locations] = False #exclude outbreak location (optionally)
    return num_recovered + num_infected, wasInfected

'''
# ================================================================================
#                               Weighted (BEWARE OF ERRORS)
# ================================================================================
#@jit(nopython=True)
def simulate_single_trajectory_weighted(edgelist, N, alpha, beta, outbreak_location, outbreak_time=0, num_iterations=1):
    """
    Monte-Carlo simulation of a single outbreak
    -----------------------------
    Input:
        edgelist (numpy.array)  : Three columns contact data -> time, source, target
                                  The edge-list describes a directed temporal network
                                  Node IDs must run contiguously from 0 to N-1.
                                  Time starts at t=1. 
                                  All data types are assumed to be integers.
        N (int)                 : Number of nodes
        alpha (float)           : infection transmission probability per contact.
        beta (float)            : recovery probability per time step
        outbreak_location (int) : initially infected node
        outbreak_time (int)     : Infection time of the initial node, i.e. the "outbreak_location".
                                  Contacts before and at(!) the initial infection time are ignored. 
                                  Outbreak_time=0 by default, i.e. one time step before the first 
                                  contact.
        single_run (bool)       : simulation either stops after all infected nodes recover
                                  assuming periodic boundary conditions or, after one full 
                                  period, i.e. if "outbreak_time" == 0 the simulation stops
                                  at the end of the edgelist.
    Output:
        arrival_times (numpy.array) : Array of size N with integer values according to the 
                                      infection time. Intuitively, we find 
                                      arrival_times[outbreak_location] = outbreak_time
                                      Nodes that have not been infected are set to -1.
                                  
    """
    # Initialize simulation
    isInfected = np.zeros(N, dtype=np.bool_)
    isInfected[outbreak_location] = True
    isRecovered = np.zeros(N, dtype=np.bool_)
    isSusceptible = np.ones(N, dtype=np.bool_)
    isSusceptible[outbreak_location] = False

    num_infected = 1
    num_recovered = 0
    num_contacts = len(edgelist)
    Tmax = edgelist[-1,0] + 1

    infected  = np.zeros(Tmax+1, dtype=np.int64) #assuming that times start with 0! ?????????
    recovered = np.zeros(Tmax+1, dtype=np.int64)
    # Find initial contact
    eIndex = 0 # edgelist index
    while outbreak_time > edgelist[eIndex,0]:
        eIndex += 1
    t, u, v, w = edgelist[eIndex]
    time = outbreak_time
    # ---------------------------------------------------------------
    # start Monte-Carlo simulation
    while num_infected > 0:
        # -----------------
        # Infection transmission
        while time == t:
            if isInfected[u] and isSusceptible[v]:
                if np.random.random() < 1. - (1. - alpha)**w:
                    isInfected[v] = True
                    isSusceptible[v] = False
                    num_infected += 1
                    infected[t+1:Tmax+1] += 1 #SHIFT by one accounts for the initial condition
            eIndex += 1
            if eIndex == num_contacts:
                eIndex %= num_contacts # Periodic boundary condition
                edgelist[:,0] += Tmax # Assuming t = 0, ..., Tmax
            t, u, v, w = edgelist[eIndex]
        # -----------------
        # Recovery
        for n in xrange(N):
            if isInfected[n]:
                if np.random.random() < beta:
                    isInfected[n] = False
                    isRecovered[n] = True
                    num_infected -= 1
                    num_recovered += 1
                    infected[t+1:Tmax+1] -= 1 #SHIFT by one accounts for the initial condition
                    recovered[t+1:Tmax+1] += 1 #SHIFT by one accounts for the initial condition
        time += 1
        if single_run:
            if eIndex == outbreak_time: #One full cycle completed
                break

    edgelist[:,0] -= edgelist[0,0] #reset edgelist to the original state
    wasInfected = ~ isSusceptible #indicates all infected or recovered nodes
    wasInfected[outbreak_location] = False #exclude outbreak location (optionally)
    return num_recovered + num_infected, wasInfected, infected, recovered
'''

if __name__ == "__main__":
    edgelist = np.array([[0,0,1,2], [1,0,1,2]])
    N = 2
    alpha = 0.1
    beta = 0.
    outbreak_location = 0
    directed = True
    Tmax = 10
    infected_array = np.zeros((N, Tmax+2), dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack_array = np.zeros((N, Tmax+2), dtype=np.float64)
    infected = np.zeros(Tmax+2, dtype=np.float64) # len(infected) = max(times)+1: initial condition stored in 0
    attack = np.zeros(Tmax+2, dtype=np.float64)
    it = 1
    results = simulate_single_attack_weighted(edgelist, directed, N, alpha, beta, outbreak_location, Tmax)
    #results = simulate_single_target_node(edgelist, directed, N, alpha, beta, outbreak_location, Tmax, it, infected, attack, infected_array, attack_array)

    print(results)
