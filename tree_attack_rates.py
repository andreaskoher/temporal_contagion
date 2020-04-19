from __future__ import print_function
import numpy as np
from tqdm import tqdm
from simulation_SIR_attack_rate import simulation_SIR
import os
import shutil


# ================================================================================
#                               Init. Simulation
# ================================================================================
directory = "/home/andreasko/Schreibtisch/Simulation/Tree/0101/"
fname = "/home/andreasko/Schreibtisch/Data/Tree/tree_edgelist_tuv_directed_large.npy"
alphas            = np.linspace(0.04,1,25)[::-1]
beta              = 0.5
init_prob         = 1.
outbreak_location = -1
outbreak_time     = 0
single_run        = True
store             = 10
iter_max          = 100000
rel_tol           = 1e-4


edgelist = np.load(fname)
nodes = set(edgelist[:,1]) | set(edgelist[:,2])
N = len(nodes)
T = np.max(edgelist[:,0]) + 1
assert nodes == set(xrange(N))
del nodes
print("Number of nodes:", N)
print("Time runs from", edgelist[0,0], "to", edgelist[-1,0], "in", len(np.unique(edgelist[:,0])), "time steps")

if outbreak_time == -1:
    outbreak_times = range(T) 
else:
    outbreak_times =  [outbreak_time]
if outbreak_location == -1:
    outbreak_locations = range(N)
else:
    outbreak_locations =  [outbreak_location]

# ================================================================================
#                               Prepare directory
# ================================================================================
if not os.path.exists(directory):
    os.makedirs(directory)
shutil.copy2(fname, directory) # complete target filename given
shutil.copy2("/home/andreasko/Schreibtisch/Simulation/simulation_SIR_attack_rate.py", directory) # complete target filename given
shutil.copy2("/home/andreasko/Schreibtisch/Simulation/tree_attack_rates.py", directory)


with open(directory + "parameters.csv", "wb") as fb:
    fb.write("beta,{}\n".format(beta))
    fb.write("outbreak_location,{}\n".format(outbreak_location))
    fb.write("outbreak_time,{}\n".format(outbreak_time))
    fb.write("single_run,{}\n".format(single_run))
    fb.write("store,{}\n".format(store))
    fb.write("iter_max,{}\n".format(iter_max))
    fb.write("rel_tol,{}\n".format(rel_tol))
    fb.write("init_prob,{}\n".format(init_prob))
    fb.write("fname,{}\n".format(fname))

with open(directory + "attack_distribution.csv", "wb") as fb:
    fb.write("Alpha")
    for node in xrange(N):
        fb.write(",Node_{}".format(node))
    fb.write("\n")
with open(directory + "node_risk.csv", "wb") as fb:
    fb.write("Alpha")
    for node in xrange(N):
        fb.write(",Node_{}".format(node))
    fb.write("\n")
with open(directory + "node_vulnerability.csv", "wb") as fb:
    fb.write("Alpha")
    for node in xrange(N):
        fb.write(",Node_{}".format(node))
    fb.write("\n")
with open(directory + "error.csv", "wb") as fb:
    fb.write("Alpha")
    for node in xrange(N):
        fb.write(",Node_{}".format(node))
    fb.write("\n")

for ii, alpha in enumerate(alphas):
    
    results = simulation_SIR(edgelist, 
                            T,
                            N,
                            alpha, 
                            beta,
                            outbreak_locations,
                            outbreak_times,
                            init_prob=init_prob,
                            single_run=single_run,
                            store=store,
                            iter_max=iter_max,
                            rel_tol=rel_tol)
    

    attack_rate = results[0]
    node_risk = results[1]
    node_vulnerability = results[2]
    error = results[3]

    with open(directory + "attack_distribution.csv", "ab") as fb:
        fb.write("{}".format(alpha))
        for node in xrange(N):
            fb.write(",{}".format(attack_rate[node,0])) #Achtung: speichert nur eine Ausbruchszeit!
        fb.write("\n")
    with open(directory + "node_risk.csv", "ab") as fb:
        fb.write("{}".format(alpha))
        for node in xrange(N):
            fb.write(",{}".format(node_risk[node,0]))
        fb.write("\n")
    with open(directory + "node_vulnerability.csv", "ab") as fb:
        fb.write("{}".format(alpha))
        for node in xrange(N):
            fb.write(",{}".format(node_vulnerability[node,0]))
        fb.write("\n")
    with open(directory + "error.csv", "ab") as fb:
        fb.write("{}".format(alpha))
        for node in xrange(N):
            fb.write(",{}".format(error[node,0]))
        fb.write("\n")
    
    print("-----------------------------------------")
    print("alpha:", alpha),"Nr.",ii,"out of", len(alphas)
    print("attack:", np.dot(np.arange(1,N+1)[np.newaxis,:],attack_rate)[0,0])
    print("error", np.mean(error))