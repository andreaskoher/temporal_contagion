# ===================== Load Packages =================
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx
import scipy.sparse as sp
'''
outbreak_location = int(sys.argv[1])
alpha = float(sys.argv[2])
beta = float(sys.argv[3])
ensemble = int(sys.argv[4])
directory = str(sys.argv[5])

print len(sys.argv), "arguments found in", sys.argv[0]
print "Outbreak location:", outbreak_location, type(outbreak_location)
print "alpha:", alpha, type(alpha)
print "beta:", beta, type(beta)
print "Ensemble size:", ensemble, type(ensemble)
fname = directory + "simulation_SIR_ht09_ensemble%i_alpha%1.2f_beta%1.6f_outbreak%i" %(ensemble, alpha, beta, outbreak_location)
print "Save results at:", fname
'''

# ============= Load Data ===========================
class Edgelist:
    def __init__(self):
        self.accessibility_matrix = None
        self.adjacency_matrix_list = None
        self.new_to_old_dict = None
        self.old_to_new_dict = None

    # ============================================================================
    def load(self, 
             edgelist,
             directed,
             source,
             target,
             time,
             weight=None,
             weighted = None,
             t0=0,
             aggregate=None,
             period=None,
             verbose=True,
             strongly_connected=False,
             time_to_index=False,
             padding = False,
             **kwargs):

        """
        Load an edgelist from a file (str), a Numpy array, list or take a networkx graph. 
        The edgelist is supposed to have two, three or four columns: source, target, time, weight.
        If only two columns are provided, we assume a uniform edge weight. Node IDs have to run continuously from 0 to Number_of_nodes-1 and the final graph is saved internally in self.graph.
        Parameters
        ----------
            graph : str, numpy.array, list or networkx.DiGraph / networkx.Graph
                If type(graph) == str, assume a path to the edgelist file readable 
                by numpy.genfromtxt. Additional keywords such as "dtype", "delimiter" and "comments"
                can be past to the function.
                If type(graph) == numpy.array, assume a two, three or four column edgelist
                If type(graph) == list, assume an list of tuples, convertible via np.array(graph)
                weights are stored as edge attributes "weight".
            
            verbose : bool
                Print information about the data. True by Default
            
            directed : bool
                Interprete the edgelist accordingly (True by default). If False, then reciprical edges are added.
                In any case the resulting graph is directed.  

            strongly_connected : bool
                If True (default), then only the giant strongly connected component will be considered.
                
            **kwargs
                Default parameters are passed to numpy.genfromtxt. Some of the supported keywords are
                    dtype     : float (default)
                    delimiter : ","   (default)
                    comments  : "#"   (default)
        """
        self.directed = directed
        self.weighted = weighted
        
        # --------------- accept file name ---------------------------------
        if isinstance(edgelist, str): #expect path to file with edgelist
            edgelist = np.genfromtxt(edgelist, unpack=False, **kwargs)
            if verbose:
                print "file successfully loaded..."

        # --------------- accept numpy.ndarray ---------------------------------
        if isinstance(edgelist, np.ndarray) or isinstance(edgelist, list):
            edgelist_array = np.array(edgelist)
            edgelist = pd.DataFrame(data={"source" : edgelist_array[:,source],
                                          "target" : edgelist_array[:,target],
                                          "time"   : edgelist_array[:,time]})
            source, target, time = "source", "target", "time"

            if weight is not None:
                edgelist["weight"] = edgelist_array[:,weight]
                weight = "weight"
                self.weighted = True
            del edgelist_array

        # --------------- accept pandas.DataFrame ---------------------------------
        if isinstance(edgelist, pd.DataFrame):
            self.df = edgelist.copy() #local copy avoids unexpected changes to the original dataset
            del edgelist
            self.df.rename({source : "source", target : "target", time : "time"}, axis=1, inplace=True)
            
            if verbose:
                dtype = type(self.df.source[0])
                if not isinstance(self.df.source[0], int) or not isinstance(self.df.target[0], int):
                    print "dtype of time is {}. convert dtype to integer...".format(dtype)
            self.df.source = self.df.source.astype(int) #assuming integer values!
            self.df.target = self.df.target.astype(int) #assuming integer values!
            
            # =============================================
            #               configure weights
            # =============================================
            if weight is None:
                if self.weighted is None:
                    self.weighted = False
                self.df["weight"] = 1.
                weight = "weight"
                if verbose:
                    "unit weights added in column 'weight' ..."
            else:
                self.weighted = True
                self.df.rename({weight : "weight"}, axis=1, inplace=True)
                self.df.weight = self.df.weight.astype(float)
            self.df = self.df[["source", "target", "weight", "time"]]
            
            # =============================================
            #               configure time
            # =============================================
            if period is not None:
                assert len(period) == 2, "set start and end time as (start, end) with 'start' included and 'end' excluded"
                self.df = self.df[self.df.time < period[1]]
                self.df = self.df[self.df.time >= period[0]].reset_index( drop = True )
                self.period = (period[0], period[1])
            else:
                self.period = (self.df.time.min(), self.df.time.max() + 1)

            if aggregate is not None:
                self.aggregate_edgelist(aggregate, padding=padding, verbose=verbose)

            self.set_time_to_index( t0, verbose )

            # =============================================
            #               update topology
            # =============================================
            self.remove_self_loops(verbose)            

            self.remove_parallel_edges(verbose)

            if strongly_connected:
                self.static_strongly_connected_graph( verbose )
            #elif strongly_connected == "temporal":
            #    self.temporal_strongly_connected_graph( verbose )
            #else:
            #    assert strongly_connected == False, "choose 'strongly_connected' as 'static' or 'temporal' not {}".format(strongly_connected)
            self.relabel_nodes(verbose)
            
            self.set_time_to_index( t0, verbose ) # time attributes may have changed


            # =============================================
            #               update attributes
            # =============================================
            self.nodes = set(self.df.source) | set(self.df.target)
            self.Nnodes = len(self.nodes)
            assert self.nodes == set( xrange(self.Nnodes) )

            self.Ncontacts = len(self.df)
            self.Nedges = len( self.df[['weight','source','target']].groupby(['source','target']).sum().reset_index( drop = False ) )
            self.times = self.df.time.unique()
            self.period = (self.times[0], self.times[-1] + 1)
            
            self.df.source = self.df.source.astype(int)
            self.df.target = self.df.target.astype(int)
            self.df.time = self.df.time.astype(int)
            self.df.weight = self.df.weight.astype(float)
            self.df = self.df[["source", "target", "weight", "time"]]

            if verbose:
                print "access edgelist in pandas.DataFrame self.df with following keys: source, target, time, weight"
                print "edgelist has been as a pandas.DataFrame in self.df with column names 'source', 'target', 'time', 'weight'"
                print "number of nodes, (temporal) contacts and (static) edges:", self.Nnodes, self.Ncontacts, self.Nedges
                print "time runs from {} to {} in {} time steps".format(self.times[0], self.times[-1], len(self.times))
                print "weighted:", self.weighted, ", average edge weight:", self.df.weight.mean()
        else:
            raise ValueError("Expected type of graph is either str, numpy.ndarray or pandas.DataFrame. Instead got {}".format(type(edgelist)))
    

    def set_time_to_index(self, t0, verbose):
        assert self.df.time.is_monotonic_increasing

        if verbose:
            if self.df.time[0] != 0:
                print "change initial time from {} to 0".format(self.df.time[0])
        self.df.time -= self.df.time[0]
        self.times = self.df.time.unique()
        if len(self.times) > 1: #otherwise we have a fully aggregated network with len(times) == 1
            diff = np.diff(self.times).min() #unique and sorted values
            if diff != 1. :
                self.df.time /= float(diff) #assume constant time difference between slices
                self.df.time = self.df.time.astype(int) #assuming integer values!
                self.times = self.df.time.unique()
                if verbose:
                    print "change time steps from {} to {}".format(diff, self.times[1] - self.times[0])
            diff = np.diff(self.times).min()
            assert diff == 1, "time index has to increase in multiples of one. Instead got {}".format(diff)
            assert ( ( self.times % diff ) == 0).all(), "non-uniform time index: time has to increase in multiples of the minimum time step"
        
        self.df.time = self.df.time.astype(int) #assuming integer values!
        self.df.time += t0
        self.times = self.df.time.unique()
        self.Ncontacts = len(self.df)

    def remove_self_loops(self, verbose):
        selfloops = self.df.source == self.df.target
        self.df = self.df[~ selfloops].reset_index( drop = True )
        if verbose:
            if selfloops.any():
                print "{} self-loops removed...".format(selfloops.sum())


    def remove_parallel_edges(self, verbose):
        # sort source and target index for undirected networks
        if not self.directed:
            switch = self.df.source > self.df.target
            self.df.loc[switch, ("source", "target")] = self.df.loc[switch, ["target", "source"]].values

        Ncontacts_before = len(self.df)
        self.df = self.df.groupby(['time','source','target']).sum().reset_index( drop = False ) 
        if not self.weighted:
            self.df.weight = 1.
        self.Ncontacts = len(self.df)

        if verbose:
            if Ncontacts_before > self.Ncontacts:
                print "remove parallel edges: {} before, {} after".format(Ncontacts_before, self.Ncontacts)
        self.Nedges = len( self.df[['weight','source','target']].groupby(['source','target']).sum().reset_index( drop = False ) )
        self.df.source = self.df.source.astype(int) #assuming integer values!
        self.df.target = self.df.target.astype(int) #assuming integer values!
            

    def relabel_nodes(self, verbose):
        nodes  = set(self.df.source) | set(self.df.target)
        if set(xrange(len(nodes))) != nodes:
            old_to_new_dict = {old: new for new, old in enumerate(nodes)}
            self.df["source"] = self.df.source.map(old_to_new_dict)
            self.df["target"] = self.df.target.map(old_to_new_dict)
            if self.new_to_old_dict is None: #relabel the first time
                self.new_to_old_dict = {new: old for old, new in old_to_new_dict.iteritems()}
            else: #update previously existing dict
                self.new_to_old_dict = {new : self.new_to_old_dict[old] for new, old in old_to_new_dict.iteritems()}
            if verbose:
                print "\nThe node IDs have to run continuously from 0 to Number_of_nodes-1."
                print "Node IDs have been changed according to the requirement."
                print "A dict 'new ID' -> 'old ID' is saved in self.new_to_old_dict \n-----------------------------------\n"
        self.nodes = set(self.df.source) | set(self.df.target)
        self.Nnodes = len(self.nodes)

    def aggregate_edgelist(self, dt, padding=False, verbose=False):
        """
        Aggregate in time each dt time steps
        Args:
            dt(int) : Time interval used to aggregate. dt = 0 specifies full aggregation.
            store   : if true stores aggregated movements in self.aggregations_t, otherwise returns it
        Returns:
            movements (DataFrame) : if store = False the updated movements are returned, otherwise None.
        """
        assert isinstance(dt, int), "aggregation interval must be of type int, got {}".format(type(dt))
        t0 = 0
        self.set_time_to_index( t0, verbose )
        T = self.df.time.max() + 1

        if dt == 0:
            self.df.time = 0
        else:
            self.df['time'] //= dt
        self.df = self.df.groupby(['time','source','target']).sum().reset_index( drop = False )
        # Check if T is divisible by dt. If not, drop last interval.
        if dt>0:
            if T%dt != 0:
                self.df = self.df[self.df.time != self.df.time.max()]
        
        if padding:
            self.add_padding(dt)
        self.Ncontacts = len(self.df)
        self.times = self.df.time.unique()


    def add_padding(self, dt):
        edgelist = self.df[["time", "source", "target", "weight"]].values
        edgelist_padded = np.zeros((edgelist.shape[0] * dt, edgelist.shape[1]), dtype=float)
        start_idx = 0
        end_idx = 0
        for time in self.times:
            snapshot = edgelist[edgelist[:,0] == time]
            Nsnapshot = len(snapshot)
            time_step = time * dt - 1
            for time_idx in xrange(dt):
                time_step += 1
                start_idx = end_idx
                end_idx = start_idx + Nsnapshot 
                edgelist_padded[start_idx : end_idx, 1::] = snapshot[:,1::]
                edgelist_padded[start_idx : end_idx, 0] = time_step
        self.df = pd.DataFrame(edgelist_padded, columns=["time", "source", "target", "weight"])
        self.df.source = self.df.source.astype(int) #assuming integer values!
        self.df.target = self.df.target.astype(int) #assuming integer values!
        self.df.time = self.df.time.astype(int) #assuming integer values!
        self.times = self.df.time.unique()
        self.period = (self.times[0], self.times[-1] + 1)

        
    def get_activity(self):
        self.df["activity"] = 1.
        activity_df = self.df[["time", "activity"]].groupby("time").sum().reset_index( drop = False )
        self.df.drop("activity", axis=1, inplace=True)
        return activity_df
    

    def static_strongly_connected_graph(self, verbose=False):
        if not self.directed:
            graph = nx.Graph()
            graph.add_edges_from( self.df[["source", "target"]].values )
            Nnodes_original = graph.number_of_nodes()
            nodes = max(nx.connected_components(graph), key=lambda x: len(x))
        else:
            graph = nx.DiGraph()
            graph.add_edges_from( self.df[["source", "target"]].values )
            Nnodes_original = graph.number_of_nodes()
            nodes =  max(nx.strongly_connected_components(graph), key=lambda x: len(x))
        self.df = self.df[self.df.source.isin(nodes) & self.df.target.isin(nodes)].reset_index( drop = True )
        if verbose:
            print "Original data: {} nodes, GSCC: {} nodes".format(Nnodes_original, len(nodes))
        
        self.relabel_nodes(verbose) # updates self.nodes and self.Nnodes
        self.Nedges = len( self.df[['weight','source','target']].reset_index( drop = True )(['source','target']).sum() )
        self.Ncontacts = len(self.df)
        self.times = self.df.time.unique()
    
    def temporal_strongly_connected_graph(self, verbose=False):
        self.static_strongly_connected_graph(verbose=verbose)
        accessibility_matrix = self.get_accessibility_matrix(verbose=verbose).todok()
        accessibility_graph = nx.DiGraph()
        for (u,v), w in accessibility_matrix.items():
            accessibility_graph.add_edge(v,u) # row -> target and col -> source convention
        accessibility_graph = accessibility_graph.to_undirected(reciprocal=True, as_view=False)
        graphview = accessibility_graph.subgraph( max(nx.connected_components(accessibility_graph)) )
        nodes = max(nx.clique.find_cliques(graphview))
        self.df = self.df[self.df.source.isin(nodes) & self.df.target.isin(nodes)]
        if verbose:
            print "Original data: {} nodes, temporal GSCC: {} nodes".format(self.Nnodes, len(nodes))
        
        self.relabel_nodes(verbose) # updates self.nodes and self.Nnodes
        self.Nedges = len( self.df[['weight','source','target']].groupby(['source','target']).sum() )
        self.Ncontacts = len(self.df)
        self.times = self.df.time.unique()
    
    def get_accessibility_matrix(self, save=False, verbose=False):
        if self.accessibility_matrix is not None:
            return self.accessibility_matrix
        else:
            accessibility_matrix = sp.eye(self.Nnodes, dtype=np.int32, format="csr")
            times = np.hstack((self.times,self.times)) #go through contacts twice to consider periodic boundary condition
            if verbose:
                times_iter = tqdm(times)
            else:
                times_iter = times
            for time in times_iter:
                accessibility_matrix = accessibility_matrix + self.get_adjacency_matrix(time, dtype=np.int32) * accessibility_matrix
                accessibility_matrix.data = np.ones_like(accessibility_matrix.data, dtype=np.int32)
            if save:
                self.accessibility_matrix = accessibility_matrix
            return accessibility_matrix

    def get_adjacency_dct(self, which="out", save=False):
        assert which == "out" or which == "in"

        #adjacency_dct = {t: {n: [] for n in self.nodes} for t in self.times}
        adjacency_dct = {t: {} for t in self.times}
        for t in self.times:
            snapshot = self.df[self.df.time == t]
            for u,v in snapshot[["source", "target"]].values:
                if self.directed:
                    if which == "out":
                        if u in adjacency_dct[t]:
                            adjacency_dct[t][u].append(v)
                        else:
                            adjacency_dct[t][u] = [v]
                    else:
                        if v in adjacency_dct[t]:
                            adjacency_dct[t][v].append(u)
                        else:
                            adjacency_dct[t][v] = [u]
                else:
                    if u in adjacency_dct[t]:
                        adjacency_dct[t][u].append(v)
                    else:
                        adjacency_dct[t][u] = [v]
                    if v in adjacency_dct[t]:
                        adjacency_dct[t][v].append(u)
                    else:
                        adjacency_dct[t][v] = [u]
        if save:
            self.adjacency_dct = adjacency_dct
        else:
            return adjacency_dct
    
    
    def get_adjacency_list(edgelist, which="out", save=False):
        assert which == "out" or which == "in"

        adjacency_list = [[[] for n in xrange(edgelist.Nnodes) ] for t in xrange( edgelist.times[0], edgelist.times[-1] + 1)]
        for t in edgelist.times:
            snapshot = edgelist.df[edgelist.df.time == t]
            for u,v in snapshot[["source", "target"]].values:
                if edgelist.directed:
                    if which == "out":
                        adjacency_list[t][u].append(v)
                    else:
                        adjacency_list[t][v].append(u)
                else:
                    adjacency_list[t][u].append(v)
                    adjacency_list[t][v].append(u)
        if save:
            edgelist.adjacency_list = adjacency_list
        else:
            return adjacency_list


    def get_adjacency_matrix(self, time, dtype=np.float128):
        if time in self.times:
            if self.adjacency_matrix_list is not None:
                return self.adjacency_matrix_list[time]
            else:
                edgelist = self.df[self.df.time == time]
                data = edgelist.weight
                row = edgelist.target
                col = edgelist.source
                if not self.directed:
                    data = data.append(edgelist.weight)
                    row = row.append(edgelist.source)
                    col = col.append(edgelist.target)
                return sp.csr_matrix((data, (row, col)), shape=(self.Nnodes, self.Nnodes), dtype=dtype)
        else:
            return sp.csr_matrix((self.Nnodes, self.Nnodes), dtype=dtype)

    def get_adjacency_matrix_list(self, save=False, dtype=np.float128):
        if self.adjacency_matrix_list is not None:
            return self.adjacency_matrix_list
        else:
            N = self.Nnodes
            adjacency_matrix_list = []
            for time in xrange(self.period):
                adjacency_matrix_list.append( self.get_adjacency_matrix(time, dtype=dtype) )
            if save:
                self.adjacency_matrix_list = adjacency_matrix_list
            return adjacency_matrix_list

    def average_degree_statistics(self, verbose=False):
        degree_list = []
        neighbor_degree_list = []
        degree_M2_list = []
        for time in self.times:
            edgelist = self.df[self.df.time == time][["source", "target"]]
            if self.directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            G.add_edges_from(edgelist.values)
            degree = nx.degree(G)
            neighbor_degree = nx.average_neighbor_degree(G)
            degree_list.append( np.mean([degree[n] if n in G else 0 for n in xrange(self.Nnodes)]) )
            neighbor_degree_list.append( np.mean([neighbor_degree[n] for n in xrange(self.Nnodes) if n in G]) )
            degree_M2_list.append(np.mean([degree[n]**2 if n in G else 0 for n in xrange(self.Nnodes)]))
        
        return {"mean_degree" : np.mean(degree_list), 
                "mean_nnDegree" : np.mean(neighbor_degree_list),
                "M2_degree" : np.mean(degree_M2_list),
                "mean_degree_list" : degree_list, 
                "mean_nnDegree_list" : neighbor_degree_list,
                "M2_degree_list" : degree_M2_list}

    # ============================================================================
    '''
    def simulation(self, alpha, beta, single_run=False, init_prob=1, store=10, iter_max=10000, iter_min=0, rel_tol=1e-4, return_details=False):
        attack_array = np.zeros(num_nodes) #Binning from 0 to N with additional time coordinate
        node_risk = np.zeros(num_nodes) #Number of nodes attacked by a given outbreak location
        node_vulnerability = np.zeros(num_nodes) #Number of times that a given node has been infected
        relative_error = np.zeros(num_nodes)
        iterations = np.zeros(num_nodes)
    '''

if __name__ == "__main__":
    state = "09" # Bayern
    folder = "/home/andreas/TU/Message_Passing/Data/HIT/2010/GSCC/Bundesland/"+state+"/"
    fname = folder + "edgelistements.csv"

    source = 0
    target = 1
    time = 2
    directed = False
    kwargs = {
        "delimiter" : ",",
        "verbose" : True,
        "weight" : None,
        "period" : None,
        "aggregate" : None,
        "t0" : 0
    }