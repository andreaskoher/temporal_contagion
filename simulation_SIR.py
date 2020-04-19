# ===================== Load Packages =================
import pickle
import numpy as np
from tqdm import tqdm

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
class Simulation:
    def __init__(self):
        pass
    # ============================================================================
    def load(self, edgelist, directed, source, target, time, weight=None, aggregate=None, period=None, verbose=True, strongly_connected=False, **kwargs):
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
        
        # --------------- accept file name ---------------------------------
        if isinstance(edgelist, str): #expect path to file with edgelist
            edgelist = np.genfromtxt(edgelist, unpack=False, **kwargs)
            if verbose:
                print "file successfully loaded..."

        # --------------- accept numpy.ndarray ---------------------------------
        if isinstance(edgelist, np.ndarray) or isinstance(edgelist, list):
            edgelist_array = np.array(edgelist)
            edgelist = pd.DataFrame(data={"source" : edgelist_array[:,source],
                                        "target" : edgelist_array[:,target]},
                                        "time": edgelist_array[:,target])
            source, target, time = "source", "target", "time"
            
            if weight is not None:
                edgelist["weight"] = edgelist_array[:,weight]
                weight = "weight"
            del edgelist_array

        # --------------- accept pandas.DataFrame ---------------------------------
        if isinstance(edgelist, pd.DataFrame):
            self.edgelist = edgelist.copy()
            del edgelist
            self.edgelist.rename({source : "source", target : "target", time: "time"}, axis=1, inplace=True)

            self.edgelist.source = self.edgelist.source.astype(int)
            self.edgelist.target = self.edgelist.target.astype(int)
            self.Nnodes = len( set(self.edgelist.source) | set(self.edgelist.target) )

            self.edgelist.time = self.edgelist.time.astype(int) #assuming integer values!
            if self.edgelist.time[0] != 0:
                if verbose:
                    print "time stamps will be shifted and start with 0..."
                self.edgelist.time -= self.edgelist.time[0]
            self.times = self.edgelist.time.unique()

            if weight is None:
                self.weighted = False
                self.edgelist["weight"] = 1.
                weight = "weight"
                if verbose:
                    "unit weights added in column 'weight' ..."
            else:
                self.edgelist.rename({weight : "weight"}, axis=1, inplace=True)
            self.edgelist.weight = self.edgelist.weight.astype(float)
            self.edgelist = self.edgelist[["source", "target", "weight", "time"]]
            
            if aggregate is not None:
                assert isinstance(aggregate, int), "aggregation interval must be of type int, got {}".format(type(aggregate))
                self.aggregate_edgelist(aggregate)
                self.times = self.edgelist.time.unique()
                
            if period is not None:
                self.edgelist = self.edgelist[self.edgelist.time < period]
                self.period = period
                self.times = self.edgelist.time.unique()
            else:
                self.period = self.times[-1] + 1
            
            selfloops = self.edgelist.source == self.edgelist.target
            self.edgelist = self.edgelist[~ selfloops].copy()
            if verbose:
                if selfloops.any():
                    print "{} self-loops removed...".format(selfloops.sum())

            if not directed:
                switch = self.edgelist.source > self.edgelist.target
                self.edgelist.loc[switch, ("source", "target")] = self.edgelist.loc[switch, ["target", "source"]].values
            self.edgelist = self.edgelist.groupby(['time','source','target']).sum().reset_index()
            if not self.weighted:
                    self.edgelist.weight = 1.
            #self.Nedges = len( self.edgelist[['weight','source','target']].groupby(['source','target']).sum().reset_index() )
            self.Ncontacts = len(self.edgelist)

            self.edgelist = self.edgelist[["source", "target", "time"]].values #BEWARE: ignoring weights...
            self.neighbours = {n}

            if verbose:
                print "graph has been saved in self.graph"
                print "number of nodes, (temporal) contacts", self.Nnodes, self.Ncontacts
                print "time runs from {} to {} in {} time steps".format(self.times[0], self.times[-1], len(self.times)) 
        else:
            raise ValueError("Expected type of graph is either str, numpy.ndarray or pandas.DataFrame. Instead got {}".format(type(graph)))

    def aggregate_edgelist(self, dt):
        """
        Aggregate in time each dt time steps
        Args:
            dt(int) : Time interval used to aggregate. dt = 0 specifies full aggregation.
            store   : if true stores aggregated movements in self.aggregations_t, otherwise returns it
        Returns:
            movements (DataFrame) : if store = False the updated movements are returned, otherwise None.
        """
        if dt == 0:
            self.edgelist.time = 0
        else:
            self.edgelist['time'] //= dt
        self.edgelist = self.edgelist.groupby(['time','source','target']).sum().reset_index()
        # Check if T is divisible by dt. If not, drop last interval.
        T = self.times[-1]+1
        if dt>0 and T%dt != 0:
            self.edgelist = self.edgelist[self.edgelist.time != self.times[-1]]
    # ============================================================================
    '''
    def simulation(self, alpha, beta, single_run=False, init_prob=1, store=10, iter_max=10000, iter_min=0, rel_tol=1e-4, return_details=False):
        attack_array = np.zeros(num_nodes) #Binning from 0 to N with additional time coordinate
        node_risk = np.zeros(num_nodes) #Number of nodes attacked by a given outbreak location
        node_vulnerability = np.zeros(num_nodes) #Number of times that a given node has been infected
        relative_error = np.zeros(num_nodes)
        iterations = np.zeros(num_nodes)
    '''