"""Code for simulating lineage tracing data and reconstructing trees."""

import sys
import numpy as np
import pandas as pd
import cassiopeia as cas
import multiprocessing as mp
from itertools import product
from pathlib import Path
from tqdm.auto import tqdm

# Configure paths
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
module_path = Path(__file__).parent.parent
sys.path.append(str(module_path))

# Load source
from src.config import threads, log_path

# Load config
print(threads)
logfile = None if log_path is None else results_path / "log"

# Load indel distribution
indel_dist = pd.read_csv(data_path / "indel_distribution.tsv",sep = "\t")

# Solvers
solvers = {"nj":cas.solver.NeighborJoiningSolver(cas.solver.dissimilarity.weighted_hamming_distance,add_root=True,fast=True),
           "greedy":cas.solver.VanillaGreedySolver(),
           "upgma":cas.solver.UPGMASolver(cas.solver.dissimilarity.weighted_hamming_distance,fast=True),
           "hybrid": cas.solver.HybridSolver(
                top_solver = cas.solver.VanillaGreedySolver(),
                bottom_solver= cas.solver.ILPSolver(convergence_time_limit=100,
                                                    maximum_potential_graph_layer_size=1000,
                                                    maximum_potential_graph_lca_distance = 10),
               cell_cutoff=20, threads=1,progress_bar=False)}

# Lineage simulator
def lineage_simulator(num_extant):
    return cas.sim.BirthDeathFitnessSimulator(
        birth_waiting_distribution = lambda scale: np.random.lognormal(mean = np.log(scale),sigma = .5),
        initial_birth_scale = 1,
        death_waiting_distribution = lambda: np.random.uniform(0,4),
        mutation_distribution = lambda: 0,
        fitness_distribution = lambda: 0,
        fitness_base = 1,
        num_extant = num_extant)

# Tracing simulator
def tracing_simulator(characters, edit_rate, missing_rate, state_priors):
    return cas.sim.Cas9LineageTracingDataSimulator(
        number_of_cassettes = characters,
        size_of_cassette = 1,
        mutation_rate = edit_rate,
        state_priors = state_priors, 
        heritable_silencing_rate=0,
        collapse_sites_on_cassette=False,
        stochastic_silencing_rate = missing_rate)

# Helper functions
def normalized_entropy(distribution):
    # Remove zero probabilities for log calculation
    distribution = distribution[distribution > 0]
    # Calculate entropy
    entropy = -np.sum(distribution * np.log2(distribution))
    # Normalize entropy
    num_states = len(distribution)
    normalized_entropy = entropy / np.log2(num_states)
    return normalized_entropy

def generate_state_distribution(n_states,entropy):
    # iterate through exponential distributions
    for i in np.arange(0,30,.01):
        dist = np.logspace(i,0,n_states,base = np.e)
        dist = dist/sum(dist)
        # return if correct entropy
        if abs(entropy - normalized_entropy(dist)) < .01:
            return dist
        
def edit_frac_to_mutation_rate(tree,edit_frac):
    # get total simulation time
    total_time = tree.get_time(tree.leaves[0])
    # calculate necessary mutation rate
    mutation_rate = -(np.log(1 - edit_frac) / total_time) * .9
    return mutation_rate

# Tree evaluation
def eval_tree(param):
    np.random.seed(int(param["iteration"]))
    # Simulate the tree
    lineage_sim = lineage_simulator(int(param["size"]))
    tree = None
    while tree is None:
        try: tree = lineage_sim.simulate_tree()
        except Exception as e: continue
    # Get state priors
    if param["indel_dist"]:
        state_priors = indel_dist.to_dict()['probability']
    else:
        dist = generate_state_distribution(param["states"],param["entropy"])
        state_priors = {i+1: p for i, p in enumerate(dist)}
    # Simulate the data
    mutation_rate = edit_frac_to_mutation_rate(tree,param["edit_frac"])
    tracing_simulator(param["characters"],mutation_rate,
                      param["missing_rate"],state_priors).overlay_data(tree)
    # Reconstruct tree
    reconstructed_tree = tree.copy()
    if param["solver"] in ["greedy","hybrid"]:
        reconstructed_tree.priors = {i:state_priors for i in range (param["characters"])}
    solvers[param["solver"]].solve(reconstructed_tree,logfile = None)
    # Calculate metrics
    triplets = cas.critique.compare.triplets_correct(tree, reconstructed_tree, 
                                                     number_of_trials=1000,min_triplets_at_depth=100)
    mean_triplets = np.mean(list(triplets[0].values())) 
    rf, rf_max = cas.critique.compare.robinson_foulds(tree, reconstructed_tree)
    # Format results
    result = pd.DataFrame({"mean_triplets":mean_triplets,"rf":rf/rf_max},index = [0])
    result["triplets_by_depth"] = [list(triplets[0].values())]
    for key in param.keys():
        if isinstance(param[key],list):
            result[key] = [param[key]]
        else:
            result[key] = param[key]
    return result

# Simulate trees varying the number of states and the entropy
def state_distribution_simulation(threads = 30):
    # Define parameters
    params = {"solver":["nj","upgma","greedy","hybrid"],
        "size":[1000],
        "edit_frac":[0.7],
        "characters":[60],
        "missing_rate":[.1],
        "entropy":[1,.9,.8,.7,.6,.5],
        "states":[2,4,6,8,10,12,14,16,18,20],
        "indel_dist":[False],
        "iteration":range(10)} 
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
                                total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "state_distribution_simulation.tsv",
                   index = False,sep = "\t")

# Simulate trees sweeping the other parameters
def parameter_sweep_simulation(threads = 30):
    # Define parameters
    default_params = {"solver":["greedy","nj","upgma","hybrid"],
        "tree_simulator":["lognormal_fit"],
        "tracing_simulator":["pe"],
        "size":[1000],
        "edit_frac":[0.7],
        "missing_rate":[.1],
        "characters":[60],
        "entropy":[1],
        "states":[8],
        "indel_dist":[True,False],
        "iteration":range(10)}
    param_ranges = {"size":[500,1000,2000,5000,10000],
        "edit_frac":[0.3,0.5,0.7,0.9],
        "characters":[20,40,60,80,100],
        "missing_rate":[0,.1,.2,.3,.4,.5]}
    # Generate parameter combinations
    params = []
    for param in param_ranges.keys():
        for value in param_ranges[param]:
            value_params = pd.DataFrame(list(product(*default_params.values())), columns=default_params.keys())
            value_params.loc[:,param] = value
            params.append(value_params)
    params = pd.concat(params)
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
                                total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "parameter_sweep_simulation.tsv",
                   index = False,sep = "\t")
    
def test(threads = 30):
    # Define parameters
    params = {"solver":["nj"],
            "tree_simulator":["lognormal_fit"],
            "tracing_simulator":["pe"],
            "size":[500,1000],
            "edit_frac":[0.7],
            "characters":[10],
            "missing_rate":[0],
            "entropy":[1],
            "states":[8],
            "indel_dist":[True,False],
            "iteration":range(10)}
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    parallel_list = [row.to_dict() for index,row in params.iterrows()]
    results = eval_tree(parallel_list[0])
    print(results)
    #with mp.Pool(processes=threads) as pool:
    #    parallel_list = [row.to_dict() for index,row in params.iterrows()]
    #    results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
    #                        total=len(parallel_list)))
    #    results = pd.concat(results)
    #results.to_csv(results_path / "size_vs_cassettes_simulation.tsv",
    #               index = False,sep = "\t")

# Run simulations
if __name__ == "__main__":
    #print("Simulating trees varying the number of states and the entropy")
    #state_distribution_simulation(threads = threads)
    #print("Simulating trees sweeping the other parameters")
    #param_sweep_simulation(threads = threads)
