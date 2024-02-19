# Run: nohup python lineage_tracer_simulation/simulate.py > output.log 2>&1 &

import sys
import numpy as np
import pandas as pd
import cassiopeia as cas
import multiprocessing as mp
import logging
from itertools import product
from pathlib import Path
from tqdm.auto import tqdm

# Configure paths
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
module_path = Path(__file__).parent.parent
sys.path.append(str(module_path))

# Load config
from config import threads, log_path
logfile = None if log_path is None else results_path / "log"

# Load indel distribution
indel_dist = pd.read_csv(data_path / "indel_distribution.tsv",sep = "\t")

# Solvers
solvers = {"nj":cas.solver.NeighborJoiningSolver(
                cas.solver.dissimilarity.weighted_hamming_distance,
                add_root=True,fast=True),
           "vanilla_greedy":cas.solver.VanillaGreedySolver(),
           "upgma":cas.solver.UPGMASolver(
                cas.solver.dissimilarity.weighted_hamming_distance,fast=True),
           "vanilla_hybrid": cas.solver.HybridSolver(
                top_solver = cas.solver.VanillaGreedySolver(),
                bottom_solver= cas.solver.ILPSolver(
                    convergence_time_limit=10,
                    maximum_potential_graph_layer_size=1000,
                    maximum_potential_graph_lca_distance = 10),
               cell_cutoff=20, threads=1, progress_bar =False)}

# Lineage simulators
lineage_simulators = {
    "lognormal_fit": lambda num_extant: cas.sim.BirthDeathFitnessSimulator(
        birth_waiting_distribution = lambda scale: 
            np.random.lognormal(mean = np.log(scale),sigma = .5),
        initial_birth_scale = 1,
        death_waiting_distribution = lambda: np.random.uniform(0,4),
        mutation_distribution = lambda: 1,
        fitness_distribution = lambda: np.random.normal(0, .25),
        fitness_base = 1,
        num_extant = num_extant),
}

# Tracing simulators
tracing_simulators = {
    "pe": lambda cassettes, edit_rate, missing_rate, state_priors: 
    cas.sim.Cas9LineageTracingDataSimulator(
        number_of_cassettes = cassettes,
        size_of_cassette = 3,
        mutation_rate = edit_rate,
        state_priors = state_priors, 
        heritable_silencing_rate=0,
        collapse_sites_on_cassette=False,
        stochastic_silencing_rate = missing_rate),
}

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
    lineage_sim = lineage_simulators[param["tree_simulator"]](int(param["size"]))
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
    tracing_simulators[param["tracing_simulator"]](param["cassettes"],
        edit_frac_to_mutation_rate(tree,param["edit_frac"]),
        param["missing_rate"],state_priors).overlay_data(tree)
    # Reconstruct tree
    reconstructed_tree = tree.copy()
    if param["solver"] in ["vanilla_greedy","vanilla_hybrid"]:
        reconstructed_tree.priors = {i:state_priors for i in range (param["cassettes"]*3)}
    solvers[param["solver"]].solve(reconstructed_tree,logfile = logfile)
    # Calculate metrics
    triplets = cas.critique.compare.triplets_correct(
        tree, reconstructed_tree, number_of_trials=500,min_triplets_at_depth=5)
    mean_triplets = np.mean(list(triplets[0].values())) 
    rf, rf_max = cas.critique.compare.robinson_foulds(tree, reconstructed_tree)
    # Format results
    result = pd.DataFrame({"triplets":mean_triplets,"rf":rf/rf_max},index = [0])
    result["triplets_by_depth"] = [list(triplets[0].values())]
    for key in param.keys():
        if type(param[key]) is list:
            result[key] = [param[key]]
        else:
            result[key] = param[key]
    return result

# Simulate trees varying the number of states and the entropy
def state_distribution_simulation(threads = 30):
    # Define parameters
    params = {"solver":["vanilla_greedy","nj","vanilla_hybrid","upgma"],
            "tree_simulator":["lognormal_fit"],
            "tracing_simulator":["pe"],
            "size":[1000],
            "edit_frac":[0.7],
            "cassettes":[8],
            "missing_rate":[0],
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

# Simulate trees varying the number of cassettes and the missing rate
def missing_vs_cassettes_simulation(threads = 30):
    # Define parameters
    params = {"solver":["vanilla_greedy","nj","vanilla_hybrid","upgma"],
            "tree_simulator":["lognormal_fit"],
            "tracing_simulator":["pe"],
            "size":[1000],
            "edit_frac":[0.7],
            "cassettes":[5,10,15,20,25],
            "missing_rate":[0,.1,.2,.3],
            "entropy":[1],
            "states":[8],
            "indel_dist":[True,False],
            "iteration":range(10)}
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
                            total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "missing_vs_cassettes_simulation.tsv",
                   index = False,sep = "\t")
    
# Simulate trees varying the number of cassettes and number of cells
def size_vs_cassettes_simulation(threads = 30):
    # Define parameters
    params = {"solver":["vanilla_greedy","nj","vanilla_hybrid","upgma"],
            "tree_simulator":["lognormal_fit"],
            "tracing_simulator":["pe"],
            "size":[500,1000,2000,5000],
            "edit_frac":[0.7],
            "cassettes":[5,10,15,20,25],
            "missing_rate":[0],
            "entropy":[1],
            "states":[8],
            "indel_dist":[True,False],
            "iteration":range(10)}
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
                            total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "size_vs_cassettes_simulation.tsv",
                   index = False,sep = "\t")
    
def size_vs_cassettes_simulation(threads = 30):
    # Define parameters
    params = {"solver":["vanilla_greedy","nj","vanilla_hybrid","upgma"],
            "tree_simulator":["lognormal_fit"],
            "tracing_simulator":["pe"],
            "size":[500,1000,2000,5000],
            "edit_frac":[0.7],
            "cassettes":[5,10,15,20,25],
            "missing_rate":[0],
            "entropy":[1],
            "states":[8],
            "indel_dist":[True,False],
            "iteration":range(10)}
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
                            total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "size_vs_cassettes_simulation.tsv",
                   index = False,sep = "\t")
    
def test(threads = 30):
    # Define parameters
    params = {"solver":["vanilla_hybrid"],
            "tree_simulator":["lognormal_fit"],
            "tracing_simulator":["pe"],
            "size":[500,1000],
            "edit_frac":[0.7],
            "cassettes":[5],
            "missing_rate":[0],
            "entropy":[1],
            "states":[8],
            "indel_dist":[True,False],
            "iteration":range(10)}
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    parallel_list = [row.to_dict() for index,row in params.iterrows()]
    results = eval_tree(parallel_list[0])
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
    #print("Simulating trees varying the number of cassettes and the missing rate")
    #missing_vs_cassettes_simulation(threads = threads)
    print("Simulating trees varying the number of cassettes and number of cells")
    size_vs_cassettes_simulation(threads = threads)
