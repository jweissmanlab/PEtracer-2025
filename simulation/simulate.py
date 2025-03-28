"""Code for simulating lineage tracing data and reconstructing trees."""
import multiprocessing as mp
from itertools import product

import cassiopeia as cas
import numpy as np
import pandas as pd
import petracer
from petracer.config import log_path, threads
from tqdm.auto import tqdm

base_path, data_path, plots_path, results_path = petracer.config.get_paths("simulation")
petracer.config.set_theme()

np.random.seed(42)

# Load source


# Load config
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
    """Simulate a lineage tree."""
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
    """Simulate lineage tracing data."""
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
    """Calculate normalized entropy of a distribution."""
    # Remove zero probabilities for log calculation
    distribution = distribution[distribution > 0]
    # Calculate entropy
    entropy = -np.sum(distribution * np.log2(distribution))
    # Normalize entropy
    num_states = len(distribution)
    normalized_entropy = entropy / np.log2(num_states)
    return normalized_entropy


def generate_state_distribution(n_states,entropy):
    """Generate a state distribution with a given entropy."""
    # iterate through exponential distributions
    for i in np.arange(0,30,.01):
        dist = np.logspace(i,0,n_states,base = np.e)
        dist = dist/sum(dist)
        # return if correct entropy
        if abs(entropy - normalized_entropy(dist)) < .01:
            return dist


def edit_frac_to_mutation_rate(tree,edit_frac):
    """Calculate mutation rate from edit fraction."""
    # get total simulation time
    total_time = tree.get_time(tree.leaves[0])
    # calculate necessary mutation rate
    mutation_rate = 1 - (1 - edit_frac) ** (1 / total_time)
    return mutation_rate


def identify_best_solver(results,group,metric = "rf",max = False):
    """Identify the best solver for each group."""
    results = results.copy()
    average_value = results.groupby(group + ['solver'], as_index=False)[metric].mean()
    if max:
        best_solver_idx = average_value.groupby(group)[metric].idxmax()
    else:
        best_solver_idx = average_value.groupby(group)[metric].idxmin()
    average_value['best_solver'] = average_value.index.isin(best_solver_idx)
    results = results.merge(average_value[group + ['solver', 'best_solver']], 
                            on=group + ['solver'], 
                            how='left')
    return results["best_solver"].values

# Tree evaluation
def eval_tree(param):
    """Evalulate tracing a tree with a given set of parameters."""
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
    result = pd.DataFrame({"triplets":mean_triplets,"rf":rf/rf_max},index = [0])
    result["triplets_by_depth"] = [list(triplets[0].values())]
    for key in param.keys():
        if isinstance(param[key],list):
            result[key] = [param[key]]
        else:
            result[key] = param[key]
    return result


def states_vs_entropy_simulation(threads = 30):
    """Simulate trees varying the number of states and the entropy."""
    # Define parameters
    params = {"solver":["nj","upgma","greedy"],
        "size":[1000],
        "edit_frac":[0.7],
        "characters":[60],
        "missing_rate":[.1],
        "entropy":[.5,.6,.7,.8,.9,1],
        "states":[1,2,4,8,16,32],
        "indel_dist":[False],
        "iteration":range(10)} 
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    threads = 30
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
                                total=len(parallel_list)))
        results = pd.concat(results)
    results["best_triplets"] = identify_best_solver(results, ["entropy", "states"], metric = "triplets", max=True)
    results["best_rf"] = identify_best_solver(results, ["entropy", "states"], metric = "rf")
    results.to_csv(results_path / "states_vs_entropy_simulation.csv",index = False)


def states_vs_frac_simulation(threads = 30):
    """Simulate trees varying the number of states and the edit fraction."""
    # Define parameters
    params = {"solver":["nj","upgma","greedy"],
        "size":[1000],
        "edit_frac":[0.4,0.5,0.6,0.7,0.8,0.9],
        "characters":[60],
        "missing_rate":[.1],
        "entropy":[1],
        "states":[1,2,4,8,16,32],
        "indel_dist":[False],
        "iteration":range(10)} 
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_tree, parallel_list), 
                                total=len(parallel_list)))
        results = pd.concat(results)
    results["best_triplets"] = identify_best_solver(results, ["entropy", "states"], metric = "triplets", max=True)
    results["best_rf"] = identify_best_solver(results, ["entropy", "states"], metric = "rf")
    results.to_csv(results_path / "states_vs_frac_simulation.csv",index = False)


def edit_frac_simulation(num_simulations=100,max_generations=100,max_characters=120):
    """Simulate the fraction of branches with edit."""
    results = []
    for edit_rate in np.linspace(0, .3, 301):
        edits = np.zeros((num_simulations,max_generations+1,max_characters+1), dtype=bool)
        for i in range(num_simulations):
            edited = np.zeros(max_characters+1, dtype=bool)
            for j in range(max_generations+1):
                edits[i,j] = (np.random.rand(max_characters+1) < edit_rate) & (~edited)
                edited = edited | edits[i,j]
        for j in range(1,max_generations+1):
            for k in range(1,max_characters+1):
                has_edit = np.any(edits[:,:j,:k],axis = (2))
                branch_edit_frac = np.sum(has_edit) / (j * num_simulations)
                site_edit_frac = edits[:,:j,:k].sum() / (k * num_simulations)
                results.append({"generations":j,"characters":k,
                                "branch_edit_frac":branch_edit_frac,
                                "site_edit_frac":site_edit_frac,
                                "edit_rate":edit_rate})
    results = pd.DataFrame(results)
    return results 


def min_characters_simulation(min_edit_fracs = (.7,.8,.9), num_simulations = 10):
    """Simulate the minimum number of characters for fraction of branches with edit."""
    results = []
    for iteration in range(num_simulations):
        edit_fracs = edit_frac_simulation(num_simulations=10,max_generations=30)
        for min_frac in min_edit_fracs:
            min_characters = edit_fracs.query("branch_edit_frac > @min_frac").groupby(
                "generations").agg({"characters":"min"}).reset_index()
            min_characters["branch_edit_frac"] = min_frac
            min_characters["iteration"] = iteration
            results.append(min_characters)
    results = pd.concat(results)
    results.to_csv(results_path / "min_characters_simulation.csv",index = False)


def edit_rate_simulation(min_edit_fracs = (.5,.6,.7,.8,.9), num_simulations = 10):
    """Simulate the optimal edit rate for fraction of sites with edit."""
    results = []
    for iteration in range(num_simulations):
        edit_fracs = edit_frac_simulation(num_simulations=10,max_characters=61)
        for edit_frac in min_edit_fracs:
            required_rate = edit_fracs.query("site_edit_frac > @edit_frac & characters == 60").sort_values(
                "site_edit_frac").groupby("generations").agg({"edit_rate":"first"}).reset_index()
            required_rate["iteration"] = iteration
            required_rate["site_edit_frac"] = edit_frac
            results.append(required_rate)
    results = pd.concat(results)
    results.to_csv(results_path / "edit_rate_simulation.csv",index = False)


def parameter_sweep_simulation(threads = 30):
    """Simulate trees sweeping the other parameters."""
    # Define parameters
    default_params = {"solver":["nj","upgma","greedy"],
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
    param_ranges = {"size":[300,1000,3000,10000,30000,100000],
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
    results.to_csv(results_path / "parameter_sweep_simulation.csv",index = False)

def get_indel_hnorm():
    """Calculate the normalized entropy of the indel distribution."""
    indel_dist = pd.read_csv(data_path / "indel_distribution.tsv", index_col=0,sep="\t")
    return normalized_entropy(indel_dist["probability"])


# Run simulations
if __name__ == "__main__":
    print("Indel distribution normalized entropy: ",get_indel_hnorm())
    print("Simulating trees varying the number of states and the entropy")
    states_vs_entropy_simulation(threads = threads)
    print("Simulating trees varying the number of states and the edit fraction")
    states_vs_frac_simulation(threads = threads)
    print("Simulating trees sweeping the other parameters")
    parameter_sweep_simulation(threads = threads)
    print("Simulating minimum number of characters for large trees")
    min_characters_simulation()
    print("Simulating optimal edit rate vs experiment length")
    edit_rate_simulation()
