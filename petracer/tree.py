import numpy as np
import pandas as pd
import networkx as nx
import cassiopeia as cas
import matplotlib.pyplot as plt
import treedata as td
import pycea
from numba import njit

from .config import edit_ids, edit_palette

def get_root(tree):
    return [node for node in tree.nodes if tree.in_degree(node) == 0][0]

def get_leaves(tree: nx.DiGraph):
    """Finds the leaves of a tree"""
    return [node for node in nx.dfs_postorder_nodes(tree, get_root(tree)) if tree.out_degree(node) == 0]

def get_edit_frac(characters):
    characters = np.array(characters)
    characters[np.isnan(characters)] = -1
    detected = characters[characters != -1]
    return np.sum(detected > 0)/len(detected)

def mask_truncal_edits(characters):
    masked_characters = {}
    for column in characters.columns:
        filtered_values = characters[characters[column] != -1][column]
        value_counts = filtered_values.value_counts()
        if len(value_counts[(value_counts.index != 0) & (value_counts.index != -1)]) > 0:
            most_common_value = value_counts[(value_counts.index != 0) & (value_counts.index != -1)].idxmax()
            fraction = (filtered_values == most_common_value).sum() / len(filtered_values)
            if fraction < .95:
                masked_characters[column] = characters[column]
    return pd.DataFrame(masked_characters)

def alleles_to_characters(alleles,edit_ids = edit_ids,min_prob = None,other_id = 9,order = None,index = "cellBC"):
    characters = alleles.copy()
    if isinstance(index,str):
        index = [index]
    # Map alleles to characters
    for site, mapping in edit_ids.items():
        characters[site] = characters[site].map(mapping).fillna(other_id).astype(int)
        if min_prob is not None and f"{site}_prob" in characters.columns:
            characters.loc[characters[f"{site}_prob"] < min_prob,site] = -1
    characters = pd.melt(characters[index + ["intID"] + list(edit_ids.keys())],
                               id_vars = index + ["intID"],var_name = "site",value_name = "allele")
    characters = characters.pivot_table(index = index,columns = ["intID","site"],values = "allele").fillna(-1).astype(int)
    # sort by max allele fraction
    def max_fraction(int_id):
        int_data = characters.xs(int_id, level=0, axis=1)
        counts = int_data.apply(pd.Series.value_counts, axis=0).fillna(0)
        total_counts = counts.sum(axis=0)
        valid_counts = counts.loc[lambda x: x.index > 0]  # Exclude -1 and 0
        max_fraction_value = (valid_counts / total_counts).max().max()
        return max_fraction_value
    if order is None:
        order = sorted(characters.columns.levels[0], key=max_fraction, reverse=True)
    characters = characters.reindex(order, level=0, axis=1)
    # Reindex
    characters.columns = ['{}-{}'.format(intID, site) for intID, site in characters.columns]
    return characters

def reconstruct_ancestral_characters(tdata,tree = "tree",key = "characters",edit_cost = .6,copy = True):
    tree_key = tree
    if copy:
        tdata = tdata.copy()
    n_characters = max(tdata.obsm[key].max().max() + 1,10)
    costs = np.ones(shape = (n_characters,n_characters),dtype=float)
    costs[0,:] = edit_cost
    np.fill_diagonal(costs,0)
    costs = pd.DataFrame(costs,index = range(0,n_characters),columns = range(0,n_characters))
    pycea.tl.ancestral_states(tdata,keys = key,method = "sankoff",costs = costs,missing_state=-1,tree = tree_key)
    if copy:
        return tdata
    
def estimate_branch_lengths(tdata,tree = "tree",key = "characters",copy = True):
    tree_key = tree
    if copy:
        tdata = tdata.copy()
    tree = tdata.obst[tree_key]
    node_attrs = dict(tree.nodes(data=True))
    for node in node_attrs:
        del node_attrs[node]["time"]
    cas_tree = cas.data.CassiopeiaTree(character_matrix = tdata.obsm[key],tree = tree)
    for node in tree.nodes:
        cas_tree.set_character_states(node,tree.nodes[node][key])
    cas.tl.IIDExponentialMLE().estimate_branch_lengths(cas_tree)
    tree = cas_tree.get_tree_topology()
    nx.set_node_attributes(tree,node_attrs)
    for node in tree.nodes:
        del tree.nodes[node]["character_states"]
    del tdata.obst[tree_key]
    tdata.obst[tree_key] = tree
    if copy:
        return tdata

def collapse_mutationless_edges(tdata,tree = "tree",key = "characters",copy = True,tree_added = "tree",mutation_key = None):
    tree_key = tree
    if copy:
        tdata = tdata.copy()
    tree = tdata.obst[tree].copy()
    root = [node for node in tree.nodes if tree.in_degree(node) == 0][0]
    for edge in reversed(list(nx.dfs_edges(tree,root))):
        if mutation_key is not None:
            has_mutation = tree.edges[edge][mutation_key]
        else:
            has_mutation = np.any(tree.nodes[edge[1]][key] != tree.nodes[edge[0]][key])
        if not has_mutation:
            children = list(tree.successors(edge[1]))
            if len(children) > 0:
                for child in children:
                    tree.add_edge(edge[0],child)
                tree.remove_edge(*edge)
                tree.remove_node(edge[1])
    tdata.obst[tree_added] = tree
    if copy:
        return tdata


def majority_character(characters,min_size = 20,min_frac = .8):
    """Find the majority character in a list of characters"""
    characters = np.array(characters)
    if len(characters) < min_size:
        return -1
    characters = characters[characters != -1]
    if len(characters) == 0:
        return -1
    unique_values, counts = np.unique(characters, return_counts=True)
    for value, count in zip(unique_values, counts):
        if count / len(characters) > min_frac:
            return value
    return 0


def same_characters(c1, c2):
    """Check if two arrays are equal, ignoring -1 values"""
    c1 = np.array(c1)
    c2 = np.array(c2)
    mask = (c1 != -1) & (c2 != -1)
    if not mask.any():
        return True
    return np.array_equal(c1[mask], c2[mask])


def identify_mutations(tdata,tree = "tree",key = "characters",key_added = "has_mutation",min_frac = .75,copy = False):
    """Mark edges with a mutation"""
    if copy:
        tdata = tdata.copy()
    method = lambda x: majority_character(x,min_frac = min_frac)
    pycea.tl.ancestral_states(tdata, keys = key, method = method,keys_added="majority_characters",tree = tree)
    for edge in tdata.obst[tree].edges:
        has_mutation = not (same_characters(tdata.obst[tree].nodes[edge[0]]["majority_characters"],
                                          tdata.obst[tree].nodes[edge[1]]["majority_characters"]) and
                            same_characters(tdata.obst[tree].nodes[edge[0]][key],
                                            tdata.obst[tree].nodes[edge[1]][key]))
        tdata.obst[tree].edges[edge][key_added] = has_mutation
    for node in tdata.obst[tree].nodes:
        del tdata.obst[tree].nodes[node]["majority_characters"]
    if copy:
        return tdata


def bfs_names(tree):
    bfs_nodes = list(nx.bfs_tree(tree, source=pycea.utils.get_root(tree)))
    leaves = pycea.utils.get_leaves(tree)
    return {node: f"node{i}" for i, node in enumerate(bfs_nodes) if node not in leaves}


def reconstruct_tree(tdata,solver = "upgma",key = "characters",tree_added = "tree",estimate_lengths = True,edit_cost = .6,
                     reconstruct_characters = True,collapse_edges = False,keep_distances = False,mask_truncal = False,
                     upweight = None):
    solvers = {"nj":cas.solver.NeighborJoiningSolver(cas.solver.dissimilarity.weighted_hamming_distance,
                                                 add_root=True,fast = True),
           "upgma":cas.solver.UPGMASolver(cas.solver.dissimilarity.weighted_hamming_distance,fast = True),
           "greedy":cas.solver.VanillaGreedySolver()}
    characters = tdata.obsm[key]
    masked_characters = mask_truncal_edits(characters).copy() if mask_truncal else characters.copy()
    if upweight is not None:
        for character in upweight:
            for i in range(1):
                masked_characters[f"{character}_{i}"] = masked_characters[character]
    cas_tree = cas.data.CassiopeiaTree(character_matrix=masked_characters)
    solvers[solver].solve(cas_tree)
    #cas_tree.reconstruct_ancestral_characters()
    tree = cas_tree.get_tree_topology()
    for node in tree.nodes:
        del tree.nodes[node]["character_states"]
    tdata.obst[tree_added] = tree
    if keep_distances:
        if solver == "greedy":
            cas_tree.compute_dissimilarity_map(cas.solver.dissimilarity.weighted_hamming_distance)
        tdata.obsp["character_distances"] = cas_tree.get_dissimilarity_map().loc[tdata.obs_names,tdata.obs_names]
    if reconstruct_characters:
        reconstruct_ancestral_characters(tdata,copy = False,tree = tree_added,edit_cost=edit_cost)
    if estimate_lengths:
        estimate_branch_lengths(tdata,copy = False,tree = tree_added)
    if collapse_edges:
        collapse_mutationless_edges(tdata,copy = False,tree_added = tree_added,tree = tree_added)
    tdata.obst[tree_added] = nx.relabel_nodes(tdata.obst[tree_added],bfs_names(tdata.obst[tree_added]))
    return tdata

def plot_grouped_characters(tdata,ax = None,width = .1,label = False,offset = .05):
    """Plot allele table grouped by integration"""
    if ax is None:
        ax = plt.gca()
    tdata.obs = tdata.obs.merge(tdata.obsm["characters"].astype(str),left_index=True,right_index=True)
    for i in range(0,tdata.obsm["characters"].shape[1],3):
        integration = tdata.obsm["characters"].columns[i].split("-")[0]
        label = integration.replace("intID","") if label else False
        gap = offset if i == 0 else width/2
        pycea.pl.annotation(tdata,keys=[f"{integration}-RNF2",f"{integration}-HEK3",f"{integration}-EMX1"],border_width=.5,
                            label = label,width=width,gap = gap,palette = edit_palette,ax = ax)
    tdata.obs = tdata.obs.drop(columns = tdata.obsm["characters"].columns)
    ax.tick_params(axis='x', pad=0)

@njit
def hamming_distance(arr1, arr2):
    valid_mask = (arr1 != -1) & (arr2 != -1)
    hamming_distance = 0
    for x, y in zip(arr1[valid_mask], arr2[valid_mask]):
        if x == y:
            pass
        elif x == 0 or y == 0:
            hamming_distance += 1
        else:
            hamming_distance += 2
    num_valid_comparisons = np.sum(valid_mask)
    if num_valid_comparisons == 0:
        return 0
    normalized_distance = hamming_distance / num_valid_comparisons
    return normalized_distance

def estimate_leaf_fitness(tdata,tree = "tree",depth_key = "depth",key_added = "fitness",copy = False):
    tree_key = tree
    if copy:
        tdata = tdata.copy()
    nx_tree = tdata.obst[tree_key].copy()
    for node in nx_tree:
        nx_tree.nodes[node]["_depth"] = nx_tree.nodes[node][depth_key]
    cas_tree = cas.data.CassiopeiaTree(tree = nx_tree)
    for edge in cas_tree.depth_first_traverse_edges():
        t1 = cas_tree.get_attribute(edge[0],"_depth")
        t2 = cas_tree.get_attribute(edge[1],"_depth")
        cas_tree.set_branch_length(edge[0], edge[1], abs(t1-t2))
    fitness_estimator = cas.tools.fitness_estimator.LBIJungle()
    fitness_estimator.estimate_fitness(cas_tree)
    fitnesses = np.array([cas_tree.get_attribute(cell, 'fitness') for cell in cas_tree.leaves])
    fitnesses = pd.Series(fitnesses, index=cas_tree.leaves)
    tdata.obs[key_added] = tdata.obs_names.map(fitnesses)
    if copy:
        return tdata
    
def n_extant(tdata, depth_key, groupby = None, bins = 20, tree = "tree"):
    # Get nodes
    if groupby is None:
        groupby = "_all"
        nodes = pycea.utils.get_keyed_node_data(tdata, keys = [depth_key],tree = tree)
        nodes["_all"] = 1
    else:
        nodes = pycea.utils.get_keyed_node_data(tdata, keys = [depth_key, groupby])
    nodes.index = nodes.index.droplevel("tree")
    # Get timepoints
    timepoints = np.histogram_bin_edges(nodes[depth_key],bins = bins)
    # Get counts
    groups = nodes[groupby].unique()
    group_counts = {group: np.zeros_like(timepoints) for group in groups}
    for edge in tdata.obst[tree].edges:
        birth_idx = np.searchsorted(timepoints,nodes.loc[edge[0],depth_key] - 1e-4, side='right')
        death_idx = np.searchsorted(timepoints,nodes.loc[edge[1],depth_key] + 1e-4, side='left')
        group_counts[nodes.loc[edge[0],groupby]][birth_idx:death_idx] += 1
    group_n = []
    for group, counts in group_counts.items():
        group_n.append(pd.DataFrame({"time":timepoints,"n_extant":counts,groupby:group}))
    group_n = pd.concat(group_n)
    if groupby == "_all":
        group_n.drop(columns = "_all",inplace = True)
    return group_n