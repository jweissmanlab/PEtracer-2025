import numpy as np
import pandas as pd
import networkx as nx
import cassiopeia as cas
import matplotlib.pyplot as plt
import treedata as td
import pycea

from .config import edit_ids, edit_palette

def get_root(tree):
    return [node for node in tree.nodes if tree.in_degree(node) == 0][0]

def get_leaves(tree):
    return [node for node in tree.nodes if tree.out_degree(node) == 0]

def get_edit_frac(characters):
    return np.float64(np.apply_over_axes(np.sum,characters != 0,(0,1)).item()/
                      np.apply_over_axes(np.sum,~np.isnan(characters),(0,1)).item())

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

def reconstruct_ancestral_characters(tdata,tree = "tree",key = "characters",copy = True):
    tree_key = tree
    if copy:
        tdata = tdata.copy()
    costs = np.ones(shape = (10,10),dtype=float)
    costs[0,:] = .4
    np.fill_diagonal(costs,0)
    costs = pd.DataFrame(costs,index = range(0,10),columns = range(0,10))
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
    
def collapse_mutationless_edges(tdata,tree = "tree",key = "characters",copy = True):
    tree_key = tree
    if copy:
        tdata = tdata.copy()
    tree = tdata.obst[tree].copy()
    root = [node for node in tree.nodes if tree.in_degree(node) == 0][0]
    for edge in reversed(list(nx.dfs_edges(tree,root))):
        if tree.nodes[edge[0]][key] == tree.nodes[edge[1]][key]:
            children = list(tree.successors(edge[1]))
            if len(children) > 0:
                for child in children:
                    tree.add_edge(edge[0],child)
                tree.remove_edge(*edge)
                tree.remove_node(edge[1])
    tdata.obst[tree_key] = tree
    if copy:
        return tdata

def reconstruct_tree(tdata,solver = "upgma",key = "characters",tree_added = "tree",estimate_lengths = True,
                     reconstruct_characters = True,collapse_edges = False):
    solvers = {"nj":cas.solver.NeighborJoiningSolver(cas.solver.dissimilarity.weighted_hamming_distance,
                                                 add_root=True,fast = True),
           "upgma":cas.solver.UPGMASolver(cas.solver.dissimilarity.weighted_hamming_distance,fast = True),
           "greedy":cas.solver.VanillaGreedySolver()}
    cas_tree = cas.data.CassiopeiaTree(character_matrix=tdata.obsm[key])
    solvers[solver].solve(cas_tree)
    cas_tree.reconstruct_ancestral_characters()
    tree = cas_tree.get_tree_topology()
    for node in tree.nodes:
        del tree.nodes[node]["character_states"]
    tdata.obst[tree_added] = tree
    if solver != "greedy":
        tdata.obsp["character_distances"] = cas_tree.get_dissimilarity_map().loc[tdata.obs_names,tdata.obs_names]
    if reconstruct_characters:
        reconstruct_ancestral_characters(tdata,copy = False)
    if estimate_lengths:
        estimate_branch_lengths(tdata,copy = False)
    if collapse_edges:
        collapse_mutationless_edges(tdata,copy = False)
    return tdata

def plot_grouped_characters(tdata,ax = None,width = .1,label = False,offset = .05):
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