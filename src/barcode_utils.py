import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from .config import sequential_cmap
from .tree_utils import get_root, get_leaves

def score_kinship(int_counts,ints):
    return int_counts[ints].sum(axis = 1)/int_counts.sum(axis = 1)

def group_from_seed(int_counts,seed,min_frac = .8,min_kinship = .8,min_count = 10):
    int_pass = int_counts >= min_count
    int_fracs = int_pass.loc[int_pass[seed],:].mean()
    ints = int_fracs[int_fracs >= min_frac].index
    kinship = score_kinship(int_counts,ints)
    cells = kinship[kinship >= min_kinship].index
    return list(ints), list(cells)

def cluster_barcodes(barcodes,min_size = 2,min_count = 10, plot = False,plot_title = None):
    barcode_counts = barcodes.pivot_table(index=['cellBC'], columns='intBC', values='UMI').fillna(0)
    seeds = list((barcode_counts > min_count).sum(axis = 0).sort_values(ascending = False).index)
    unassigned = set(barcode_counts.index)
    cluster = 1
    cell_to_cluster = []
    int_order = []
    while len(unassigned) > 0 and len(seeds) > 0:
        seed = seeds.pop(0)
        ints, cells = group_from_seed(barcode_counts.loc[list(unassigned),:],seed,min_count = min_count)
        if len(cells) >= min_size:
            cell_to_cluster.append(pd.DataFrame({"cellBC":cells,"cluster":cluster}))
            int_order += list(set(ints) - set(int_order))
            cluster += 1
            unassigned = unassigned - set(cells)
    cell_to_cluster = pd.concat(cell_to_cluster)
    cell_to_cluster["cluster"] = cell_to_cluster["cluster"].astype(str)
    if plot:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        sns.heatmap(barcode_counts.loc[cell_to_cluster.cellBC,int_order],cmap = sequential_cmap,vmax = 50,
                    yticklabels=False,xticklabels=False,ax = axes)
        plt.title(plot_title)
    return cell_to_cluster

def get_barcode_clades(tdata, barcode, key = None):
    """Get clades that maximize the FMI for a given barcode"""
    # Setup
    if key is None:
        key = tdata.obst_keys()[0]
    tree = tdata.obst[key]
    nodes = list(tree.nodes)
    root = get_root(tree)
    barcode_values = tdata.obs[barcode].dropna().unique()
    # Count barcode under each node
    node_barcode_counts = pd.DataFrame(index = nodes, columns = barcode_values, data = 0)
    for leaf, value in tdata.obs[barcode].dropna().items():
        node_barcode_counts.loc[leaf,value] += 1
    for node in nx.dfs_postorder_nodes(tree,root):
        if tree.out_degree(node) == 0:
            continue
        children = list(tree.successors(node))
        node_barcode_counts.loc[node] = node_barcode_counts.loc[children].sum()
    # Calculate FMI
    node_barcode_recall = node_barcode_counts.div(node_barcode_counts.loc[root],axis = 1)
    node_barcode_precision = node_barcode_counts.div(node_barcode_counts.sum(axis = 1),axis = 0)
    node_barcode_fmi = pd.DataFrame(index = nodes, columns = barcode_values, 
            data = np.sqrt(node_barcode_precision * node_barcode_recall))
    # Get clades
    barcode_clades = node_barcode_fmi.idxmax().to_frame(name = "node")
    barcode_clades["fmi"] = node_barcode_fmi.max(skipna=True)
    barcode_clades["n"] = node_barcode_counts.loc["root"]
    barcode_clades["group"] = barcode_clades.index.astype(str)
    return barcode_clades