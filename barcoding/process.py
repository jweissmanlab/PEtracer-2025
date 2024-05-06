"""Code for processing pre-edited validatation data."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import cassiopeia as cas
import multiprocessing as mp
import pickle

# Configure paths
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

# Load source
from src.config import threads
from src.tree_utils import edit_frac, clade_barcode_stats

# Solvers
solvers = {"nj":cas.solver.NeighborJoiningSolver(cas.solver.dissimilarity.weighted_hamming_distance,
                                                 add_root=True,fast = True),
           "upgma":cas.solver.UPGMASolver(cas.solver.dissimilarity.weighted_hamming_distance,fast = True)}

# Reconstruct trees for each clone
def reconstruct_clone(clone, solver):
    allele_table_clone = allele_table[allele_table["clone"] == clone]
    character_matrix, _, _ = cas.pp.convert_alleletable_to_character_matrix(allele_table_clone)
    tree = cas.data.CassiopeiaTree(character_matrix=character_matrix)
    tree.cell_meta = cell_meta[cell_meta["clone"] == clone]
    solvers[solver].solve(tree, collapse_mutationless_edges=True)
    tree.reconstruct_ancestral_characters()
    return clone, solver, tree

def reconstruct_trees(allele_table, cell_meta):
    clones = cell_meta["clone"].unique()
    params = [(clone, solver) for clone in clones for solver in solvers.keys()]
    with mp.Pool(processes=threads) as pool:
        results = pool.starmap(reconstruct_clone, params)
    trees = {"nj": {}, "upgma": {}}
    for clone, solver, tree in results:
        trees[solver][clone] = tree
    return trees

# Evaluate trees for each clone
def evaluate_clone(clone, solver, shuffle=False,barcodes = ["puroGrp","blastGrp"]):
    tree = trees[solver][clone].copy()
    stats = pd.DataFrame({"edit_frac": edit_frac(tree),"n_characters": tree.n_character, 
                        "clone":clone,"solver":solver,"shuffled":shuffle}, index=[0])
    if shuffle:
        tree.cell_meta["blastGrp"] = np.random.permutation(tree.cell_meta["blastGrp"])
        tree.cell_meta["puroGrp"] = np.random.permutation(tree.cell_meta["puroGrp"])
    barcode_fmi = clade_barcode_stats(tree, barcodes, missing="None")
    clades = []
    for barcode in barcodes:
        barcode_clades = barcode_fmi[barcode].reset_index().rename(columns={barcode: "clade"}).drop(columns=["index"])
        barcode_clades["barcode"] = barcode.replace('Grp','')
        stats[f"{barcode.replace('Grp','')}_fmi"] = np.average(barcode_clades["max_fmi"], weights=barcode_clades["n"])
        clades.append(barcode_clades)
    clades = pd.concat(clades)
    clades["clone"] = clone
    clades["solver"] = solver
    stats["fmi"] = np.average(clades["max_fmi"], weights=clades["n"])
    return stats, clades

def evaluate_trees(trees):
    # Evaluate trees
    clones = list(trees["nj"].keys())
    params = [(clone, solver) for clone in clones for solver in solvers.keys()]
    with mp.Pool(processes=threads) as pool:
        results = pool.starmap(evaluate_clone, params)
    stats = pd.concat([stats for stats, _ in results])
    clades = pd.concat([clades for _, clades in results]) 
    # Evaluate trees with shuffled barcodes
    params = [(clone, "nj", True) for clone in clones] * 10
    with mp.Pool(processes=threads) as pool:
        results = pool.starmap(evaluate_clone, params)
    stats_shuffled = pd.concat([stats for stats, _ in results])
    return clades, stats, stats_shuffled

# Get edit counts
def count_clone_edits(clone, solver):
    # Reconstruct tree
    allele_table_clone = allele_table[allele_table["clone"] == clone]
    character_matrix, _,  edit_dict = cas.pp.convert_alleletable_to_character_matrix(allele_table_clone)
    tree = cas.data.CassiopeiaTree(character_matrix=character_matrix)
    solvers[solver].solve(tree, collapse_mutationless_edges=False)
    tree.reconstruct_ancestral_characters()
    # Get branches
    parent_states = []
    states = []
    for edge in tree.depth_first_traverse_edges():
        parent_states.append(tree.get_character_states(edge[0]))
        states.append(tree.get_character_states(edge[1]))
    parent_states = pd.DataFrame(parent_states,columns = np.arange(tree.n_character))
    states = pd.DataFrame(states,columns = np.arange(tree.n_character))
    edges = pd.melt(parent_states,var_name = "site",value_name = "parent_state")
    edges["state"] = pd.melt(states,var_name = "site",value_name = "state")["state"]
    # Get edits
    edits = edges.query("parent_state == 0 & state != -1").copy()
    edits["edit"] = edits.apply(lambda x: edit_dict[x["site"]].get(x["state"],"None"),axis = 1)
    edits["site"] = edits["site"].apply(lambda x: ["RNF2","HEK3","EMX1"][x%3])
    edits["clone"] = clone
    # Get edit counts
    edit_counts = edits.groupby(["site", "edit"]).size().reset_index(name="n")
    edit_counts['n_edges'] = edit_counts.groupby('site')['n'].transform('sum')
    edit_counts["clone"] = clone
    return edit_counts.query("edit != 'None'")

def count_edits(allele_table):
    clones = allele_table["clone"].unique()
    params = [(clone, "nj") for clone in clones]
    with mp.Pool(processes=threads) as pool:
        results = pool.starmap(count_clone_edits, params)
    edit_counts = pd.concat([counts for counts in results])
    return edit_counts

# Process barcoding data
if __name__ == "__main__":

    # Load cell metadata
    cell_meta = pd.read_csv(data_path / "barcoded_4T1_cell_meta.tsv",sep="\t",
                            dtype={"puroGrp":str,"blastGrp":str}).fillna("None")
    cell_meta = cell_meta[cell_meta["type"] == "normal"]
    cell_meta["clone"] = cell_meta.clone.astype(int)
    cell_meta.index = cell_meta["cellBC"].values

    # Load allele table
    allele_table = pd.read_csv(data_path / "barcoded_4T1_alleles.tsv",sep="\t",
                            keep_default_na=False,index_col=0)
    allele_table = allele_table[allele_table["cellBC"].isin(cell_meta.cellBC)]
    allele_table = allele_table.rename(columns={"RNF2":"r1","HEK3":"r2","EMX1":"r3","intID":"intBC"})

    # Reconstruct trees
    print("Reconstructing trees...")
    '''
    trees = reconstruct_trees(allele_table, cell_meta)
    with open(results_path / "trees.pkl", "wb") as f:
        pickle.dump(trees, f)
    '''

    # Evaluate trees
    print("Evaluating trees...")
    '''
    clades, stats, stats_shuffled = evaluate_trees(trees)
    clades.to_csv(results_path / "tree_clades.csv",index = False)
    stats.to_csv(results_path / "tree_stats.csv",index = False)
    stats_shuffled.to_csv(results_path / "tree_stats_shuffled.csv",index = False)
    '''

    # Get edit counts
    print("Getting edit counts...")
    edit_counts = count_edits(allele_table)
    edit_counts.to_csv(results_path / "tree_edit_counts.csv",index = False)