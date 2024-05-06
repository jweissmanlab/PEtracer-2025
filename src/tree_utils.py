import numpy as np
import pandas as pd
import cassiopeia as cas
import networkx as nx

def edit_frac(tree):
    return np.float64(np.apply_over_axes(np.sum,tree.character_matrix != 0,(0,1))/
                      np.apply_over_axes(np.sum,~np.isnan(tree.character_matrix),(0,1)))

def clade_barcode_stats(tree,barcodes,missing = "None"):
    # Setup networkx tree to keep track of barcode precision and recall
    nx_tree = tree.get_tree_topology()
    barcode_stats = {}
    for barcode in barcodes:
        count = tree.cell_meta[barcode].value_counts()
        if missing in count.keys():
            del count[missing]
        barcode_stats[barcode] = pd.DataFrame({'n': count, 'index': range(len(count)), 
                                                'max_fmi': np.zeros(len(count)), 'max_node': "None"})
        for node in nx_tree.nodes:
            nx_tree.nodes[node][barcode] = {"n":np.zeros(len(count)),"precision":np.zeros(len(count)),
                                            "recall":np.zeros(len(count)),"fmi":np.zeros(len(count))}
    # Use metadata to assign barcodes to nodes
    for node in tree.leaves:
        for barcode in barcode_stats.keys():
            value = tree.cell_meta.loc[node,barcode]
            if value != missing:
                index = barcode_stats[barcode].loc[value,"index"]
                nx_tree.nodes[node][barcode]["n"][index] += 1
    # Calculate precision, recall, and fmi for each node
    layers = dict(enumerate(nx.bfs_layers(nx_tree,tree.root)))
    bfs_postorder = [node for layer in reversed(layers.values()) for node in layer]
    for node in bfs_postorder:
        for barcode in barcode_stats.keys():
            for child in nx_tree.successors(node):
                nx_tree.nodes[node][barcode]["n"] += nx_tree.nodes[child][barcode]["n"]
            if np.sum(nx_tree.nodes[node][barcode]["n"]) > 0:
                precision = nx_tree.nodes[node][barcode]["n"]/np.sum(nx_tree.nodes[node][barcode]["n"])
                recall = nx_tree.nodes[node][barcode]["n"]/barcode_stats[barcode]["n"].to_numpy()
                nx_tree.nodes[node][barcode]["precision"] = precision
                nx_tree.nodes[node][barcode]["recall"] = recall
                nx_tree.nodes[node][barcode]["fmi"] = np.sqrt(precision*recall)
    # For each barcode value find the node with the highest fmi
    for node in nx_tree.nodes:
        for barcode in barcodes:
            for index, row in barcode_stats[barcode].iterrows():
                fmi = nx_tree.nodes[node][barcode]["fmi"][row["index"]]
                if fmi > row['max_fmi']:
                    barcode_stats[barcode].loc[index,'max_fmi'] = fmi
                    barcode_stats[barcode].loc[index,'max_node'] = node
    return barcode_stats