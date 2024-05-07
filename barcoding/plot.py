import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.patches as mpatches
from pathlib import Path
import cassiopeia as cas
import pickle
import multiprocessing as mp

# Configure
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
ref_path = base_path / "reference"
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

# Load source
from src.config import colors,discrete_cmap,threads,site_names,edit_ids,edit_names
from src.utils import save_plot

def site_edit_rates(plot_name):
    edit_counts = pd.read_csv(results_path / "tree_edit_counts.csv")

    site_edit_rate = edit_counts.groupby(["site","clone"]).agg({"n":"sum","n_edges":"first"}).reset_index()
    site_edit_rate["rate"] = site_edit_rate["n"]/site_edit_rate["n_edges"]
    site_edit_rate["site"] = site_edit_rate["site"].map(site_names).str.split(" ").str[-1]
    site_edit_rate = site_edit_rate.sort_values("site")

    fig, ax = plt.subplots(figsize=(1.5, 1.8),dpi = 300)

    sns.barplot(data = site_edit_rate, y = "rate",x = "site",hue = "site",palette = discrete_cmap[3],
        dodge = False, errwidth=2,errcolor = "black",saturation = 1)
    plt.ylabel("Edits per branch")
    plt.xlabel("Edit site")
    plt.gca().get_legend().remove()

    save_plot(fig, plot_name, plots_path)

def fmi_violin(plot_name):

    stats = pd.read_csv(results_path / "tree_stats.csv")
    stats_shuffled = pd.read_csv(results_path / "tree_stats_shuffled.csv")
    stats_with_shuffled = pd.concat([stats,stats_shuffled])

    fig, ax = plt.subplots(figsize=(1.5, 1.8),dpi = 300)

    sns.violinplot(data=stats_with_shuffled.query("solver == 'nj'"), x="shuffled", y="fmi",inner = None, 
                palette=discrete_cmap[2],saturation=1,linewidth =.5,bw = 1)
    sns.swarmplot(data=stats_with_shuffled.query("solver == 'nj'"), x="shuffled", y="fmi", color="black",size=2)
    plt.ylabel("Average barcode FMI")
    plt.xlabel("")
    plt.ylim(0,1.05)
    plt.xticks([0,1],["Clones","Permuted\nbarcodes"])

    save_plot(fig, plot_name, plots_path)

def nj_vs_upgma_fmi(plot_name):

    stats = pd.read_csv(results_path / "tree_stats.csv")
    stats_long = stats.melt(id_vars = ["clone","solver","shuffled"],value_vars = ["blast_fmi","puro_fmi"])
    stats_long = stats_long.pivot_table(index = ["clone","variable"],columns = ["solver"],values = "value").reset_index()
    stats_long["Barcode"] = stats_long["variable"].map({"blast_fmi":"Blast","puro_fmi":"Puro"})

    fig, ax = plt.subplots(figsize=(2.5, 2.5),dpi = 300)

    sns.scatterplot(data = stats_long.rename(columns={"clone":"Clone"}),x = "nj",y = "upgma",
                    hue = "Clone",style = "Barcode",s = 60,palette = discrete_cmap[6]) 
    plt.plot([.7,1],[.7,1],color = "black",linestyle = "--",zorder = 0)
    plt.xlabel("Neighbor Joining barcode FMI")
    plt.ylabel("UPGMA barcode FMI")

    save_plot(fig, plot_name, plots_path)

def tree_with_clades(clone,solver,figsize,scale,title,plot_name):

    with open(results_path / "trees.pkl", "rb") as f:
        trees = pickle.load(f)
    tree = trees[solver][clone]
    clades = pd.read_csv(results_path / "tree_clades.csv",dtype={"clade":str})
    tree_clades = clades.query("solver == 'nj' & clone == @clone").copy()

    cmap = discrete_cmap[19].copy()
    cmap.insert(0,"white")
    cmap = mcolors.ListedColormap(cmap)
    tree_clades["vmap"] = tree_clades["clade"].astype(int) % 18 + 1
    tree_clades["color"] = tree_clades["vmap"].map(cmap)
    vmap = tree_clades.set_index("clade")["vmap"].to_dict()
    vmap.update({"None":0})
    nodes_sizes = {node:20 for node in tree_clades.query("n >= 3").max_node}
    clade_colors = tree_clades.set_index("max_node")["color"].to_dict()

    if scale:
        n_cells = len(tree.cell_meta)
        figsize = (figsize[0]*np.sqrt(n_cells/5000),figsize[1]*np.sqrt(n_cells/5000))

    fig, ax = plt.subplots(figsize=figsize,dpi = 300)

    cas.pl.plot_matplotlib(tree, meta_data=["puroGrp","blastGrp"],categorical_cmap = cmap,
                            value_mapping = vmap,ax=ax,clade_colors=clade_colors,node_sizes=nodes_sizes,branch_kwargs={"linewidth":.5,"c":"gray"},leaf_kwargs={"s":0})
    if title:
        plt.title(title, y = .95)
    save_plot(fig,plot_name,plots_path,transparent=True)

def tree_with_edits(clone,solver,figsize,plot_name):

    with open(results_path / "trees.pkl", "rb") as f:
        trees = pickle.load(f)
    allele_table = pd.read_csv(data_path / "barcoded_4T1_alleles.tsv",sep="\t",
                            keep_default_na=False,index_col=0)
    
    tree = trees[solver][clone]
    tree_allele_table = allele_table[allele_table.cellBC.isin(tree.cell_meta.cellBC)].copy()
    for site in list(site_names.keys()):
        tree_allele_table[site] = tree_allele_table[site].map(edit_ids[site]).fillna(9).astype(int).astype(str)
    tree_allele_table = tree_allele_table.rename(columns={"RNF2":"r1","HEK3":"r2","EMX1":"r3","intID":"intBC"})
    edit_colors = [(.8, .8, .8)] + discrete_cmap[8].copy() + [(.3, .3, .3)]
    edit_colors_df = pd.DataFrame({"color":[mcolors.rgb_to_hsv(i) for i in edit_colors]},
            index=[str(i) for i in range(0, 10)])

    fig, ax = plt.subplots(figsize=figsize,dpi = 300,layout="constrained")

    cas.pl.plot_matplotlib(tree, orient="right",allele_table=tree_allele_table,indel_colors=edit_colors_df,
        ax = ax,leaf_kwargs={"s":0},branch_kwargs={"linewidth":.5})

    edit_labels = [name for name in edit_names["EMX1"].values()] + ["Other"]
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, color=color, label=label)
                    for color, label in zip(edit_colors,edit_labels)]
    fig.legend(handles=legend_handles,loc='lower left',bbox_to_anchor=(.94,.04),ncol=1)
    save_plot(fig,plot_name,plots_path)

# Generate plots
if __name__ == "__main__":
    site_edit_rates("site_edit_rates")
    fmi_violin("fmi_violin")
    #nj_vs_upgma_fmi("nj_vs_upgma_fmi")
    #params = [(clone, "nj", (6,6), True, f"Clone {clone}",
    #           f"clone_{clone}_tree") for clone in range(1,7)]
    #with mp.Pool(processes=threads) as pool:
    #    results = pool.starmap(tree_with_clades, params)
    #tree_with_clades(5,"nj",(6.5,6.5),False,None,"example_tree")
    #tree_with_edits(5,"nj",(3.2,2.4),"example_tree_with_edits")

