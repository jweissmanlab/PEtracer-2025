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
import treedata as td
import pycea
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
from src.tree_utils import plot_grouped_characters

# Define constants
metric_names = {"fmi":"Clone Barcode FMI"}
param_names = {"characters":"Number of edit sites",
               "detection_rate":"Detection rate (%)"}

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

def clone_fmi_violin(plot_name,figsize = (2,2)):
    clone_fmi = pd.read_csv(results_path / "clone_fmi.csv")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.violinplot(data=clone_fmi.query("solver == 'upgma'"), x="permute", y="fmi",inner = None, 
                palette=discrete_cmap[2],saturation=1,linewidth =.5,bw = 1)
    sns.swarmplot(data=clone_fmi.query("solver == 'upgma'"), x="permute", y="fmi", color="black",size=2)
    plt.ylabel(metric_names["fmi"]);
    plt.xlabel("");
    plt.ylim(0,1.05);
    plt.yticks([0,0.5,1]);
    plt.xticks([0,1],["True\nbarcodes","Permuted\nbarcodes"]);
    save_plot(fig, plot_name, plots_path)

def clone_fmi_lineplot(plot_name,x,figsize = (2,2)):
    fmi = pd.read_csv(results_path / f"fmi_vs_{x}.csv")
    if x == "detection_rate":
        fmi[x] = fmi[x]*100
    fig, ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    sns.lineplot(data=fmi,x=x,y="fmi",ax=ax,hue = "clone",palette = discrete_cmap[6],legend = False,linewidth = 1.5)
    plt.ylim(0,1);
    ax.set_xlabel(param_names[x])
    ax.set_ylabel(metric_names["fmi"])

    save_plot(fig, plot_name, plots_path)

def nj_vs_upgma_fmi_scatterplot(plot_name,figsize = (2.5,2.5)):

    clone_fmi = pd.read_csv(results_path / "clone_fmi.csv")
    fmi_long = clone_fmi.query("~permute").melt(id_vars = ["clone","solver"],value_vars = ["blast_fmi","puro_fmi"])
    fmi_long = fmi_long.pivot_table(index = ["clone","variable"],columns = ["solver"],values = "value").reset_index()
    fmi_long["Barcode"] = fmi_long["variable"].map({"blast_fmi":"Blast","puro_fmi":"Puro"})

    fig, ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")

    sns.scatterplot(data = fmi_long,x = "nj",y = "upgma",markers = ["o","s"],legend=False,
                    hue = "clone",style = "Barcode",s = 40,palette = discrete_cmap[6]) 
    plt.plot([.7,1],[.7,1],color = "black",linestyle = "--",zorder = 0)
    plt.xlabel("NJ barcode FMI")
    plt.ylabel("UPGMA barcode FMI")

    save_plot(fig, plot_name, plots_path)

def polar_tree_with_clades(plot_name,clone,barcode,title = None,scale = False,figsize = (5,5)):
    tdata = td.read_h5ad(data_path / f"barcoding_clone_{clone}.h5td")
    clade_palette = {str(clade):color for clade, color in enumerate(colors[1:21]*100)}
    if scale is True:
        n_cells = len(tdata.obs)
        figsize = (figsize[0]*np.sqrt(n_cells/5000),figsize[1]*np.sqrt(n_cells/5000))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize,dpi = 600, layout = "constrained")
    pycea.pl.branches(tdata, depth_key = "time", polar = True,ax = ax,
                      color = f"{barcode}_clade",palette = clade_palette,linewidth=.3)
    if barcode in ["puro","combined"]:
        pycea.pl.nodes(tdata,color = "puro_lca",ax = ax,palette = clade_palette,style = "s",size = 15)
        pycea.pl.annotation(tdata,ax = ax,keys = ["puro"],palette=clade_palette)
    if barcode in ["blast","combined"]:
        pycea.pl.nodes(tdata,color = "blast_lca",ax = ax,palette = clade_palette,size = 10)
        pycea.pl.annotation(tdata,keys = ["blast"],gap = .02, palette = clade_palette)
    if title:
        ax.set_title(title) 
    save_plot(fig,plot_name,plots_path,transparent=True)

def tree_with_characters(plot_name,clone,figsize = (5,5)):
    tdata = td.read_h5ad(data_path / f"barcoding_clone_{clone}.h5td")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    pycea.pl.branches(tdata, depth_key = "time",ax = ax,linewidth=.3)
    plot_grouped_characters(tdata,ax = ax)
    save_plot(fig,plot_name,plots_path)

# Generate plots
if __name__ == "__main__":
    #site_edit_rates("site_edit_rates")
    #fmi_violin("fmi_violin")
    # FMI
    clone_fmi_violin("clone_fmi_violin",figsize = (2,2))
    nj_vs_upgma_fmi_scatterplot("nj_vs_upgma_fmi_scatterplot",figsize = (2,2))
    clone_fmi_lineplot("clone_fmi_vs_characters_lineplot",x = "characters",figsize = (2,2))
    clone_fmi_lineplot("clone_fmi_vs_detection_lineplot",x = "detection_rate",figsize = (2,2))
    # Clone trees
    #for clone in range(1,7):
    #    polar_tree_with_clades(f"clone_{clone}_combined_clades",clone,"combined",
    #        title = f"Clone {clone}",scale = True,figsize = (3.7,3.7))
    # Example clone
    example = 4
    #polar_tree_with_clades(f"clone_{example}_puro_clades",example,"puro",figsize = (4,4))
    #polar_tree_with_clades(f"clone_{example}_blast_clades",example,"blast",figsize = (4,4))
    #polar_tree_with_clades(f"clone_{example}_combined_clades",example,"combined",figsize = (4,4))
    #tree_with_characters(f"clone_{example}_with_characters",example,figsize = (3,2))
