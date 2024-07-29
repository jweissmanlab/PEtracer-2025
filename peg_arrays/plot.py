"""Code to generate peg array plots."""

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import multiprocessing as mp

# Configure
results_path = Path(__file__).parent / "results"
plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
ref_path = base_path / "reference"
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

from src.config import discrete_cmap,edit_ids,site_ids
from src.utils import save_plot

## Define constants
site_colors = {"RNF2":"#B3E6FF","HEK3":"#99A3FF","EMX1":"#66FFFF"}

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

def load_edit_fracs():
    alleles = pd.read_csv(results_path / "peg_array_allele_counts.csv",keep_default_na=False)
    edit_fracs = alleles.query("~edit.isin(['None','Other'])").copy()
    edit_fracs["totalCount"] = edit_fracs.groupby(["sample","site"])["readCount"].transform("sum") 
    edit_fracs["edit_frac"] = (edit_fracs["readCount"] / edit_fracs["totalCount"]) * 100
    return edit_fracs

## Plotting functions
def array_installation_barplot(plot_name,edit_fracs,site,version,figsize):
    fig, axes = plt.subplots(1,3,figsize=figsize,dpi = 600, layout = "constrained",sharey=True)
    for i in range(1,4):
        ax = axes[i-1]
        array_fracs = edit_fracs.query(f"site == '{site}' & version == '{version}' & array == 'array{i}'").copy()
        array_fracs = array_fracs.sort_values("position")
        entropy = normalized_entropy(array_fracs.groupby("edit")["edit_frac"].mean().values/100)
        sns.barplot(data = array_fracs,x = "edit",y = "edit_frac",ax = ax,errorbar=None,color = site_colors[site], edgecolor='black', linewidth=0.5)
        sns.scatterplot(data = array_fracs,x = "edit",y = "edit_frac",ax = ax,color = "black",s=5,linewidth=0)
        ax.text(0.5,0.95,f"$H_{{norm}}$={entropy:.2f}",transform=ax.transAxes,ha="center",va="center")
        for item in ax.get_xticklabels():
            item.set_rotation(90)
        ax.set_xlabel("")
        ax.tick_params(axis='x', length=0)
        ax.tick_params(axis='y', length=2)
        ax.set_ylim(0,35)
        ax.set_yticks([0,10,20,30])
        ax.tick_params(axis='x', labelsize=9)
        if i == 1:
            ax.set_ylabel("Norm. LM installation (%)")
        else:
            ax.set_ylabel("")
        ax.set_title(f"Array {i}",fontsize=10)
    fig.get_layout_engine().set(wspace=0.04,w_pad=0)
    save_plot(fig,plot_name,plots_path)

def final_array_installation_barplot(plot_name,edit_fracs,site,figsize = (1.3,1.9)): 
    fig, ax = plt.subplots(figsize=(1.3,1.9),dpi = 600)
    site_edit_frac = edit_fracs.query("site == @site & array == '24mer'").copy()
    site_edit_frac = site_edit_frac.sort_values("position")
    entropy = normalized_entropy(site_edit_frac.groupby("edit")["edit_frac"].mean().values/100)
    sns.barplot(data=site_edit_frac, x="edit", y='edit_frac',hue = "position",palette=discrete_cmap[8],
        errorbar=None,saturation = 1,legend=False, edgecolor='black', linewidth=0.5)
    sns.scatterplot(data=site_edit_frac,
                    x='edit', y='edit_frac',color="black",s=5,linewidth=0)
    plt.text(0.5, 0.95, f"$H_{{norm}}$ = {entropy:.2f}", ha='center', va='center', transform=ax.transAxes)
    plt.xticks(rotation=90);
    ax.tick_params(axis='x', which='both', length=0)
    plt.ylim(0,25)
    plt.ylabel("Norm. LM installation (%)")
    plt.xlabel("")
    save_plot(fig, plot_name, plots_path)

# Generate plots
if __name__ == "__main__":
    # Load peg array edit fracs
    edit_fracs = load_edit_fracs()
    for site in site_ids.keys():
        for version in ["v1","v2"]:
            array_installation_barplot(f"8mer_{version}_{site}_installation_barplot",edit_fracs,site,version,(3,2))
        final_array_installation_barplot(f"24mer_{site}_installation_barplot",edit_fracs,site)



