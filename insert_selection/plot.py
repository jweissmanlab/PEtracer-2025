import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from scipy.cluster.hierarchy import linkage, leaves_list
from pathlib import Path

# Configure
results_path = Path(__file__).parent / "results"
plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

# Load source
from src.utils import save_plot
from src.config import colors, sequential_cmap, site_names

def crosshyb_heatmap(crosshyb,selected,subset = False,ax = None):
    """Plot the crosshyb heatmap"""
    order = crosshyb.index[leaves_list(linkage(np.log10(crosshyb), method='average'))]
    mask = np.triu(np.ones_like(crosshyb, dtype=bool),k = 1)
    if subset:
        order = [insert for insert in selected if insert in order]
        mask = mask[0:len(order),0:len(order)]
    sns.heatmap(np.log10(crosshyb.loc[order,order]), cmap=sequential_cmap, mask=mask, square=True,vmax = 0, vmin = -6,
            cbar = False,xticklabels=False,yticklabels=False,ax = ax)
    if not subset:
        for i, insert in enumerate(order):
            if insert in selected:
                ax.add_patch(plt.Rectangle((i - .25, i - .25), 1.5, 1.5, fill=False, edgecolor=colors[2], lw=1))
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    return ax

def site_crosshyb(plot_name,figsize = (6,2.2)):
    """Plot the crosshyb heatmap for each site"""
    fig, axs = plt.subplots(1,3,figsize = figsize,dpi = 600,layout = "constrained")
    inserts = pd.read_csv(results_path / "top_inserts.tsv",sep="\t")
    for i,site in enumerate(site_names.keys()):
        crosshyb = pd.read_csv(results_path / f"{site}_crosshyb.tsv",sep="\t",index_col=0)
        selected = inserts.query("site == @site & final_20")["insert"].values
        crosshyb_heatmap(crosshyb,selected,ax = axs[i])
        axs[i].set_title(site_names[site])
        if i == 0:
            axs[i].set_ylabel("5mer probes")
        axs[i].set_xlabel("5mer probes")
    save_plot(fig,plot_name,plots_path)

def selected_crosshyb(plot_name,site = "HEK3",figsize = (6,2.2)):
    """Plot selected crosshyb heatmap for each site"""
    inserts = pd.read_csv(results_path / "top_inserts.tsv",sep="\t")
    crosshyb = pd.read_csv(results_path / f"{site}_crosshyb.tsv",sep="\t",index_col=0)
    selected = inserts.query("site == @site & final_20")["insert"].values
    fig, ax = plt.subplots(1,1,figsize = figsize,dpi = 600,layout = "constrained")
    crosshyb_heatmap(crosshyb,selected,ax = ax,subset=True)
    save_plot(fig,plot_name,plots_path)

if __name__ == "__main__":
    site_crosshyb("site_crosshyb_heatmap")
    for site in site_names.keys():
        selected_crosshyb(f"{site}_20_crosshyb",site)
