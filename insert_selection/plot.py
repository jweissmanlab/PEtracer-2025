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
from src.config import colors, sequential_cmap, site_names, discrete_cmap, edit_ids, edit_palette

def crosshyb_vs_length_lineplot(plot_name,y,y_label,figsize = (2,2)):
    results = pd.read_csv(results_path / "crosshyb_vs_length.csv")
    results["correct_pct"] = results["correct_frac"] * 100
    fig, ax = plt.subplots(figsize=figsize, dpi=600,layout="constrained")
    sns.lineplot(data=results.query("length > 1"),x="length",y=y,color = colors[1],ax=ax)
    ax.set_xlabel("Insert length")
    ax.set_ylabel(y_label)
    plt.xticks(range(2,9));
    save_plot(fig, plot_name, plots_path)

def crosshyb_heatmap(plot_name = None,site = "HEK3", metric = "free_energy",subset = None,highlight = None,
    lower = True,vmax = -12,vmin = -25,ticklabels = False,ax = None,figsize = (2,2)):
    """Plot the crosshyb heatmap"""
    # Load crosshyb for site
    crosshyb = pd.read_csv(results_path / "top_insert_crosshyb.csv",keep_default_na=False).query("site == @site")
    if metric == "probe_frac":
        crosshyb["probe_frac"] = np.log10(crosshyb["probe_frac"])
    crosshyb = crosshyb.pivot(index = "probe",columns="target",values=metric)
    order = crosshyb.index[leaves_list(linkage(crosshyb, method='average'))]
    # Select inserts
    inserts = pd.read_csv(results_path / "top_inserts.csv",keep_default_na=False)
    if subset is not None:
        selected = inserts.query(f"site == @site & {subset}")["insert"].tolist()
        order = [insert for insert in order if insert in selected]
    if subset == "final_8":
        order = list(edit_ids[site].keys())
    crosshyb = crosshyb.loc[order,order]
    # Plot heatmap
    if lower:
        mask = np.triu(np.ones_like(crosshyb, dtype=bool),k = 1)
    else:
        mask = np.tril(np.ones_like(crosshyb, dtype=bool),k = -1) 
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,dpi=600,layout = "constrained")
    cmap = sequential_cmap.reversed() if metric == "free_energy_diff" else sequential_cmap
    sns.heatmap(crosshyb, cmap=cmap, mask=mask, square=True,vmax=vmax,vmin=vmin,
            cbar = False,xticklabels=ticklabels,yticklabels=ticklabels,ax = ax)
    # Highlight selected inserts
    if highlight is not None:
        if subset == "final_8":
            selected = inserts.query(f"site == @site & {highlight}")["insert"].tolist()
            for i, insert in enumerate(order):
                if insert in selected:
                    color = edit_palette[str(edit_ids[site][insert])]
                    ax.add_patch(plt.Rectangle((i + .1, i + .1), .8, .8, fill=False, edgecolor=color, lw=2))
        else:
            ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
            ax.set_ylim(ax.get_ylim()[0] + 1, ax.get_ylim()[1] - 1) 
            selected = inserts.query(f"site == @site & {highlight}")["insert"].tolist()
            for i, insert in enumerate(order):
                if insert in selected:
                    ax.add_patch(plt.Rectangle((i - .1, i - .1), 1.2, 1.2, fill=False, edgecolor=colors[2], lw=1))
    # Format plot
    plt.xlabel("DNA with LMs")
    plt.ylabel("LM probes")
    if lower:
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
    else:
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
    if plot_name is not None:
        save_plot(fig, plot_name, plots_path)
        pass
    else:
        return ax

if __name__ == "__main__":
    crosshyb_vs_length_lineplot("free_energy_diff_vs_length","free_energy_diff","$\Delta G$ - on-target $\Delta G$")
    crosshyb_vs_length_lineplot("correct_frac_vs_length","correct_pct","Correct probe bound (%)")
    vmax =8
    vmin = 0
    metric = "free_energy_diff"
    for site in site_names.keys():
        crosshyb_heatmap(f"{site}_free_energy_diff",site,metric=metric,highlight="final_20",
                        lower=True,figsize = (2,2),vmax = vmax,vmin = vmin)
        crosshyb_heatmap(f"{site}_20_free_energy_diff",site,metric=metric,subset="final_20",
                        lower=False,figsize = (2,2),vmax = vmax,vmin = vmin)
        crosshyb_heatmap(f"{site}_8_free_energy_diff",site,metric=metric,subset="final_8",highlight="final_8",
                        lower=True,figsize = (1.2,1.2),vmax = vmax,vmin = vmin,ticklabels=False)
