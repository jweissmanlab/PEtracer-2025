""" Functions to generate plots for the kinetics experiments"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configure
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
ref_path = base_path / "reference"
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

# Load source
from src.config import colors,sequential_cmap,site_names,discrete_cmap,site_ids
from src.utils import save_plot
from src.legends import add_cbar

# Define constants
speed_palette = {"Other":"lightgray","WT":colors[0],"2-3 weeks":colors[1],"4-6 weeks":colors[2]}
speed_sizes = {"Other":1,"WT":2,"2-3 weeks":2,"4-6 weeks":2}

# Plotting functions
def edit_frac_lineplot(plot_name,site,cell_line = "4T1",min_cells = 25,agg = "mean",figsize = (2.5,2)):
    """Plot mean edit fraction over time as a line plot"""
    # Load data
    pegs = pd.read_csv(data_path / "pegRNAs.csv")
    cells = pd.read_csv(data_path / f"{cell_line}_kinetics_cells.csv",index_col=0)
    cells = cells.merge(pegs,on=["peg","peg_site"],how="left")   
    cells["speed"] = cells["speed"].fillna("Other") 
    # Get mean
    mean_edit_frac = cells.query("n_cells > @min_cells").groupby(["day","peg_site","peg","speed"]).agg({"edit_frac":agg}).reset_index()
    day0 = mean_edit_frac.groupby(["peg_site","peg","speed"]).first().reset_index().assign(day=0,edit_frac=0)
    mean_edit_frac = pd.concat([mean_edit_frac,day0])
    mean_edit_frac["edit_pct"] = mean_edit_frac["edit_frac"]*100
    # Plot
    fig,ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    sns.lineplot(data = mean_edit_frac.query("peg_site == @site"), x = "day",y = "edit_pct",size = "speed",
                 hue = "speed",style = "peg",legend=False, dashes=False,palette = speed_palette,sizes = speed_sizes)
    plt.xticks([0,7,14,21,28])
    plt.ylim(0,100)
    plt.xlabel("Day")
    plt.ylabel("Mean edit fraction (%)")
    save_plot(fig,plot_name,plots_path)


def speed_edit_frac_heatmap(plot_name,site,cell_line = "4T1",min_cells = 25,figsize = (2.5,2)):
    """Plot heatmap of cell edit fractions over time for different speeds"""
    # Load data
    pegs = pd.read_csv(data_path / "pegRNAs.csv")
    cells = pd.read_csv(data_path / f"{cell_line}_kinetics_cells.csv",index_col=0)
    cells = cells.merge(pegs,on=["peg","peg_site"],how="left")
    # Plot
    fig, axes = plt.subplots(1,3,figsize = (2.5,2),dpi = 600,layout = "constrained")
    for i, speed in enumerate(["WT","2-3 weeks","4-6 weeks"]):
        np.random.seed(0)
        ax = axes[i]
        selected_cells = cells.query("speed == @speed & peg_site == @site & n_cells >= @min_cells")
        selected_cells = selected_cells.groupby(["day"]).apply(lambda x: x.sample(min_cells), include_groups=False).reset_index()
        selected_cells = selected_cells.sort_values("edit_frac",ascending = False)
        selected_cells["rank"] = selected_cells.groupby(["day"]).cumcount()
        frac_mat = selected_cells.pivot(index = ["day"],columns = "rank",values = "edit_frac")
        sns.heatmap(frac_mat.T,cmap = sequential_cmap,ax = ax,yticklabels = False,cbar = False)
        ax.set_title(speed)
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.set_xticks(np.array(range(frac_mat.shape[0])) + .5 ,labels = frac_mat.index.to_list())
        plt.setp(ax.get_xticklabels(), rotation=90)
    save_plot(fig,plot_name,plots_path)


def edit_rate_scatterplot(plot_name,log=True,figsize = (2,2)):
    """Plot scatterplot of editing rates between 4T1 and B16F10"""
    # Load data
    rates = pd.read_csv(results_path / "edit_rates.csv")
    rates["site"] = rates["peg_site"].apply(lambda x: site_ids[x]) 
    # Plot
    fig, ax = plt.subplots(figsize=figsize,layout = "constrained",dpi=600)
    g = sns.scatterplot(data=rates.sort_values("site"),x="4T1_rate",y="B16F10_rate",
        hue = "site",palette=discrete_cmap[3],s = 25)
    g.legend(title="Edit site",loc = "lower right")
    ax.plot([0,.6],[0,0.6],color="black",linestyle="--",zorder=0)
    plt.xlabel("4T1 rate (edits/day)")
    plt.ylabel("B16F10 rate (edits/day)")
    if log:
        plt.xscale("log")
        plt.yscale("log")
    save_plot(fig,plot_name,plots_path)


def variant_rate_heatmap(plot_name,log=True,vmin = -3,vmax = 0,figsize = (4,2.5)):
    """Plot heatmap of editing rates for each base and position"""
    # Load data
    rates = pd.read_csv(results_path / "edit_rates.csv")
    # Plot
    fig, axes = plt.subplots(3,1,figsize=(4,2.5),dpi=600,layout = "constrained")
    for i, site in enumerate(site_names.keys()):
        ax = axes[i]
        pos_rate = rates.query("peg_site == @site").pivot_table(index="base",columns="pos",values="mean_rate")
        pos_rate.columns = pos_rate.columns.astype(int)
        if log:
            pos_rate = pos_rate.map(lambda x: np.log10(x))
            sns.heatmap(pos_rate,cmap="viridis",ax=ax,cbar=False,vmin = vmin,vmax = vmax)
        else:
            sns.heatmap(pos_rate,cmap="viridis",ax=ax,cbar=False,vmin = 0,vmax = 0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i == 2:
            ax.set_xticks(np.arange(pos_rate.shape[1]) + 0.5,pos_rate.columns,rotation=90)
        else:
            ax.set_xticks([])
        ax.set_yticks(np.arange(pos_rate.shape[0]) + 0.5,pos_rate.index,rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_title(site_names[site],size=10)
    # Add colorbar
    cbar_ax = fig.add_axes([1.05, 0.25, 0.03, 0.45])
    if log:
        if vmin == -3:
            add_cbar(cbar_ax,"viridis",[0,-1,-2,-3],"Edit rate (edits/day)",
                ticklabels=["$10^{0}$","$10^{-1}$","$10^{-2}$","$10^{-3}$"])
        if vmin == -2:
            add_cbar(cbar_ax,"viridis",[0,-1,-2],"Edit rate (edits/day)",
                ticklabels=["$10^{0}$","$10^{-1}$","$10^{-2}$"])
    else:
        add_cbar(cbar_ax,"viridis",[0,.1,.2,.3,.4,.5],"Edit rate (edits/day)")
    save_plot(fig,plot_name,plots_path)

# Generate plots
if __name__ == "__main__":
    for cell_line in ["4T1","B16F10"]:
        for site in site_names.keys():
            edit_frac_lineplot(f"{cell_line}_{site}_mean_edit_frac",site,cell_line=cell_line,agg = "mean",min_cells=20)
            speed_edit_frac_heatmap(f"{cell_line}_{site}_edit_frac_vs_speed",site,cell_line=cell_line,min_cells=20)
    edit_rate_scatterplot("variant_log_rate_scatterplot",log=True,figsize = (2.5,2.5))
    edit_rate_scatterplot("variant_rate_scatterplot",log=False,figsize = (2.5,2.5))
    variant_rate_heatmap("variant_log_rate_heatmap",log=True,figsize = (4,2.5))
    variant_rate_heatmap("variant_clipped_log_rate_heatmap",vmin = -2,log=True,figsize = (4,2.5))
    variant_rate_heatmap("variant_rate_heatmap",log = False,figsize = (4,2.5))
