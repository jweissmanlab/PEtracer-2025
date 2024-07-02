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
from src.config import colors,sequential_cmap,site_names
from src.utils import save_plot

# Define constants
speed_palette = {"Other":"lightgray","WT":colors[0],"2-3 weeks":colors[1],"4-6 weeks":colors[2]}
speed_sizes = {"Other":1,"WT":2,"2-3 weeks":2,"4-6 weeks":2}

def edit_frac_lineplot(plot_name,site,cell_line = "4T1",min_cells = 25,agg = "mean",figsize = (2.5,2)):
    # Load data
    speeds = pd.read_csv(results_path / "speeds.csv")
    cells = pd.read_csv(data_path / f"{cell_line}_kinetics_cells.csv",index_col=0)
    cells = cells.merge(speeds,on=["peg","peg_site"],how="left")   
    cells["speed"] = cells["speed"].fillna("Other") 
    # Get median
    median_edit_frac = cells.query("n_cells > @min_cells").groupby(["day","peg_site","peg","speed"]).agg({"edit_frac":agg}).reset_index()
    day0 = median_edit_frac.groupby(["peg_site","peg","speed"]).first().reset_index().assign(day=0,edit_frac=0)
    median_edit_frac = pd.concat([median_edit_frac,day0])
    median_edit_frac["edit_pct"] = median_edit_frac["edit_frac"]*100
    # Plot
    fig,ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    sns.lineplot(data = median_edit_frac.query("peg_site == @site"), x = "day",y = "edit_pct",size = "speed",
                 hue = "speed",style = "peg",legend=False, dashes=False,palette = speed_palette,sizes = speed_sizes)
    plt.xticks([0,7,14,21,28])
    plt.ylim(0,100)
    plt.xlabel("Day")
    plt.ylabel("Mean edit fraction (%)")
    save_plot(fig,plot_name,plots_path)

def speed_edit_frac_heatmap(plot_name,site,cell_line = "4T1",min_cells = 25,figsize = (2.5,2)):
    # Load data
    speeds = pd.read_csv(results_path / "speeds.csv")
    cells = pd.read_csv(data_path / f"{cell_line}_kinetics_cells.csv",index_col=0)
    cells = cells.merge(speeds,on=["peg","peg_site"],how="left")
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

# Generate plots
if __name__ == "__main__":
    for cell_line in ["4T1","B16F10"]:
        for site in site_names.keys():
            edit_frac_lineplot(f"{cell_line}_{site}_mean_edit_frac",site,cell_line=cell_line,agg = "mean",min_cells=20)
            speed_edit_frac_heatmap(f"{cell_line}_{site}_edit_frac_vs_speed",site,cell_line=cell_line,min_cells=20)
