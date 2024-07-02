import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from pathlib import Path

# Configure
results_path = Path(__file__).parent / "results"
plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

# Load source
from src.utils import save_plot
from src.config import colors, sequential_cmap

# Define constants
metric_names = {"rf":"Robinson-Foulds distance",
           "triplets":"Mean triplets correct"}
solver_names = {"upgma":"UPGMA",
            "nj":"NJ",
            "greedy":"Greedy"}
param_names = {"size":"Number of cells",
          "edit_frac":"Edit sites with LM (%)",
          "characters":"Number of edit sites",
          "detection_rate":"Detection rate (%)",
          "states":"Number of LMs",
          "entropy":"LM distribution ($H_{norm}$)",}
param_defaults = {"size":1000,
                 "edit_frac":70,
                 "characters":60,
                 "detection_rate":90}
solver_colors = colors[1:3] + [colors[4]]
solver_colors = {solver: solver_colors[i] for i, solver in enumerate(solver_names.keys())}


# Heatmap comparing two parameters
def metric_heatmap(data,x,y,metric = "rf",vmin = 0,vmax = 1,figsize = (2.2,2.2)):
    data = data.copy()
    data["edit_frac"] = (data["edit_frac"] * 100).astype(int)
    data["detection_rate"] = (100 - data["missing_rate"] * 100).astype(int)
    data = data.pivot_table(index = y,columns = x,values = metric)
    data = data.sort_index(ascending = False)
    fig, ax = plt.subplots(figsize = figsize)
    cmap = sequential_cmap.reversed() if metric == "rf" else sequential_cmap
    sns.heatmap(data,cmap = cmap,annot = True,cbar=False,
                fmt = ".2f",ax = ax,annot_kws={"size": 9},vmax = vmax,vmin = vmin)
    plt.yticks(rotation=0)
    ax.set_xlabel(param_names[x].replace("\n"," "))
    ax.set_ylabel(param_names[y])
    save_plot(fig, f"{metric}_heatmap_{x}_vs_{y}", plots_path)

# Parameter sweep line plots
def parameter_lineplots(data,metric,params = ["size","edit_frac","characters","detection_rate"]):
    data = data.query("solver.isin(@solver_names.keys())").copy()
    data["edit_frac"] = (data["edit_frac"] * 100).astype(int)
    data["detection_rate"] = (100 - data["missing_rate"] * 100).astype(int)
    metric_min = data[metric].min()
    metric_max = data[metric].max()
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.2),layout = "constrained")
    for i, param in enumerate(params):
        param_data = data.copy()
        for var_param in param_defaults.keys():
            if var_param != param:
                param_data = param_data[param_data[var_param] == param_defaults[var_param]]
        sns.lineplot(x=param, y=metric, hue="solver", style="indel_dist", data=param_data,
                    palette=solver_colors , ax=axes[i], legend=False, markers=False, markersize=8,)
        mean_data = param_data.groupby([param, "solver","indel_dist"]).agg({metric: "mean"}).reset_index()
        sns.scatterplot(x=param, y=metric, hue="solver", data=mean_data.query("~indel_dist"),edgecolor = "black",
                    palette=solver_colors, ax=axes[i], marker ="o", legend=False, s=30, zorder=4,linewidth = .5)
        sns.scatterplot(x=param, y=metric, hue="solver", data=mean_data.query("indel_dist"),edgecolor = "black",
                    palette=solver_colors, ax=axes[i], marker ="X", legend=False, s=80, zorder=3,linewidth = .5)
        axes[i].axvline(param_defaults[param],color='gray', linestyle='--', linewidth=1,zorder = -1)
        axes[i].set_ylim(metric_min, metric_max) 
        axes[i].set_xlabel(param_names[param])
        if i == 0:
            axes[i].set_ylabel(metric_names[metric])
        else:
            axes[i].set_ylabel("") 
            axes[i].set_yticklabels([])
        axes[i].set_xticks(param_data[param].unique())
        if param == "size":
            axes[i].set_xscale('log')
    solver_handles = [mlines.Line2D([], [], color=solver_colors[solver],
                                    label=solver_names[solver].replace(" ", "\n")) for solver in solver_names.keys()]
    fig.legend(handles=solver_handles, loc='center left', bbox_to_anchor=(1, 0.8), title = "Solver")
    indel_handles = [mlines.Line2D([], [], color='black', linestyle='-', marker='o', label="8 uniform"),
                     mlines.Line2D([], [], color='black', linestyle='--', marker='x', label="Indel",markersize = 8)]
    fig.legend(handles=indel_handles, loc='center left', bbox_to_anchor=(1, 0.5),
               title="Distribution")
    save_plot(fig, f"{metric}_parameter_sweep_lineplot", plots_path)

# Min characters line plot
def min_characters_lineplot(figsize=(2, 2)):
    fig, ax = plt.subplots(figsize=figsize)
    data = pd.read_csv(results_path / "min_characters_simulation.csv")
    data["cells"] = 2**data["generations"]
    data["min_pct"] = (data["min_frac"] * 100).astype(int)
    sns.lineplot(x="cells", y="characters", data=data, hue="min_pct",palette=colors[:3])
    plt.legend(title="Branches \nwith edit (%)",alignment = "left")
    plt.xscale('log')
    plt.xlabel(param_names["size"])
    plt.ylabel(param_names["characters"])
    ax.xaxis.set_major_locator(ticker.FixedLocator([1e3, 1e6, 1e9]))
    save_plot(fig, "min_characters_lineplot", plots_path)

if __name__ == "__main__":
    # Load data
    states_vs_frac = pd.read_csv(results_path / "states_vs_frac_simulation.csv")
    states_vs_entropy = pd.read_csv(results_path / "states_vs_entropy_simulation.csv")
    param_sweep = pd.read_csv(results_path / "parameter_sweep_simulation.csv")
    # Generate plots
    for metric in metric_names.keys():
        vmax = .7 if metric == "rf" else .9
        vmin = .3
        metric_heatmap(states_vs_frac,"edit_frac","states",metric,vmin = vmin,vmax =  vmax)
        metric_heatmap(states_vs_entropy,"entropy","states",metric,vmin = vmin,vmax =  vmax)
        parameter_lineplots(param_sweep,metric)
    min_characters_lineplot()
    
    

