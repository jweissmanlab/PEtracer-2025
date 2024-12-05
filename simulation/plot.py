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
from petracer.utils import save_plot
from petracer.config import colors, sequential_cmap, discrete_cmap

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
    fig, ax = plt.subplots(figsize = figsize,layout = "constrained",dpi = 600)
    cmap = sequential_cmap.reversed() if metric == "rf" else sequential_cmap
    sns.heatmap(data,cmap = cmap,annot = True,cbar=False,
                fmt = ".2f",ax = ax,annot_kws={"size": 9},vmax = vmax,vmin = vmin)
    plt.yticks(rotation=0)
    ax.set_xlabel(param_names[x].replace("\n"," "))
    ax.set_ylabel(param_names[y])
    save_plot(fig, f"{metric}_heatmap_{x}_vs_{y}", plots_path)

# Parameter sweep line plots
def parameter_lineplots(data,metric,params = ["size","edit_frac","characters","detection_rate"],figsize=(6,2.2)):
    data = data.query("solver.isin(@solver_names.keys())").copy()
    data["edit_frac"] = (data["edit_frac"] * 100).astype(int)
    data["detection_rate"] = (100 - data["missing_rate"] * 100).astype(int)
    metric_min = data[metric].min()
    metric_max = data[metric].max()
    fig, axes = plt.subplots(1, 4, figsize=figsize,layout = "constrained",dpi = 600)
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
    fig.legend(handles=solver_handles, loc='upper left', bbox_to_anchor=(0.15, .05), title = "Solver",ncol = 3,columnspacing = 1)
    indel_handles = [mlines.Line2D([], [], color='black', linestyle='-', marker='o', label="8 uniform"),
                     mlines.Line2D([], [], color='black', linestyle='--', marker='x', label="Indel",markersize = 8)]
    fig.legend(handles=indel_handles, loc='upper left', bbox_to_anchor=(0.6, .05),
               title="LM distribution",ncol = 2,columnspacing = 1)
    save_plot(fig, f"{metric}_parameter_sweep_lineplot", plots_path)

# Min characters line plot
def min_characters_lineplot(plot_name,figsize=(2, 2)):
    fig, ax = plt.subplots(figsize=figsize)
    data = pd.read_csv(results_path / "min_characters_simulation.csv")
    data["cells"] = 2**data["generations"]
    data["edit_pct"] = (data["branch_edit_frac"] * 100).astype(int)
    sns.lineplot(x="cells", y="characters", data=data, hue="edit_pct",palette=colors[:3])
    plt.legend(title="Branches \nwith edit (%)",alignment = "left")
    plt.xscale('log')
    plt.xlabel(param_names["size"])
    plt.ylabel(param_names["characters"])
    ax.xaxis.set_major_locator(ticker.FixedLocator([1e3, 1e6, 1e9]))
    save_plot(fig,plot_name,plots_path)

# Optimal rate line plot
def edit_rate_lineplot(plot_name,log = True,figsize = (3,2)):
    edit_rate = pd.read_csv(results_path / "edit_rate_simulation.csv")
    edit_rate["edit_pct"] = (edit_rate["site_edit_frac"] * 100).astype(int)
    fig, ax = plt.subplots(figsize = figsize,layout = "constrained",dpi = 600)
    sns.lineplot(data = edit_rate,x = "generations",y = "edit_rate",hue = "edit_pct",
                 palette = ["gray",colors[1],colors[2],colors[4],"lightgray"])
    ax.set_xlabel("Days of tracing")
    ax.set_ylabel("Edit rate (edits/day)")
    if log:
        plt.yscale("log")
        plt.ylim(.006,.35)
    plt.xticks([0,20,40,60,80,100])
    plt.grid(True, which="both", ls="--", c='black', alpha=.7,linewidth = .4)
    plt.legend(title = "Edit sites\nwith LM (%)",bbox_to_anchor=(1, 1), loc='upper left')
    save_plot(fig, plot_name, plots_path)

# Fraction of edit sites over time line plot
def frac_over_time_lineplot(plot_name,figsize=(2.8,2)):
    # Load data
    results = pd.read_csv(results_path / "frac_over_times_simulation.csv")
    results["edit_pct"] = results["site_edit_frac"] * 100
    # Plot
    fig, ax = plt.subplots(figsize = (2.8,2),layout = "constrained",dpi = 600)
    sns.lineplot(data = results,x = "generations",y = "edit_pct",hue = "edit_rate",legend = True,palette = discrete_cmap[5])
    ax.set_xlabel("Day")
    ax.set_ylabel("Edit sites with LM (%)")
    plt.xticks([0,10,20,30]);
    plt.legend(title = "Edit rate\n(edits/day)",bbox_to_anchor=(1, 1), loc='upper left')
    save_plot(fig,plot_name,plots_path)

# RF vs triplets scatter plot
def rf_vs_triplets_scattterplot(plot_name,figsize = (2.2,2.2)):
    fig,ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    sns.scatterplot(data=param_sweep,x="rf",y="triplets",s = 10,alpha = 0.8,ax=ax,color = colors[1])
    sns.regplot(data=param_sweep,x="rf",y="triplets",scatter=False,color="black",line_kws={"linewidth":1},ax=ax)
    # add r2 value to plot
    r2 = np.corrcoef(param_sweep["rf"],param_sweep["triplets"])[0,1]**2
    plt.text(0.6,0.95,f"$r^2$ = {r2:.2f}")
    plt.xlabel(metric_names["rf"])
    plt.ylabel(metric_names["triplets"])
    save_plot(fig,plot_name,plots_path,rasterize=True)

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
        parameter_lineplots(param_sweep,metric,figsize=(5.9,2.2))
    min_characters_lineplot("min_characters_lineplot",figsize=(2.2,0))
    edit_rate_lineplot("log_edit_rate_lineplot",log = True,figsize = (3.1,2))
    edit_rate_lineplot("edit_rate_lineplot",log = False,figsize = (3,2))
    rf_vs_triplets_scattterplot("rf_vs_triplets_scatterplot",figsize = (2,2))
    
    

