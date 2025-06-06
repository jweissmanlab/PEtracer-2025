import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import petracer
import treedata as td
import pycea as py
import seaborn as sns
from petracer.config import colors, discrete_cmap, sequential_cmap
from petracer.utils import save_plot, get_clade_palette

base_path, data_path, plots_path, results_path = petracer.config.get_paths("simulation")
petracer.config.set_theme()

np.random.seed(42)

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


# Plotting functions
def metric_heatmap(data,x,y,metric = "rf",vmin = 0,vmax = 1,figsize = (2.2,2.2)):
    """Plot a heatmap comparing two parameters."""
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


def parameter_lineplots(data,metric,params = ("size","edit_frac","characters","detection_rate"),figsize=(6,2.2)):
    """Line plots comparing solvers across a range of parameters."""
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


def min_characters_lineplot(plot_name,figsize=(2, 2)):
    """Line plot showing the minimum number of characters needed to achieve a certain edit fraction."""
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


def edit_rate_lineplot(plot_name,log = True,figsize = (3,2)):
    """Line plot showing the optimal edit rate for different edit fractions."""
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


def frac_over_time_lineplot(plot_name,figsize=(2.8,2)):
    """Line plot showing the fraction of edited sites over time."""
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


def rf_vs_triplets_scattterplot(plot_name,figsize = (2.2,2.2)):
    """Scatter plot showing the relationship between RF and triplets."""
    fig,ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    sns.scatterplot(data=param_sweep,x="rf",y="triplets",s = 10,alpha = 0.8,ax=ax,color = colors[1])
    sns.regplot(data=param_sweep,x="rf",y="triplets",scatter=False,color="black",line_kws={"linewidth":1},ax=ax)
    r2 = np.corrcoef(param_sweep["rf"],param_sweep["triplets"])[0,1]**2
    plt.text(0.6,0.95,f"$r^2$ = {r2:.2f}")
    plt.xlabel(metric_names["rf"])
    plt.ylabel(metric_names["triplets"])
    save_plot(fig,plot_name,plots_path,rasterize=True)


def example_ground_truth_tree(plot_name,figsize = (.75, 2)):
    """Plot the ground truth tree for the example simulation."""
    tdata = td.read_h5ad(data_path / "example_tree_simulation.h5ad")
    fig, ax = plt.subplots(figsize=figsize, dpi=300, layout = "constrained")
    py.pl.branches(tdata,tree = "tree",depth_key="time",ax = ax,linewidth=.2)
    clade_palette = get_clade_palette(tdata)
    py.pl.annotation(tdata,keys = 'clade',ax = ax,palette=clade_palette,width = .2,label = "")
    save_plot(fig,plot_name,plots_path,rasterize=True)


def example_tree_reconstructions(figsize = (2, 2)):
    """Plot reconstucted trees of example simulation."""
    tdata = td.read_h5ad(data_path / "example_tree_simulation.h5ad")
    clade_palette = get_clade_palette(tdata)
    cmaps = {
        "balenced_8":  petracer.config.edit_cmap,
        "skewed_4": mcolors.ListedColormap(['white', 'lightgray'] +  petracer.config.colors[1:5]),
        "single": mcolors.ListedColormap(['white', 'lightgray',petracer.config.colors[1]])
    }
    for name in ["balenced_8","skewed_4","single"]:
        fig, ax = plt.subplots(figsize=figsize)
        tdata.obs["clade_num"] = tdata.obs.clade.astype(int)
        py.tl.ancestral_states(tdata,tree = name,keys = "clade_num")
        py.tl.sort(tdata,tree = name,key = "clade_num")
        py.pl.branches(tdata,depth_key="time",tree = name,ax = ax,linewidth = .2)
        py.pl.annotation(tdata,keys = 'clade',ax = ax,label = "",width = .2,palette=clade_palette)
        py.pl.annotation(tdata,keys = f"{name}_characters",ax = ax,label = "",
                        cmap = cmaps[name],width = .04)
        ax.text(.3, 1, f"RF = {tdata.uns[f'{name}_rf']:.2f}, FMI = {tdata.uns[f'{name}_fmi']:.2f}", fontsize=8, transform=ax.transAxes)
        save_plot(fig,f"{name}_lm_example",plots_path,rasterize=True)


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
    example_ground_truth_tree("ground_truth_example")
    example_tree_reconstructions(figsize = (2,2))

