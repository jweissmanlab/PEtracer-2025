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
from src.config import default_colors

# Define constants
metrics = {"rf":"Robinson-Foulds distance",
           "triplets":"Mean triplets correct"}
solvers = {"nj":"Neighbor Joining",
            "upgma":"UPGMA",
            "hybrid":"Cassiopeia Hybrid"}
params = {"size":"Number of cells",
          "edit_frac":"Fraction of\ncharacters edited",
          "characters":"Number of characters",
          "missing_rate":"Fraction of\ncharacters missing"}
param_defaults = {"size":1000,
                 "edit_frac":.7,
                 "characters":60,
                "missing_rate":.1}

# Load results
state_dist_results = pd.read_csv(
    results_path / "state_distribution_simulation.tsv",sep="\t")
param_results = pd.read_csv(
    results_path / "parameter_sweep_simulation.tsv",sep="\t")

# Remove greedy solver
state_dist_results = state_dist_results[state_dist_results["solver"] != "greedy"]
param_results = param_results[param_results["solver"] != "greedy"]

# State distribution line plot
def state_distribution_lineplot(results,metric,metric_label):
    plt.figure(figsize=(2,2))
    sns.lineplot(data=results[results[f'best_{metric}']], x="states", y=metric, 
        hue="entropy")
    plt.xlabel("Number of states")
    plt.ylabel(metric_label)
    plt.xticks(results['states'].unique());
    plt.legend(title="$H_{norm}$")
    return plt

# Heatmap for each solver
def solver_heatmaps(results,x,xlabel,y,ylabel,metric,metric_label,
                           cmap = "viridis", log = False):
    metric_min = results[metric].min()
    metric_max = results[metric].max()
    fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True,
                             gridspec_kw={'wspace': 0.1})
    for i, solver in enumerate(solvers.keys()):
        ax = axes[i]
        metric_mat = results[results["solver"] == solver].pivot_table(index=y, 
                             columns=x, values=metric, aggfunc='mean')
        sns.heatmap(metric_mat, ax=ax, cbar=False, cmap=cmap, vmin=metric_min, vmax=metric_max)
        for j, j_value in enumerate(metric_mat.index):
            for k, k_value in enumerate(metric_mat.columns):
                value = metric_mat.loc[j_value, k_value]
                is_best = results[(results[x] == k_value) & (results[y] == j_value) & 
                                  (results[f'best_{metric}'])]['solver'].iloc[0] == solver
                weight = 'bold' if is_best else 'normal'
                ax.text(k + 0.5, j + 0.5, f'{value:.2f}'.lstrip('0'), ha='center', va='center', 
                        fontsize=7.5, fontweight = weight)
        ax.set_title(solvers[solver])
        ax.set_xlabel(xlabel if i == 1 else "") 
        if i == 0:
            ax.set_ylabel(ylabel)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        else:
            ax.set_ylabel("")
    return plt

# Line plot for each parameter
def parameter_lineplots(results,metric,metric_label):
    metric_min = results[metric].min()
    metric_max = results[metric].max()
    solver_colors = {solver: default_colors[i] for i, solver in enumerate(solvers.keys())}
    fig, axes = plt.subplots(1, 4, figsize=(8, 1.8))
    for i, param in enumerate(params.keys()):
        param_results = results.copy()
        for var_param in params.keys():
            if var_param != param:
                param_results = param_results[param_results[var_param] == param_defaults[var_param]]
        sns.lineplot(x=param, y=metric, hue="solver", style="indel_dist", data=param_results,
                    palette=solver_colors , ax=axes[i], legend=False, markers=True)
        axes[i].set_ylim(metric_min, metric_max) 
        axes[i].set_xlabel(params[param])
        axes[i].yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        if i == 0:
            axes[i].set_ylabel(metric_label)
        else:
            axes[i].set_ylabel("")
            axes[i].set_yticklabels([])
        axes[i].set_xticks(param_results[param].unique())
        if param == "size":
            axes[i].set_xscale('log')
    # Add solver legend
    solver_handles = [mlines.Line2D([], [], color=solver_colors[solver],
                                    label=solvers[solver].replace(" ", "\n")) for solver in solvers.keys()]
    fig.legend(handles=solver_handles, loc='upper center', bbox_to_anchor=(0.4, -0.18),
               ncol=len(solvers), title="Solver")
    # Tracer legend
    indel_handles = [mlines.Line2D([], [], color='black', linestyle='-', marker='o', label="8 uniform"),
                     mlines.Line2D([], [], color='black', linestyle='--', marker='x', label="Indel")]
    fig.legend(handles=indel_handles, loc='upper center', bbox_to_anchor=(0.68, -0.18),
               title="State distribution")
    return plt

# Generate plots
if __name__ == "__main__":
    for metric, label in metrics.items():
        cmap = "viridis" if metric == "triplets" else "viridis_r"
        save_plot(state_distribution_lineplot(state_dist_results,metric,label), 
                    f"{metric}_vs_state_distribution", plots_path)
        save_plot(solver_heatmaps(state_dist_results,"entropy","$H_{norm}$",
                    "states","Number of states",metric,label,cmap = cmap),
                    f"{metric}_state_distribution_heatmap", plots_path)
        save_plot(parameter_lineplots(param_results,metric,label),
                    f"{metric}_vs_parameter", plots_path)
        

