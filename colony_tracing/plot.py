import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import treedata as td
import pandas as pd
import petracer
import pycea as py

from petracer.config import colors,discrete_cmap,edit_palette
from petracer.utils import save_plot

base_path, data_path, plots_path, results_path = petracer.config.get_paths("colony_tracing")
petracer.config.set_theme()


# Define constants
clone3_clades = {
 'node296': "1",
 'node297': "2",
 'node368': "3",
 'node369': "4",
 'node35': "5",
 'node36': "6",
 'node251': "7",
 'node252': "8",
 'node125': "9",
 'node126': "10",
 'node25': "11",
 'node20': "12",
 'node21': "13",
 'node7': "14"}

# Data loading functions
def load_colonies_tdata(data_path):
    """Load the colony tracing data from the specified path."""
    tdata = td.read_h5ad(data_path / "colonies.h5ad")
    polygons = gpd.read_file(data_path / "colonies_polygons.json")
    polygons = polygons.set_crs(None, allow_override=True)
    polygons.set_index("cellBC", inplace=True)
    tdata.obs = polygons.merge(tdata.obs, left_index=True, right_index=True)
    tdata.uns["clone_colors"] = {str(i): colors[i%20+1] for i in range(1,65)}
    return tdata


def get_clone_tdata(tdata, clone, with_colony = False, tracing_only = False):
    """Get tdata object for a specific clone"""
    # Subset tdata
    if with_colony:
        colony = tdata.obs.query("clone == @clone")['colony'].values[0]
        clone_tdata = tdata[tdata.obs["colony"] == colony].copy() 
    else:
        clone_tdata = tdata[tdata.obs["clone"] == clone].copy()
    if tracing_only:
        clone_tdata = clone_tdata[clone_tdata.obs.tree.notnull()].copy()
    clone_tdata.obs = gpd.GeoDataFrame(clone_tdata.obs)
    # Subset characters
    clone_tdata.obsm["characters"] = clone_tdata.obsm["characters"].loc[:,tdata.uns["clone_characters"][clone]]
    # Subset trees
    for key in  tdata.obst.keys():
        if key != clone:
            del clone_tdata.obst[key]
    return clone_tdata


# Analysis functions
def calculate_clone_stats(tdata):
    """Calculate the number of cells, edit fraction, and detection rate for each clone."""
    clone_stats = []
    for clone in tdata.obs.clone.cat.categories:
        clone_tdata = get_clone_tdata(tdata, clone, tracing_only=False)
        detection_rate = clone_tdata.obs["detection_rate"].mean() * 100
        clone_tdata = clone_tdata[clone_tdata.obs.tree.notnull()].copy()
        py.pp.add_depth(clone_tdata)
        leaf_depth = py.utils.get_keyed_leaf_data(clone_tdata,"depth")
        site_edit_frac = petracer.tree.get_edit_frac(clone_tdata.obsm["characters"]) * 100
        edit_sites = clone_tdata.obsm["characters"].shape[1]
        clone_stats.append({"clone":clone,"n_cells":clone_tdata.n_obs,
                            "avg_depth":np.mean(leaf_depth),"edit_sites":edit_sites,
                            "site_edit_frac":site_edit_frac,"detection_rate":detection_rate})
    clone_stats = pd.DataFrame(clone_stats)
    clone_stats.to_csv(results_path / "clone_stats.csv",index=False)


# Plotting functions
def colonies_slide(plot_name,tdata,figsize = (3,3)):
    """Plot all the colonies on the slide."""
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    petracer.plotting.plot_polygons(cells = tdata.obs, palette=tdata.uns["clone_colors"], 
                ax = ax,color = "clone")
    save_plot(fig, plot_name, plots_path, rasterize = True)


def clone_with_clades(plot_name, tdata, clone, clades = None, depth = None, with_colony = False, color_branches = True, figsize = (5,2)):
    """Plot phylogenic and spatial distribution of clades for a given clone."""
    # Get clone tdata
    clone_tdata = get_clone_tdata(tdata, clone, with_colony = with_colony)
    print("Number of cells:", clone_tdata.obs.query("tree.notna()").shape[0])
    # Get clades
    if clades is None:
        clades = py.tl.clades(clone_tdata, depth_key="time",depth = depth,copy = True,key_added="all_clades")
        clades["size"] = clades["all_clades"].map(clone_tdata.obs["all_clades"].value_counts())
        clades = clades.query("size >= 5").set_index("node")["all_clades"].to_dict()
    py.tl.clades(clone_tdata, depth_key="time",clades = clades)
    clade_palette = petracer.config.get_clade_palette(clone_tdata)
    clone_tdata.obs.loc[clone_tdata.obs.clone != clone,"clade"] = "-1"
    clade_palette["-1"] = "white"
    # Plot
    fig, axes = plt.subplots(1,2,figsize=figsize,dpi = 600, layout = "constrained",gridspec_kw={'width_ratios': [3, 2]})
    if color_branches:
        py.pl.branches(clone_tdata,depth_key="time",ax = axes[0],linewidth=.3,color = "clade",palette=clade_palette)
    else:
        py.pl.branches(clone_tdata,depth_key="time",ax = axes[0],linewidth=.3)
    petracer.tree.plot_grouped_characters(clone_tdata,width=.07,ax = axes[0],label = True)
    plot_polygons(img = None, cells = clone_tdata.obs, palette=clade_palette ,color = "clade",edgecolor="black",linewidth=.2,ax = axes[1])
    axes[1].axis('off')
    save_plot(fig, plot_name, plots_path, rasterize = True)


def edit_spatial_distribution(plot_name, tdata, clone, edits, figsize = (1,1)):
    """Plot the spatial distribution of edits in a clone."""
    # Get clone tdata
    clone_tdata = get_clone_tdata(tdata, clone)
    # Add edits to obs
    previous_edit = None
    for edit, value in edits.items():
        clone_tdata.obs[edit] = clone_tdata.obsm["characters"].loc[:,edit].astype(str)
        clone_tdata.obs.loc[(clone_tdata.obs[edit] != value) & (clone_tdata.obs[edit] != "-1"),edit] = "0"
        if previous_edit is not None:
            clone_tdata.obs.loc[(clone_tdata.obs[previous_edit] < "1") & (clone_tdata.obs[edit] != "-1"),edit] = "0" 
        previous_edit = edit
    # Plot
    for i, edit in enumerate(edits.keys()):
        fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
        plot_polygons(cells = clone_tdata.obs, palette=edit_palette, ax = ax,color = edit,edgecolor="black",linewidth=.1)
        ax.axis('off')
        save_plot(fig, f"{plot_name}_{edit}-LM{edits[edit]}".replace("intID","") , plots_path)


def clone_detection_hist(plot_name, tdata, clone, figsize = (2,1.8)):
    """Histogram of intBC detection rate for a specific clone"""
    # Get detection rate
    detection_rate = tdata.obs.query("clone == @clone").sort_values("clone").copy()
    detection_rate["detection_pct"] = detection_rate["detection_rate"]*100
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.histplot(data = detection_rate,x = "detection_pct",color = "lightgray",ax = ax,bins = 11,alpha = 1,linewidth = .5)
    ax.axvline(60, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Detection rate (%)")
    plt.ylabel("Number of cells")
    mean = detection_rate["detection_rate"].mean()*100
    plt.xticks([50,mean,100], [f"50", f"{mean:.1f}", "100"])
    save_plot(fig, plot_name, plots_path)


def clone_stats_barplot(plot_name,y,figsize = (6,1)):
    """Plot barplot for clone statistics."""
    stat_names = {"n_cells":"Number\nof cells",
                "edit_sites":"Number\nof edit sites",
                "site_edit_frac":"Sites with\nLM (%)"}
    clone_stats = pd.read_csv(results_path / "clone_stats.csv")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.barplot(data = clone_stats,y = y,x = "clone",color = "lightgray",ax = ax,saturation=1,linewidth=0.5,edgecolor="black")
    plt.xticks(rotation=90,size = 7.5)
    y_mean = clone_stats[y].mean()
    yticks = list(ax.get_yticks()) + [y_mean]
    yticks = sorted(yticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:4.0f}" for tick in yticks])
    plt.ylim(0,clone_stats[y].max() + 10)
    plt.ylabel(stat_names[y])
    save_plot(fig,plot_name,plots_path)


def spatial_distance_lineplot(plot_name,clone_tdata,figsize = (2,2)):
    """lineplot with phylogenetic vs spatial distances grouped by detection rate"""
    # Get distances
    py.tl.distance(clone_tdata,key = "spatial",metric = "euclidean",sample_n = 200000,update=False)
    py.tl.tree_distance(clone_tdata,depth_key="time",connect_key="spatial")
    distances = py.tl.compare_distance(clone_tdata,dist_keys = ["spatial","tree"])
    distances["tree_distances"] = distances["tree_distances"] / 2
    distances = distances.query("obs1 != obs2").copy()
    # Bin distances
    distances["detection_1"] = distances["obs1"].map(clone_tdata.obs["detection_rate"])
    distances["detection_2"] = distances["obs2"].map(clone_tdata.obs["detection_rate"])
    distances["detection_bin_1"] = pd.cut(distances["detection_1"],bins = np.arange(.6,1.01,.1))
    distances["detection_bin_2"] = pd.cut(distances["detection_2"],bins = np.arange(.6,1.01,.1))
    distances = distances.query("detection_bin_1 == detection_bin_2").copy()
    # Get mean and se for each bin
    distances["tree_bin"] = pd.cut(distances["tree_distances"], bins = np.arange(0,6, 1))
    mean_distances = distances.groupby(["tree_bin","detection_bin_1"]).agg(
        spatial_mean=('spatial_distances', 'mean'),
        spatial_se=('spatial_distances', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        spatial_sd = ('spatial_distances', 'std')
    ).reset_index()
    mean_distances["tree_distance"] = mean_distances["tree_bin"].apply(lambda x: x.mid)
    mean_distances["permuted_mean"] = distances["spatial_distances"].mean()
    mean_distances["permuted_se"] = distances["spatial_distances"].std() / np.sqrt(len(distances))
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    # permuted
    sns.lineplot(data = mean_distances,x = "tree_distance",y = "permuted_mean",color = "black",ax=ax)
    plt.fill_between(mean_distances["tree_distance"], mean_distances["permuted_mean"] - mean_distances["permuted_se"],
                        mean_distances["permuted_mean"] + mean_distances["permuted_se"], color="black", alpha=0.2)
    # observed
    hue_categories = mean_distances["detection_bin_1"].unique()
    for i, hue in enumerate(hue_categories):
        color = discrete_cmap[4][i]
        subset = mean_distances[mean_distances["detection_bin_1"] == hue]
        sns.lineplot(data=subset, x="tree_distance", y="spatial_mean", ax=ax, label=hue, color=color)
        plt.fill_between(subset["tree_distance"],
                        subset["spatial_mean"] - subset["spatial_se"],
                        subset["spatial_mean"] + subset["spatial_se"],
                        alpha=0.2, color=color)
    plt.ylim(0,350)
    plt.ylabel("Mean spatial dist. (um)")
    plt.xlabel("Phylo. dist. (days)")
    plt.xticks(np.arange(0,6,2));
    plt.legend(title = "Detection\nrate")
    save_plot(fig,plot_name, plots_path)


def clone_detection_violin(plot_name, tdata, clones = None, figsize = (1,2)):
    """Plot distribution of detection rates for each clone"""
    # Get detection rate
    detection_rate = tdata.obs.query("clone.notnull()").sort_values("clone").copy()
    if clones is not None:
        detection_rate = detection_rate.query("clone in @clones").copy()
    detection_rate["detection_pct"] = detection_rate["detection_rate"]*100
    detection_rate.clone = detection_rate.clone.astype(str)
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.violinplot(data = detection_rate,y = "detection_pct",x = "clone",color = "lightgrey",
                ax = ax,linewidth=.5,linecolor="black",cut = 0,saturation=1,inner="quart")
    ax.axhline(60, color='black', linestyle='--', linewidth=1)
    y_mean = detection_rate["detection_rate"].mean()*100
    yticks = list(ax.get_yticks()) + [y_mean]
    yticks = sorted(yticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:4.0f}" for tick in yticks])
    plt.xlabel("")
    plt.xticks(rotation=90,size = 7.5)
    plt.ylim(0,100)
    plt.ylabel("Detection\nrate (%)")
    save_plot(fig, plot_name, plots_path)


if __name__ == "__main__":
    # Load data
    print(f"Loading data from {data_path}")
    tdata = load_colonies_tdata(data_path)
    print("Data loaded")
    # Clone statistics
    calculate_clone_stats(tdata)
    clone_detection_hist("clone_3_detection_hist",tdata,"3", figsize = (1.8,1.4))
    for metric in ["n_cells","edit_sites","site_edit_frac"]:
        clone_stats_barplot(f"clone_{metric}_barplot",metric,figsize = (6,1))
    clone_detection_violin("clone_detection_violin",tdata,figsize = (6,1))
    # Plot spatial distribution
    colonies_slide("colonies_slide",tdata)
    edit_spatial_distribution("clone_3",tdata,"3",{"intID1862-RNF2":"3","intID343-EMX1":"7","intID1364-HEK3":"4"})
    # Plot selected clones
    clone_with_clades("clone_3_clades",tdata,"3",clades = clone3_clades)
    clade_depths = {"5":.35,"8":.35,"9":.45}
    for clone, depth in clade_depths.items():
        clone_with_clades(f"clone_{clone}_clades",tdata,clone,depth = depth,with_colony = True)
    for clone in ["3","5","8","9"]:
        petracer.plotting.distance_comparison_scatter(f"clone_{clone}_phylo_vs_spatial",plots_path,
            get_clone_tdata(tdata,clone),x = "tree",y = "spatial",figsize = (2,1.9))
    petracer.plotting.distance_comparison_scatter(f"clone_3_phylo_vs_character",plots_path,
            get_clone_tdata(tdata,"3"),x = "character",y = "tree",figsize = (2,1.9),sample_n=50000)
    spatial_distance_lineplot("clone_3_spatial_distance",get_clone_tdata(tdata,"3"),figsize = (2,1.9))
    

