import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import petracer
import pycea
import scipy as sp
import seaborn as sns
import sklearn as skl
import treedata as td
from petracer.config import colors, discrete_cmap, edit_ids, edit_palette, sequential_cmap, site_ids, site_names
from petracer.utils import save_plot
from petracer.tree import calculate_edit_frac

base_path, data_path, plots_path, results_path = petracer.config.get_paths("barcoded_tracing")
petracer.config.set_theme()
np.random.seed(42)

# Define constants
metric_names = {"fmi":"Clone Barcode FMI"}
param_names = {"characters":"Number of edit sites",
               "detection_rate":"Detection rate (%)"}


## Data processing functions
def sample_distances():
    """Sample pairwise distances from the tree and character matrices."""
    distances = []
    for clone in range(1,7):
        tdata = td.read_h5ad(data_path / f"barcoded_tracing_clone_{clone}.h5td")
        pycea.tl.tree_distance(tdata,depth_key="time",sample_n = 10000,random_state=0)
        pycea.tl.distance(tdata,"characters",connect_key = "tree",metric = petracer.tree.hamming_distance,key_added = "character")
        clone_distances = pycea.tl.compare_distance(tdata,dist_keys = ["character","tree"])
        clone_distances = clone_distances.query("obs1 != obs2").copy()
        clone_distances["tree_distances"] = clone_distances["tree_distances"] / 2
        for barcode in ["puro","blast"]:
            clone_distances[f"{barcode}1"] = clone_distances["obs1"].map(tdata.obs[f"{barcode}_clade"])
            clone_distances[f"{barcode}2"] = clone_distances["obs2"].map(tdata.obs[f"{barcode}_clade"])
            clone_distances[f"{barcode}_same"] = clone_distances[f"{barcode}1"] == clone_distances[f"{barcode}2"]
        clone_distances["clone"] = clone
        distances.append(clone_distances)
    distances = pd.concat(distances)
    return distances


def sort_barcode_columns(tdata):
    """Sort the barcode columns in the tree data object."""
    for barcode in ["puro","blast"]:
        barcode_counts = tdata[petracer.tree.get_leaves(tdata.obst["tree"])].obsm[f"{barcode}_counts"]
        row_indices = np.arange(barcode_counts.shape[0])
        center_of_mass = (barcode_counts.values * row_indices[:, np.newaxis]).sum(axis=0) / barcode_counts.sum(axis=0)
        sorted_columns = barcode_counts.columns[np.argsort(center_of_mass)]
        tdata.obsm[f"{barcode}_counts"] = tdata.obsm[f"{barcode}_counts"].loc[:,reversed(sorted_columns)]


## Plotting functions
def site_edit_rates(plot_name):
    """Barplot with edit rate for each site"""
    edit_counts = pd.read_csv(results_path / "tree_edit_counts.csv")
    site_edit_rate = edit_counts.groupby(["site","clone"]).agg({"n":"sum","n_edges":"first"}).reset_index()
    site_edit_rate["rate"] = site_edit_rate["n"]/site_edit_rate["n_edges"]
    site_edit_rate["site"] = site_edit_rate["site"].map(site_names).str.split(" ").str[-1]
    site_edit_rate = site_edit_rate.sort_values("site")
    fig, ax = plt.subplots(figsize=(1.5, 1.8),dpi = 300)
    sns.barplot(data = site_edit_rate, y = "rate",x = "site",hue = "site",palette = discrete_cmap[3],
        dodge = False, errwidth=2,errcolor = "black",saturation = 1)
    plt.ylabel("Edits per branch")
    plt.xlabel("Edit site")
    plt.gca().get_legend().remove()
    save_plot(fig, plot_name, plots_path)


def clone_fmi_violin(plot_name,figsize = (2,2)):
    """Violin plot of clone FMI"""
    clone_fmi = pd.read_csv(results_path / "clone_fmi.csv")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.violinplot(data=clone_fmi.query("solver == 'upgma'"), x="permute", y="fmi",inner = None,
                palette=[colors[2],"lightgray"],saturation=1,linewidth =.5,bw = 1,ax = ax)
    sns.swarmplot(data=clone_fmi.query("solver == 'upgma'"), x="permute", y="fmi", color="black",size=3,ax = ax)
    plt.ylabel(metric_names["fmi"])
    plt.xlabel("Barcodes")
    plt.ylim(0,1.05)
    plt.yticks([0,0.5,1])
    plt.xticks([0,1],["True","Mixed"])
    save_plot(fig, plot_name, plots_path)


def clone_fmi_lineplot(plot_name,x,hue = "clone",figsize = (2,2)):
    """Line plot showing the relationship between clone FMI and the number of edit sites or detection rate"""
    fmi = pd.read_csv(results_path / f"fmi_vs_{x}.csv")
    if x == "detection_rate":
        fmi[x] = fmi[x]*100
    fig, ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    if hue is None:
        fmi = fmi.groupby(["clone",x]).agg({"fmi":"mean"}).reset_index()
        pallette = None
        if x == "characters":
            fmi = fmi[fmi[x] <= 35]
        if x == "detection_rate":
            fmi = fmi[fmi[x] <= 80]
    else:
        pallette = mcolors.LinearSegmentedColormap.from_list("lightgray_to_black", ["lightgray", "black"])
    sns.lineplot(data=fmi,x=x,y="fmi",ax=ax,hue = hue,palette = pallette,
                 legend = True,linewidth = 1.5,color = colors[1],err_kws = {"linewidth":0})
    plt.ylim(0,1)
    ax.set_xlabel(param_names[x])
    ax.set_ylabel(metric_names["fmi"])
    if x == "detection_rate":
        ax.axvline(60, linestyle="--", zorder=0,color = "black")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    save_plot(fig, plot_name, plots_path)


def nj_vs_upgma_fmi_scatterplot(plot_name,figsize = (2,2)):
    """Scatterplot comparing NJ and UPGMA FMI"""
    clone_fmi = pd.read_csv(results_path / "clone_fmi.csv")
    fmi_long = clone_fmi.query("~permute").melt(id_vars = ["clone","solver"],value_vars = ["blast_fmi","puro_fmi"])
    fmi_long = fmi_long.pivot_table(index = ["clone","variable"],columns = ["solver"],values = "value").reset_index()
    fmi_long["Barcode"] = fmi_long["variable"].map({"blast_fmi":"Blast","puro_fmi":"Puro"})
    fig, ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    sns.scatterplot(data = fmi_long,x = "nj",y = "upgma",markers = ["o","s"],legend=False,
                    hue = "clone",style = "Barcode",s = 40,palette = discrete_cmap[6])
    plt.plot([.75,1],[.75,1],color = "black",linestyle = "--",zorder = 0)
    r2 = sp.stats.pearsonr(fmi_long["nj"],fmi_long["upgma"])[0]**2
    ax.text(.9,.76,f"r2 = {r2:.2f}",fontsize = 8)
    plt.xlabel("NJ barcode FMI")
    plt.ylabel("UPGMA barcode FMI")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))
    # Add legend
    fig.legend(handles = petracer.legends.barcoding_clone_legend,loc = "center",
            bbox_to_anchor=(.35,-.12),ncol = 3,title = "Clone",columnspacing = .5,handletextpad = .2)
    fig.legend(handles = petracer.legends.barcode_legend,loc = "center",
            bbox_to_anchor=(.8,-.12),ncol = 1,title = "Barcode",columnspacing = .5,handletextpad = .2)
    save_plot(fig, plot_name, plots_path)


def polar_tree_with_clades(plot_name,clone,barcode,title = None,scale = False,figsize = (5,5)):
    """Plot a polar tree colored by clade"""
    tdata = td.read_h5ad(data_path / f"barcoded_tracing_clone_{clone}.h5td")
    clade_palette = {str(clade):color for clade, color in enumerate(colors[1:21]*100)}
    if scale is True:
        n_cells = len(tdata.obs)
        figsize = (figsize[0]*np.sqrt(n_cells/5000),figsize[1]*np.sqrt(n_cells/5000))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize,dpi = 600, layout = "constrained")
    if barcode is not None:
        pycea.pl.branches(tdata, depth_key = "time", polar = True,ax = ax,
                        color = f"{barcode}_clade",palette = clade_palette,linewidth=.3)
    else:
        pycea.pl.branches(tdata, depth_key = "time", polar = True,ax = ax,linewidth=.3)
    if barcode in ["puro","combined"]:
        pycea.pl.nodes(tdata,color = "puro_lca",ax = ax,palette = clade_palette,style = "s",size = 15)
        pycea.pl.annotation(tdata,ax = ax,keys = ["puro"],palette=clade_palette)
    if barcode in ["blast","combined"]:
        pycea.pl.nodes(tdata,color = "blast_lca",ax = ax,palette = clade_palette,size = 10)
        pycea.pl.annotation(tdata,keys = ["blast"],gap = .02, palette = clade_palette)
    if title:
        ax.set_title(title)
    save_plot(fig,plot_name,plots_path,transparent=True,rasterize=True)

def tree_with_characters(plot_name,clone,figsize = (5,5)):
    """Plot tree with characters"""
    tdata = td.read_h5ad(data_path / f"barcoded_tracing_clone_{clone}.h5td")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    pycea.pl.branches(tdata, depth_key = "time",ax = ax,linewidth=.3)
    petracer.tree.plot_grouped_characters(tdata,ax = ax,label = True)
    save_plot(fig,plot_name,plots_path,rasterize=True)


def clone_stats_table(plot_name,figsize = (4,2)):
    """Render table with clone stats"""
    # Load data
    clone_stats = pd.read_csv(results_path / "clone_stats.csv")
    clone_stats = clone_stats.round(2)
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")  # set size frame
    ax.axis('tight')
    ax.axis('off')
    clone_stats.columns = clone_stats.columns.str.replace("_","\n")
    tbl = ax.table(cellText=clone_stats.values, colLabels=clone_stats.columns, cellLoc='center', loc='center')
    for i in range(len(clone_stats.columns)):
        tbl[0, i].set_height(0.3)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    save_plot(fig,plot_name, plots_path)

def distance_comparison_kdeplot(plot_name,distances,barcode,ylabel = True,clone = None,figsize = (2.6,2.6)):
    """Plot KDE of character and phylogenetic distances"""
    # get KS statistic
    ks_statistics = {}
    for metric in ["character","tree"]:
        if clone is not None:
            same_distances = distances.query(f"clone == {clone} & {barcode}_same")[f"{metric}_distances"]
            diff_distances = distances.query(f"clone == {clone} & not {barcode}_same")[f"{metric}_distances"]
        else:
            same_distances = distances.query(f"{barcode}_same")[f"{metric}_distances"]
            diff_distances = distances.query(f"not {barcode}_same")[f"{metric}_distances"]
        ks_statistics[metric] = sp.stats.ks_2samp(same_distances,diff_distances).statistic
    # plot
    sampled_distances = distances.groupby(f"{barcode}_same").sample(1000)
    g = sns.JointGrid(data=sampled_distances, x="character_distances", y="tree_distances", hue=f"{barcode}_same", marginal_ticks=True,
                    height=figsize[0], ratio=2, space=0.5, palette={True:colors[2],False:"darkgray"})
    g.plot_joint(sns.kdeplot, common_norm=False, common_grid=True,linewidths=.7,legend = False,bw_adjust=1.5,thresh = .1,fill = True,alpha = .8)
    g.plot_marginals(sns.histplot, common_norm=False, linewidth=.5,stat = "probability",bins = 20,alpha = .7)
    g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(.5))
    g.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(4))
    g.ax_joint.set_xlabel("Pairwise character\ndistance (hamming)")
    if ylabel:
        g.ax_joint.set_ylabel("Pairwise phylogentic\ndistance (days)")
    else:
        g.ax_joint.set_ylabel("")
    g.ax_joint.set_xlim(0, 1.7)
    g.ax_joint.set_ylim(0, 18)
    g.ax_marg_x.set_ylim(0, 0.25)
    g.ax_marg_x.yaxis.set_major_locator(ticker.MultipleLocator(.1))
    # add KS statistic
    g.ax_joint.text(1.7, 25, f"KS = {ks_statistics['character']:.2f}", fontsize=8)
    g.ax_joint.text(2, 19, f"KS = {ks_statistics['tree']:.2f}", fontsize=8)
    # add legend
    handles = [mpatches.Patch(color=colors[2], label='BC match'),
            mpatches.Patch(color="darkgray", label='BC mismatch')]
    g.ax_joint.legend(handles=handles, loc='upper center', fontsize=10,ncol = 2,
                    bbox_to_anchor=(.55, -.5),columnspacing = .5)
    save_plot(g.figure, plot_name, plots_path, transparent=True)


def ks_comparison_scatterplot(plot_name,distances,figsize = (2.5,2.5)):
    """Scatterplot comparing KS statistics for character and phylogenetic distances"""
    # Calculate KS statistics
    ks_statistics = []
    for clone in range(1,7):
        for barcode in ["puro","blast"]:
            for metric in ["tree","character"]:
                same_distances = distances.query(f"clone == {clone} & {barcode}_same")[f"{metric}_distances"]
                diff_distances = distances.query(f"clone == {clone} & not {barcode}_same")[f"{metric}_distances"]
                ks_statistic = sp.stats.ks_2samp(same_distances,diff_distances).statistic
                ks_statistics.append({"clone":clone,"barcode":barcode,"metric":metric,"ks_statistic":ks_statistic})
    ks_statistics = pd.DataFrame(ks_statistics)
    ks_statistics = ks_statistics.pivot(index=["clone","barcode"],columns="metric",values="ks_statistic")
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.scatterplot(data=ks_statistics,x="character",y="tree",hue="clone",style="barcode",
                    markers = ["o","s"],s = 40,legend = False,palette = discrete_cmap[6])
    plt.plot([.75,1],[.75,1],color="black",linestyle="--")
    plt.xlabel("Character distance (KS statistic)")
    plt.ylabel("Phylogenetic distance (KS statistic)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(.1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))
    # Add legend
    fig.legend(handles = petracer.legends.barcoding_clone_legend,loc = "center",
            bbox_to_anchor=(.4,-.12),ncol = 3,title = "Clone",columnspacing = .5,handletextpad = .2)
    fig.legend(handles = petracer.legends.barcode_legend,loc = "center",
            bbox_to_anchor=(.8,-.12),ncol = 1,title = "Barcode",columnspacing = .5,handletextpad = .2)
    save_plot(fig, plot_name, plots_path)


def lca_depth_ridgeplot(plot_name,figsize):
    """Plot ridgeplot of LCA depths"""
    # Get LCA depths
    lca_depths = []
    for clone in range(1,7):
        tdata = td.read_h5ad(data_path / f"barcoded_tracing_clone_{clone}.h5td")
        for node in tdata.obst["tree"].nodes:
            node_attrs = tdata.obst["tree"].nodes[node]
            if "puro_lca" in node_attrs:
                lca_depths.append({"time":node_attrs["time"],"barcode":"puro","clone":clone})
            if "blast_lca" in node_attrs:
                lca_depths.append({"time":node_attrs["time"],"barcode":"blast","clone":clone})
    lca_depths = pd.DataFrame(lca_depths)
    # Plot
    lca_depths = lca_depths.sort_values("barcode")
    fig, axes = plt.subplots(6, 1, figsize=figsize,dpi = 600)  # set size frame
    fig.subplots_adjust(hspace=-.5)
    for i in range(1,7):
        ax = axes[i-1]
        sns.kdeplot(data = lca_depths.query("clone == @i"),hue = "barcode",x = "time",bw_adjust=1,alpha = .7,
                    ax = ax,common_norm = False,legend=False,palette = {"puro":colors[1],"blast":colors[2]},fill = True)
        ax.patch.set_alpha(0)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
        if i != 6:
            ax.set_xticks([])
            ax.set_xlabel("")
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.set_xlabel("Inferred LCA time (days)")
        if i == 4:
            ax.set_ylabel("Barcode clade density")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, .6)
        ax.text(12, .05, f"Clone {i}", fontsize=10)
        ax.axvline(5, color=colors[1], linestyle="--", linewidth=1)
        ax.axvline(7, color=colors[2], linestyle="--", linewidth=1)
    # Add legeng
    handles = [mpatches.Patch(color=colors[1], label='Puro BC'),
                mpatches.Patch(color=colors[2], label='Blast BC')]
    fig.legend(handles=handles, loc='center', fontsize=10,ncol = 2,
                bbox_to_anchor=(.5, -.16),columnspacing = .5)
    save_plot(fig,plot_name, plots_path)


def edit_fraction_stacked_barplot(plot_name,figsize=(1,2)):
    """Plot stacked barplot of edit fractions"""
    # Get edit fractions
    alleles = pd.read_csv(data_path / "barcoded_tracing_alleles.csv",keep_default_na=False)
    alleles = alleles.query("whitelist").copy()
    for site in edit_ids.keys():
        alleles[site] = alleles[site].map(edit_ids[site]).fillna(9)
    alleles = alleles.melt(id_vars=["clone"], value_vars=site_ids.keys(), var_name="site", value_name="allele")
    alleles["site"] = alleles["site"].map(site_ids)
    edit_counts = alleles.groupby(["site","allele"]).size().unstack(fill_value=0)
    edit_order = list(range(1,9)) + [9,0]
    edit_counts = edit_counts.reindex(columns=edit_order)
    edit_fracs = edit_counts.div(edit_counts.sum(axis=1), axis=0) * 100
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi=600,layout = "constrained")
    edit_fracs.plot(kind='bar', stacked=True,color = [edit_palette[str(i)] for i in edit_fracs.columns],ax = ax,width = .9)
    edit_fracs.to_csv(results_path / "overall_edit_fracs.csv")
    plt.legend().remove()
    plt.xlabel("Edit site")
    plt.ylabel("Fraction of LMs (%)")
    plt.tight_layout()
    plt.xticks(rotation=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    save_plot(fig, "edit_fraction_stacked_barplot", plots_path)


def tree_with_barcodes(plot_name,clone,figsize = (8.2,1.8)):
    """PLot tree with puro and blast barcode counts"""
    clade_palette = {str(clade):color for clade, color in enumerate(colors[1:21]*100)}
    tdata = td.read_h5ad(data_path / f"barcoded_tracing_clone_{clone}.h5td")
    sort_barcode_columns(tdata)
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    pycea.pl.branches(tdata, depth_key = "time",ax = ax,linewidth=.1)
    characters_width = 3/tdata.obsm["characters"].shape[1]
    petracer.tree.plot_grouped_characters(tdata,ax = ax,width = characters_width)
    pycea.pl.annotation(tdata,keys = ["puro"],ax = ax,palette = clade_palette, width = .2,gap = .6,label = "")
    puro_width = 3/tdata.obsm["puro_counts"].shape[1]
    pycea.pl.annotation(tdata,keys = ["puro_counts"],ax = ax,width = puro_width,vmax = 250,
                        cmap = sequential_cmap,label = "",border_width=.5)
    pycea.pl.annotation(tdata,keys = ["blast"],ax = ax,palette = clade_palette, width = .2,gap = .6,label = "")
    blast_width = 3/tdata.obsm["blast_counts"].shape[1]
    pycea.pl.annotation(tdata,keys = ["blast_counts"],ax = ax,width = blast_width,vmax = 250,
                        cmap = sequential_cmap,label = "",border_width=.5)
    save_plot(fig, plot_name, plots_path,svg=False)


def polar_tree_with_pe_edit_frac(plot_name,clone = "4",figsize = (3,3)):
    """Plot polar tree with PE expression and edit fraction"""
    tdata = td.read_h5ad(data_path / f"barcoded_tracing_clone_{clone}.h5td")
    calculate_edit_frac(tdata)
    tdata.obs["pe_expression"] = tdata[:,"PE2maxGFP"].X.toarray().flatten()
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'},dpi = 1200,layout = "constrained")
    pycea.pl.branches(tdata,depth_key="time",ax = ax,polar = True,linewidth = .1)
    pycea.pl.annotation(tdata,"pe_expression",width = .2,cmap = "Reds",vmax = 10)
    pycea.pl.annotation(tdata,"edit_frac",width = .2,cmap = "Blues",vmin = 0, vmax = 1)
    pycea.pl.annotation(tdata,"fitness",width = .2,cmap = "Purples",vmin = 0, vmax = 6)
    save_plot(fig,plot_name,plots_path,rasterize = True)


def scatter_with_regression(plot_name, clone, x, xlabel, y, ylabel, figsize = (2,2)):
    """Scatterplot with regression line"""
    tdata = td.read_h5ad(data_path / f"barcoded_tracing_clone_{clone}.h5td")
    calculate_edit_frac(tdata)
    tdata.obs["pe_expression"] = tdata[:,"PE2maxGFP"].X.toarray().flatten()
    data = tdata.obs.copy()
    if x == "pe_expression":
        data = data[data["pe_expression"] < 15]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=300)
    sns.scatterplot(data=data, y=y, x=x, alpha=.1, ax=ax, s=10)
    sns.regplot(x=x, y=y, data=data, ax=ax, scatter=False, line_kws={"color":"black"})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    slope, intercept, _, _, _ = sp.stats.linregress(data[x], data[y])
    y_pred = data[x] * slope + intercept
    r2 = skl.metrics.r2_score(data[y], y_pred)
    ax.text(0.5, 0.95, f"$r^2$ = {r2:.2f}", transform=ax.transAxes, va='top', ha='left')
    save_plot(fig, plot_name, plots_path, rasterize=True)


## Generate plots
if __name__ == "__main__":
    #Clone stats
    clone_stats_table("clone_stats_table")
    edit_fraction_stacked_barplot("edit_fraction_stacked_barplot",figsize=(1.3,2))
    # FMI
    clone_fmi_violin("clone_fmi_violin",figsize = (1.5,2))
    nj_vs_upgma_fmi_scatterplot("nj_vs_upgma_fmi_scatterplot",figsize = (2.2,2.2))
    for x in ["characters","detection_rate"]:
        clone_fmi_lineplot(f"clone_fmi_vs_{x}_lineplot",x = x,figsize = (2,2))
        clone_fmi_lineplot(f"fmi_vs_{x}_lineplot",x = x,hue = None,figsize = (2,2))
    # Distance comparison
    distances = sample_distances()
    for barcode in ["puro","blast"]:
         distance_comparison_kdeplot(f"clone_4_{barcode}_distance_comparison_kdeplot",distances,barcode,clone = 4,figsize = (2.8,2.8))
    ks_comparison_scatterplot("ks_comparison_scatterplot",distances,figsize = (2.2,2.2))
    # LCA depths
    lca_depth_ridgeplot("lca_depth_ridgeplot",(2.5,2))
    # Clone trees
    for clone in range(1,7):
        polar_tree_with_clades(f"clone_{clone}_combined_clades",clone,"combined",
           scale = True,figsize = (3.5,3.5))
        tree_with_barcodes(f"clone_{clone}_with_barcodes",clone,figsize = (8.2,1.7))
    # Example clone
    example = 4
    polar_tree_with_clades(f"clone_{example}_puro_clades",example,"puro",figsize = (4,4))
    polar_tree_with_clades(f"clone_{example}_blast_clades",example,"blast",figsize = (4,4))
    polar_tree_with_clades(f"clone_{example}_polar",example,None,figsize = (4,4))
    polar_tree_with_clades(f"clone_{example}_combined_clades",example,"combined",figsize = (4,4))
    tree_with_characters(f"clone_{example}_with_characters",example,figsize = (2.5,2))
    polar_tree_with_pe_edit_frac(f"clone_{example}_pe_edit_frac",example,figsize = (2,2))
    scatter_with_regression("clone_4_pe_vs_edit_frac","4","pe_expression","PE2max expression","edit_frac","Fraction of sites edited")
    scatter_with_regression("clone_4_pe_vs_fitness","4","pe_expression","PE2max expression","fitness","Fitness")
    scatter_with_regression("clone_4_edit_frac_vs_fitness","4","edit_frac","Fraction of sites edited","fitness","Fitness")
