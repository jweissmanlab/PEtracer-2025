import anndata as ad
import fishtank as ft
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mpl_toolkits.mplot3d as m3d
import networkx as nx
import numpy as np
import pandas as pd
import petracer
import pycea as py
import scanpy as sc
import scikit_posthocs as skp
import scipy as sp
import seaborn as sns
import squidpy as sq
import statsmodels.stats as sm
import treedata as td

np.random.seed(42)

from matplotlib_scalebar.scalebar import ScaleBar
from petracer.config import (  # noqa: E402
    colors,
    edit_palette,
    get_clade_palette,
    module_palette,
    phase_palette,
    subtype_abbr,
    subtype_palette,
)
from petracer.plotting import distance_comparison_scatter
from petracer.utils import save_plot

base_path, data_path, plots_path, results_path = petracer.config.get_paths("tumor_tracing")
petracer.config.set_theme()

# Define constants
img_paths = {
    "M1-S1": "/lab/weissman_imaging/puzheng/PE_LT/20240821-F319-4-0807-0813_ingel-intBCv2Editv3/",
}

# Data loading functions
def load_tdata():
    """Load tumor tracing data."""
    clones = {}
    mice = {}
    use_clones = []
    for mouse in ["M1","M2","M3"]:
        tdata = td.read_h5ad(data_path/ f"{mouse}_tumor_tracing.h5td")
        if mouse == "M3":
            tdata = tdata[tdata.obs["sample"] != "M3-3"].copy()
        tdata.obs["sample"] = tdata.obs["sample"].str.replace("-","-S")
        tdata.obs["mouse"] = mouse
        tdata.obs["tumor"] = tdata.obs["mouse"] + "-T" + tdata.obs["tumor"].astype(str)
        mice[mouse] = tdata
    libraries = {"M1M2":ad.concat([mice["M1"],mice["M2"]], join='outer', fill_value=0),
                "M3":mice["M3"]}
    for mouse, mouse_tdata in mice.items():
        for clone in mouse_tdata.obs["clone"].cat.categories:
            tumor = mouse_tdata.obs.query("clone == @clone")['tumor'].value_counts().idxmax()
            clone_tdata = mouse_tdata[mouse_tdata.obs["tumor"] == tumor].copy()
            clone_tdata.obsm["characters"] = clone_tdata.obsm["characters"].loc[:,clone_tdata.uns["clone_characters"][clone]]
            clone_tdata.obst["tree"] = clone_tdata.obst[f"{clone}_collapsed"].copy()
            del clone_tdata.obst[f"{clone}_collapsed"]
            del clone_tdata.obst[f"{clone}"]
            clones[f"{mouse}-T{clone}"] = clone_tdata
            if clone_tdata.obs["fitness"].notnull().any():
                use_clones.append(f"{mouse}-T{clone}")
    return libraries, clones, use_clones


def load_boundaries():
    """Load tumor and module boundaries."""
    tumor_boundaries = []
    for mouse in ["M1","M2","M3"]:
        mouse_boundaries = gpd.read_file(
            data_path / f"{mouse}_tumor_boundaries.json").set_crs(None, allow_override=True)
        mouse_boundaries["tumor"] = mouse + "-T" +  mouse_boundaries["tumor"].astype(str)
        tumor_boundaries.append(mouse_boundaries)
    tumor_boundaries = gpd.GeoDataFrame(pd.concat(tumor_boundaries, ignore_index=True))
    tumor_boundaries["sample"] = tumor_boundaries["tumor"].str.replace("-","-S")
    module_boundaries = gpd.read_file(data_path / "M3_module_boundaries.json").set_crs(None, allow_override=True)
    return tumor_boundaries, module_boundaries


# Analysis functions
def generate_combined_umap(tdata):
    """Generate a combined UMAP for mouse M1 and M2."""
    sc.pp.neighbors(tdata, use_rep="X_resolVI")
    sc.tl.umap(tdata,min_dist=.2)
    tdata[tdata.obs["mouse"] == "M1"].write_h5ad(data_path / "M1_tumor_tracing.h5ad")
    tdata[tdata.obs["mouse"] == "M2"].write_h5ad(data_path / "M2_tumor_tracing.h5ad")


def calculate_imaging_stats(libraries):
    """Calculate imaging statistics for all sections."""
    imaging_stats = []
    for library, tdata in libraries.items():
        imaging_stats.append(
            tdata.obs.sort_values("tumor").groupby("sample",observed=False).agg(
                n_fovs = ("fov", "nunique"),
                n_cells = ("cell", "count"),
                n_clones = ("clone", "nunique"),
                n_tumors = ("tumor", "nunique"),
                tumors = ("tumor", lambda x: ", ".join(x.unique())),
                total_transcripts = ("total_counts", "sum"),
                transcripts_per_cell = ("total_counts", "mean"),
                detection_rate = ("detection_rate", "mean"),
        ).reset_index().assign(library = library))
    imaging_stats = pd.concat(imaging_stats).reset_index(drop=True)
    imaging_stats["detection_rate"] = imaging_stats["detection_rate"] * 100
    imaging_stats["mouse"] = imaging_stats["sample"].str.split("-",expand = True)[0]
    imaging_stats["section"] = imaging_stats["sample"].str.split("-",expand = True)[1]
    imaging_stats["library"] = imaging_stats["library"].map({"M1M2":"124 gene","M3":"175 gene"})
    imaging_stats.to_csv(results_path / "imaging_stats.csv", index=False)


def calculate_clone_stats(clones):
    """Calculate phylogeny statistics for all clones."""
    clone_stats = []
    for clone, clone_tdata in clones.items():
        n_cells_total = clone_tdata.obs.query("clone.notnull()").shape[0]
        detection_rate = clone_tdata.obs["detection_rate"].mean() * 100
        clone_tdata = clone_tdata[clone_tdata.obs.tree.notnull()].copy()
        leaf_depth = py.utils.get_keyed_leaf_data(clone_tdata,"depth")
        site_edit_frac = petracer.tree.get_edit_frac(clone_tdata.obsm["characters"]) * 100
        edit_sites = clone_tdata.obsm["characters"].shape[1]
        mouse = clone.split("-")[0]
        id = clone_tdata.obs["clone"].values[0]
        tumor = clone_tdata.obs["tumor"].values[0]
        clone_stats.append({"mouse":mouse,"tumor":tumor,"clone":id,"name":clone,"n_cells":clone_tdata.n_obs,
                            "n_cells_total":n_cells_total,"avg_depth":np.mean(leaf_depth),"edit_sites":edit_sites,
                            "site_edit_frac":site_edit_frac,"detection_rate":detection_rate})
    clone_stats = pd.DataFrame(clone_stats)
    clone_stats.to_csv(results_path / "clone_stats.csv",index = False)


def calculate_fitness_correlation(clones, use_clones):
    """Calculate correlation between fitness and features across clones."""
    fitness_corr = []
    for clone in use_clones:
        clone_tdata = clones[clone].copy()
        sample_fitness = clone_tdata.obs.groupby("sample")["fitness"].mean()
        clone_tdata.obs["fitness"] = (clone_tdata.obs["fitness"] / clone_tdata.obs["sample"].map(sample_fitness).astype(float)) * sample_fitness.mean()
        features = pd.DataFrame(clone_tdata.layers["counts"].toarray(),
                            columns = clone_tdata.var_names,index = clone_tdata.obs_names)
        density_features = clone_tdata.obsm["subtype_density"].drop(columns = ["Malignant"])
        features = features.join(density_features)
        features = features.join(clone_tdata.obs.filter(like="boundary_dist", axis=1))
        feature_types = ["expression"] * clone_tdata.n_vars + ["subtype_density"] * density_features.shape[1] + ["distance"] * 2
        if "phase" in clone_tdata.obs.columns:
            for phase in clone_tdata.obs["phase"].unique():
                features[phase] = clone_tdata.obs["phase"] == phase
            feature_types =  feature_types + ["cell_cycle"] * 3
        if "module_scores" in clone_tdata.obsm.keys():
            features = features.join(clone_tdata.obsm["module_scores"])
            feature_types = feature_types + ["module_score"] * clone_tdata.obsm["module_scores"].shape[1]
        clone_corr = petracer.utils.pearson_corr(clone_tdata.obs["fitness"], features).assign(clone = clone)
        clone_corr["dist_cor"] = petracer.utils.pearson_corr(clone_tdata.obs["tumor_boundary_dist"], features).assign(clone = clone)["cor"]
        clone_corr["feature_type"] = feature_types
        fitness_corr.append(clone_corr)
    fitness_corr = pd.concat(fitness_corr)
    fitness_corr.to_csv(results_path / "fitness_corr.csv",index=False)


def calculate_heritability(clones, use_clones):
    """Calculate phylogenetic autocorrelation for gene expression and hotspot modules."""
    heritability = []
    for clone in use_clones:
        clone_tdata = clones[clone].copy()
        py.tl.tree_neighbors(clone_tdata, max_dist = 20,depth_key="time",tree = "tree")
        clone_heritability = py.tl.autocorr(clone_tdata,connect_key="tree_connectivities",copy = True,layer = "counts").rename(columns = {"pval_norm":"p_value"})
        clone_heritability["q_value"] = sm.multitest.multipletests(clone_heritability["p_value"], method = "fdr_bh")[1]
        clone_heritability["feature_type"] = "expression"
        clone_heritability["clone"] = clone
        heritability.append(clone_heritability)
        if "module_scores" in clone_tdata.obsm.keys():
            tdata_subset = clone_tdata[clone_tdata.obs["hotspot_module"].notnull()].copy()
            clone_heritability = py.tl.autocorr(tdata_subset,connect_key="tree_connectivities",copy = True,keys = "module_scores").rename(columns = {"pval_norm":"p_value"})
            clone_heritability["q_value"] = sm.multitest.multipletests(clone_heritability["p_value"], method = "fdr_bh")[1]
            clone_heritability["feature_type"] = "hotspot_module"
            clone_heritability["clone"] = clone
            heritability.append(clone_heritability)
    heritability = pd.concat(heritability)
    heritability["feature"] = heritability.index
    heritability.to_csv(results_path / "heritability.csv",index=False)


# Plotting functions
def decoding_thumbnail(plot_name,experiment,fov,series,colors,crop,figsize = (2,2)):
    """Plot image thumbnail for spot decoding diagram."""
    img, _ = ft.io.read_fov(img_paths[experiment],fov = fov, series = series,colors = colors)
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    if colors[2] == 405:
        #img = ft.filters.unsharp_mask(img[:2], sigma = 10)
        spot_img = np.max(img[:2],axis = 1)
        spot_img = ft.filters.unsharp_mask(spot_img, sigma = 10)
        dapi_img = img[2,20]
        dapi_img = ft.filters.unsharp_mask(dapi_img, sigma = 100)
        img = np.stack([spot_img[0],spot_img[1],dapi_img],axis = 0)
    else:
        img = np.max(img,axis = 0)
        img = ft.filters.unsharp_mask(img, sigma = 10)
    ft.pl.imshow(img[:,crop[0]:crop[2],crop[1]:crop[3]],ax = ax,vmax = "p99.9")
    ax.axis('off')
    save_plot(fig,plot_name,plots_path,rasterize = True)


def plot_umap(plot_name, tdata, color, figsize = (2,2)):
    """Plot UMAP colored by a given variable."""
    fig, ax = plt.subplots(figsize = figsize, layout = "constrained")
    if color == "cell_subtype":
        palette = subtype_palette
    else:
        palette = module_palette
    sc.pl.umap(tdata, color = color, palette = palette, ax = ax,
        show = False, legend_loc=None, title = "")
    save_plot(fig, plot_name, plots_path, rasterize = True)


def plot_marker_umaps(plot_name,tdata,markers = None,figsize = (2.5, 7)):
    """Plot a grid of UMAPs colored by different markers."""
    cmap = mcolors.ListedColormap(['#d3d3d3'] * 10 +[plt.get_cmap('Reds')(i / 100) for i in range(100)] + ['#6b010e'] * 100)
    if markers is None:
        markers = ["Wnt7b", "C1qb", "Arg1", "Itgax", "Alox15", "Cd3g","Cd8a", "Cd4", "Foxp3", "Cd22","Flt3",
            "Cxcr2",'Ncr1', 'Siglech', "Shank3",  "Col5a2","Cxcl14", "Ebf1", "Sdc1", "Chil1", "Cyp4b1"]
    fig, axes = plt.subplots(len(markers)//3, 3, figsize=figsize,layout = "constrained",dpi = 600)
    axes = axes.flatten()
    for i, marker in enumerate(markers):
        sc.pl.umap(tdata, color=marker, color_map = cmap, layer='counts',
            colorbar_loc = None, frameon=False, ax = axes[i], show=False)
    save_plot(fig,plot_name,plots_path,rasterize = True)


def subtype_barplots(plot_name, tdata, figsize = (2,2.5)):
    """Plot barplots with counts and fractions of cell subtypes."""
    fig, axes = plt.subplots(2,1,figsize=figsize,layout = "constrained",
                         sharex = True)
    subtype_counts = tdata.obs.query("sample.isin(['M1-2','M2-1']) & within_tumor").groupby(
        ["tumor","cell_subtype"],observed = False).size().unstack().fillna(0)
    subtype_counts.plot(kind = "bar", stacked = True, color = subtype_palette, ax = axes[0],
                        legend=False, width = 0.8)
    subtype_fracs = subtype_counts.div(subtype_counts.sum(axis = 1), axis = 0) * 100
    subtype_fracs = subtype_fracs[reversed(subtype_fracs.columns)]
    subtype_fracs.plot(kind = "bar", stacked = True, color = subtype_palette,
                    ax = axes[1], legend=False, width = 0.8)
    axes[0].set_ylabel("Cell number")
    axes[1].set_ylabel("Percent")
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    save_plot(fig, plot_name, plots_path)


def plot_spatial(plot_name,tdata,key,cmap = "viridis",palette = None,vmin = None,vmax = None,
    spot_size = 40,regions = None,layer = None,mask_obs = None,groups = None,basis = "spatial",
    linestyle = "--",bar_length = None, colorbar_loc = None, edgecolor = None, figsize = (2,2)):
    """Plot spatial distribution of a given variable."""
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    if key in tdata.var_names:
        if layer is None:
            tdata.obs[f"{key}_expr"] = tdata[:,key].X.mean(axis = 1)
        else:
            tdata.obs[f"{key}_expr"] = tdata[:,key].layers[layer].mean(axis = 1)
        key = f"{key}_expr"
    if "density" in key:
        tdata.obs[key] = tdata.obsm["subtype_density"][key.replace("_density","")]
    sc.pl.spatial(tdata,color = key,cmap = cmap,frameon = False,title = "",vmin = vmin,vmax = vmax,
                  groups = groups,palette = palette,spot_size = spot_size,legend_loc = None,edgecolor = edgecolor,linewidth = .1,
                  ax = ax,show = False,colorbar_loc=colorbar_loc,layer = layer,mask_obs = mask_obs,basis = basis)
    if regions is not None:
        regions.plot(edgecolor = "black",facecolor = "none",legend = False,ax = ax,linewidth = .8, linestyle = linestyle)
    ax.add_artist(ScaleBar(dx = 1,units="um",location='lower right',
                           color = "black",box_alpha=.8,fixed_value = bar_length))
    ax.invert_yaxis()
    save_plot(fig,plot_name,plots_path,rasterize = True,transparent=True)


def plot_spatial_zoom(plot_name,tdata,sample = "M2",xlim = (0,500),ylim = (-1900,-1400),figsize = (2,3)):
    """Zoomed in spatial plot of a given sample."""
    # Load polygons
    polygons = gpd.read_file(data_path / f"{sample}_polygons.json").set_crs(None, allow_override=True)
    polygons.set_index("index", inplace=True)
    polygons = polygons.merge(tdata.obs,left_index=True,right_index=True,how="left")
    # Plot
    subtype_cmap = mcolors.ListedColormap([subtype_palette[i] for i in libraries["M1M2"].obs.cell_subtype.cat.categories])
    fig, ax = plt.subplots(1, 1, figsize=(2, 2),layout = "constrained",dpi = 1200)
    polygons.query('@xlim[0] <= centroid_x <= @xlim[1] & @ylim[0] <= centroid_y <= @ylim[1]').plot(ax = ax,
        column = "cell_subtype",cmap = subtype_cmap,legend=False, edgecolor="black", linewidth=0.1)
    plt.axis('off')
    ax.add_artist(ScaleBar(1, dimension="si-length", units="um", location="upper left"
                        , box_alpha=0, font_properties={'size': 10}, fixed_value = 100))
    save_plot(fig,plot_name,plots_path,rasterize=True, dpi = 1200)


def nhood_enrichment_heatmap(plot_name,tdata,figsize = (3,3)):
    """Plot neighborhood enrichment heatmap."""
    sq.gr.spatial_neighbors(tdata, coord_type="generic")
    sq.gr.nhood_enrichment(tdata, cluster_key="subtype_abbr")
    subtype_colors = [subtype_palette[subtype] for subtype in tdata.obs["cell_subtype"].cat.categories]
    fig, ax = plt.subplots(1, 1, figsize=figsize,layout = "constrained")
    sq.pl.nhood_enrichment(tdata, cluster_key="subtype_abbr", method="single",
        cmap="RdBu_r",vmin=-100, vmax=100, center = 0, ax = ax,
        palette = mcolors.ListedColormap(subtype_colors))
    save_plot(fig,plot_name,plots_path)


def subtype_position_ridgeplot(plot_name,tdata,figsize = (2.5,5)):
    """Plot ridgeplot of cell subtypes by distance from tumor boundary."""
    # Order of cell subtypes
    subtype_order = ['Endothelial','AT1/AT2','Club cell','Neutrophil','Alveolar fibroblast 2','ALOX15 macrophage',
        'B cell','NK','CD4 T cell','Treg','Exhausted CD8 T cell','CD11c macrophage','pDC','Alveolar fibroblast 1',
        'cDC','Malignant','Cancer fibroblast','ARG1 macrophage']
    # Get smoothed density by position
    densities = pd.concat([tdata.obsm["subtype_density"],tdata.obs],axis = 1)
    densities['interface_distance'] = np.where(densities['within_tumor'],
        densities['lung_boundary_dist'],-densities['tumor_boundary_dist'])
    densities = densities.query("tumor != 'M2-T3'").copy() # excluded due to small size
    densities['interface_distance_scaled'] = densities.groupby(
        ['sample','tumor','within_tumor'])['interface_distance'].transform(lambda x: x / x.abs().max())
    densities = densities.sort_values(by = "interface_distance_scaled")
    for subtype in subtype_order: # rolling average
        densities[subtype] = densities[subtype].rolling(2000).mean().transform(lambda x: x / x.max())
    densities = densities.query("Malignant.notnull()").sample(300).sort_values("interface_distance_scaled")
    # Plot
    fig, axes = plt.subplots(len(subtype_order),1, figsize = (2.5,5))
    for i, subtype in enumerate(subtype_order):
        ax = axes[i]
        sns.lineplot(data=densities, x='interface_distance_scaled', y=subtype, ax=ax, color="white", linewidth=2)
        ax.fill_between(densities['interface_distance_scaled'], densities[subtype], color=subtype_palette[subtype], clip_on=False)
        ax.text(-0.7, .22, subtype_abbr[subtype])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.patch.set_alpha(0)
        if i == len(subtype_order) - 1:
            ax.set_xlabel("Distance from lung/tumor\n interface (scaled)")
        else:
            ax.set_xlabel("")
            ax.set_xticks([])
        ax.set_xlim(densities['interface_distance_scaled'].min(), densities['interface_distance_scaled'].max())
    fig.subplots_adjust(hspace=-.3)
    save_plot(fig,plot_name,plots_path)


def replicate_corr_scatter(plot_name,tdata,x = "M1-S1", y = "M1-S2", tumor = "M1-T1",figsize = (2,2)):
    """Plot scatter of mean gene expression for two replicates."""
    mean_counts = sc.get.aggregate(tdata[tdata.obs.tumor == tumor], by=["sample"], func=["mean"], layer = 'counts')
    mean_counts = pd.DataFrame(mean_counts.layers['mean'], index =mean_counts.obs['sample'], columns=mean_counts.var.index).T
    corr = np.log10(mean_counts[x] +  0.01).corr(np.log10(mean_counts[y] +  0.01))
    fig, ax = plt.subplots(1, 1, figsize=figsize,layout = "constrained",dpi = 600)
    plt.scatter(mean_counts[x] + 0.01, mean_counts[y] +  0.01, color = "black", s = 2)
    plt.text(0.1,  10, f'r = {corr:.2f}', ha='center', va='center')
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel('MERFISH rep. 2\n(M1 T1 S2, counts/cell)')
    plt.xlabel('MERFISH rep. 1\n(M1 T1 S1, counts/cell)')
    save_plot(fig,plot_name,plots_path)


def subtype_expr_corr_heatmap(plot_name,tdata,figsize = (3.5, 3.5)):
    """Plot correlation of cell subtype gene expression between MERFISH and scRNAseq."""
    # Subtype orders
    scRNAseq_order = ['Malignant', 'Fibroblast', 'Macrophage', 'cDC', "pDC", 'Exhausted CD8 T cell',
                  'CD4 T cell', 'Treg', 'NK', 'B cell', 'Neutrophil', 'Endothelial']
    MERFISH_order = ['Malignant', 'Cancer fibroblast', 'Alveolar fibroblast 1', 'Alveolar fibroblast 2', 'ARG1 macrophage',
                    'CD11c macrophage','ALOX15 macrophage', 'cDC', 'pDC', 'Exhausted CD8 T cell', 'CD4 T cell', 'Treg',
                    'NK', 'B cell', 'Neutrophil', 'Endothelial', 'AT1/AT2', 'Club cell']
    # Get mean expression values
    seq_adata = ad.read_h5ad(data_path / "10x_4T1_primary.h5ad")
    mean_counts = sc.get.aggregate(tdata, by=["cell_subtype"], func=["mean"], layer = 'counts')
    seq_mean_counts = sc.get.aggregate(seq_adata[seq_adata.obs.cell_subtype.notnull(),tdata.var_names], by=["cell_subtype"], func=["mean"])
    mean_counts = ad.concat([mean_counts,seq_mean_counts],axis = 0,keys = ["MERFISH","scRNAseq"],label = "dataset")
    # Calculate correlation
    corr = pd.DataFrame(np.corrcoef(mean_counts.layers["mean"]),index = mean_counts.obs_names,columns = mean_counts.obs_names)
    corr = corr.loc[mean_counts.obs.dataset == "MERFISH",mean_counts.obs.dataset == "scRNAseq"]
    corr = corr.loc[MERFISH_order,scRNAseq_order]
    corr.index = corr.index.map(subtype_abbr)
    corr.columns = corr.columns.map(subtype_abbr)
    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=600, layout="constrained")
    sns.heatmap(corr,cmap = "RdBu_r",center = 0, ax = ax)
    ax.set_xticks(np.arange(len(corr.columns)) + 0.5)
    ax.set_xticklabels(corr.columns)
    ax.set_yticks(np.arange(len(corr.index)) + 0.5)
    ax.set_yticklabels(corr.index)
    ax.set_ylabel("MERFISH (mean counts/cell)")
    ax.set_xlabel("scRNA-seq (mean counts/cell)")
    save_plot(fig,plot_name,plots_path)


def subtype_marker_dotplot(plot_name,tdata,gene_subset = None,gene_order = None,subtype_order = None,
                           swap_axes = False,largest_dot = 50,figsize = (3, 2.5)):
    """Plot dotplot of subtype marker genes."""
    # Get gene order based on subtype expression
    if gene_order is None:
        mean_counts = sc.get.aggregate(tdata, by=["subtype_abbr"], func=["mean"], layer = 'counts')
        mean_counts = pd.DataFrame(mean_counts.layers['mean'], index = mean_counts.obs['subtype_abbr'], columns=mean_counts.var.index)
        gene_sort = pd.DataFrame(mean_counts.idxmax(axis=0), columns=['subtype_abbr'])
        gene_sort = pd.merge(gene_sort, pd.DataFrame(mean_counts.max(axis=0), columns=["max_exp"]),left_index=True, right_index= True)
        if gene_subset is not None:
            gene_sort["subset"] = gene_sort.index.isin(gene_subset)
            gene_order = list(gene_sort.sort_values(['subset','subtype_abbr', 'max_exp'], ascending=[True,True, False]).index)
        else:
            gene_order = list(gene_sort.sort_values(['subtype_abbr', 'max_exp'], ascending=[True, False]).index)
    if subtype_order is None:
        subtype_order = list(tdata.obs["subtype_abbr"].cat.categories)
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    dp = sc.pl.dotplot(tdata, var_names = gene_order,layer = 'normalized',categories_order = subtype_order,
                    groupby = "subtype_abbr",swap_axes = swap_axes,ax = ax, return_fig = True)
    dp.style(largest_dot = largest_dot, cmap = "Reds")
    dp.legend(show = False)
    dp.show()
    save_plot(fig,plot_name,plots_path)


def bulk_corr_scatter(plot_name,tdata,figsize = (2,2)):
    """Plot mean bulk RNA-seq vs MERFISH counts."""
    # Get mean counts
    bulk_rna = pd.read_csv(data_path / "GSE232196_CFa1_Bulk_Tissue_CPM.txt",  sep = "\t")
    bulk_rna["gene"] = bulk_rna["gene"].replace({'Chi3l1':"Chil1", "Faim3":"Fcmr", "Galntl4": 'Galnt18', '5730559C18Rik':'Inava'})
    bulk_rna = bulk_rna.query("gene in @tdata.var_names")
    mean_counts = pd.DataFrame(tdata.layers["counts"].mean(axis = 0).T, index = tdata.var_names, columns=["4T1_MERFISH"])
    mean_counts["4T1_lung_met"] = bulk_rna.set_index("gene")[['4T1-1 lung met','4T2-1 lung met', '4T3-1 lung met']].mean(axis = 1)
    # Plot
    corr = np.log10(mean_counts['4T1_MERFISH'] + 0.1).corr(np.log10(mean_counts['4T1_lung_met'] + 0.1))
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi = 600,layout = "constrained")
    plt.scatter(mean_counts['4T1_MERFISH']+ 0.1, mean_counts['4T1_lung_met']+ 0.1, color = "black", s = 2)
    plt.text(1,  1000, f'r = {corr:.2f}', ha='center', va='center')
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel('Bulk RNA-seq (CPM)')
    plt.xlabel('MERFISH (counts/cell)')
    save_plot(fig,plot_name,plots_path)


def volume_vs_z_scatter(plot_name,figsize):
    """Plot volume vs z position of cells colored by detection rate."""
    detection_rate = pd.read_csv(results_path / "cell_detection_rate.csv")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    sns.scatterplot(data = detection_rate,x = "volume",y = "centroid_z",hue = "detection_rate",palette="viridis",
                    s = 3,alpha = .5,ax = ax,legend=False)
    plt.xlabel("Volume (µm³)")
    plt.ylabel("Z position (µm)")
    plt.xscale("log")
    plt.axhline(-7,linestyle="--",color="black",linewidth = 1)
    plt.axhline(6,linestyle="--",color="black",linewidth = 1)
    save_plot(fig,plot_name,plots_path,rasterize = True)


def edit_frac_stacked_barplot(plot_name,clone_tdata,figsize = (1.2,1.7)):
    """Plot stacked barplot of edit fractions by site."""
    alleles = clone_tdata.obsm["characters"].melt(var_name="intID-site", value_name="allele")
    alleles["site"] = alleles["intID-site"].str.split("-").str[1].map(petracer.config.site_ids)
    edit_counts = alleles.groupby(["site","allele"]).size().unstack(fill_value=0)
    edit_order = list(range(1,9)) + [9,0]
    edit_counts = edit_counts.reindex(columns=edit_order)
    edit_fracs = edit_counts.div(edit_counts.sum(axis=1), axis=0) * 100
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi=600,layout = "constrained")
    edit_fracs.plot(kind='bar', stacked=True,color = [edit_palette[str(i)] for i in edit_fracs.columns],ax = ax,width = .9)
    plt.legend().remove()
    plt.xlabel("Edit site")
    plt.ylabel("Fraction of LMs (%)")
    plt.tight_layout()
    plt.xticks(rotation=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    save_plot(fig,plot_name,plots_path)


def detection_rate_hist(plot_name, clone_tdata, figsize = (1.5,1.5)):
    """Plot histogram of intBC detection rate."""
    detection_rate = clone_tdata.obs.copy()
    detection_rate["detection_pct"] = detection_rate["detection_rate"]*100
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.histplot(data = detection_rate,x = "detection_pct",color = "lightgray",ax = ax,bins = 10,alpha = 1,linewidth = .5)
    ax.axvline(60, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Detection rate (%)")
    plt.ylabel("Number of cells")
    plt.xlim(0,100)
    xticks = [0,detection_rate["detection_pct"].mean(),100]
    ax.set_xticks(xticks)
    ax.set_yticks([0,5e3])
    ax.set_yticklabels(["0","5e3"])
    save_plot(fig, plot_name, plots_path)


def clone_detection_violin(plot_name, tdata, figsize = (2.7,1.4)):
    """Plot violin plot of detection rate for each clone."""
    # Get detection rate
    detection_rate = tdata.obs.query("detection_rate.notnull()").copy()
    detection_rate["detection_pct"] = detection_rate["detection_rate"] * 100
    detection_rate["name"] = detection_rate["mouse"].astype(str) + " T" + detection_rate["clone"].astype(str)
    detection_rate = detection_rate.sort_values(["mouse","clone"])
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.violinplot(data = detection_rate,y = "detection_pct",x = "name",color = "lightgrey",
                ax = ax,linewidth=.5,linecolor="black",cut = 0,saturation=1,inner="quart",bw_adjust=2)
    ax.axhline(60, color='black', linestyle='--', linewidth=1)
    plt.ylim(0,100)
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("Detection rate (%)")
    plt.yticks([0,round(detection_rate["detection_pct"].mean()),100])
    save_plot(fig,plot_name,plots_path)


def clone_stats_barplot(plot_name,y,plot_clones = None,use_mice = ("M1","M2"),figsize = (2.7,1.4)):
    """Plot barplot for clone statistics."""
    stat_names = {"n_cells":"Number of cells","detection_rate":"Detection rate (%)",
                "edit_sites":"Number of edit sites","site_edit_frac":"Site with LM (%)",
                "pct_cells":"Percentage of\ntotal cells (%)"}
    clone_stats = pd.read_csv(results_path / "clone_stats.csv")
    if plot_clones is not None:
        clone_stats = clone_stats.query("name in @plot_clones").copy()
    if use_mice is not None:
        clone_stats = clone_stats.query("mouse in @use_mice").copy()
    if y == "pct_cells":
        clone_stats["pct_cells"] = clone_stats["n_cells"] / clone_stats["n_cells_total"] * 100
        clone_stats["n_cells_total"] = 100.0
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    if y in ["n_cells","pct_cells"]:
        sns.barplot(data = clone_stats,y = "n_cells_total",x = "name",color = "lightgray",ax = ax,saturation=1, edgecolor='black', linewidth=0.5)
        sns.barplot(data = clone_stats,y = y,x = "name",color = colors[1],ax = ax,saturation=1, edgecolor='black', linewidth=0.5)
    else:
        sns.barplot(data = clone_stats,y = y,x = "name",color = "lightgray",ax = ax,saturation=1, edgecolor='black', linewidth=0.5)
    plt.xticks(rotation=90)
    plt.ylabel(stat_names[y])
    plt.xlabel("")
    ax.set_yticks(np.append(ax.get_yticks(),round(clone_stats[y].mean())))
    if y == "pct_cells":
        ax.set_ylim(0,100)
    save_plot(fig,plot_name,plots_path)


def lm_proportion_barplot(plot_name,clones,use_clones = ('M1-T1','M1-T2','M1-T4','M2-T5'),figsize = (4,1.4)):
    """Plot barplot of mean LM proportions across selected clones."""
    # Calculate number of times each LM is installed in the phylogeny
    character_df_all = []
    for clone in use_clones:
        total_sites = {'EMX1': 0, 'HEK3': 0, 'RNF2': 0}
        character_dict = {}
        sites = pd.Series(clones[clone].obsm["characters"].columns).replace(".*-", "", regex = True)
        for edge in clones[clone].obst['tree'].edges:
            parent = edge[0]
            child = edge[1]
            parent_characters = clones[clone].obst['tree'].nodes[parent]["characters"]
            child_characters = clones[clone].obst['tree'].nodes[child]["characters"]
            for edit_site in total_sites.keys():
                parent_characters_site = [parent_characters[i] for i, x in enumerate(sites) if x == edit_site]
                child_characters_site = [child_characters[i] for i, x in enumerate(sites) if x == edit_site]
                parent_zero = [i for i, x in enumerate(parent_characters_site) if x == 0]
                total_sites[edit_site] += len(parent_zero)
                for character in [child_characters_site[i] for i in parent_zero if child_characters_site[i] != 0]:
                    dict_key = edit_site + "_" + str(character)
                    if dict_key not in character_dict.keys():
                        character_dict[dict_key] = 1
                    else :
                        character_dict[dict_key] += 1
        for edit_site in total_sites.keys():
            for character in [i for i in character_dict.keys() if edit_site in i]:
                character_dict[character] = (character_dict[character]/total_sites[edit_site])*100
        character_df = pd.DataFrame(character_dict.items(), columns = ['character', 'percentage'])
        character_df['clone'] = clone
        character_df_all.append(character_df)
    character_df_all = pd.concat(character_df_all)
    character_df_all['site'] = [petracer.config.site_names[i] for i in character_df_all['character'].replace("_\\d$","", regex = True)]
    character_df_all['allele'] = "LM" + character_df_all['character'].replace("^.*_","", regex = True)
    character_df_all['percentage_norm'] = character_df_all.groupby(['clone', 'site'])['percentage'].transform(
        lambda x: (x / x.sum())*100)
    character_df_all = character_df_all.sort_values("allele")
    # Plot barplot
    fig, axs = plt.subplots(ncols = 3, figsize=figsize,dpi = 600, layout = "constrained")
    axs = axs.flatten()
    edit_sites = list(petracer.config.site_names.values())
    for i in list(range(3)):
        sns.barplot(data = character_df_all[character_df_all['site'] == edit_sites[i]],x = "allele",
                    y = "percentage_norm",ax = axs[i],errorbar=None,hue = "allele",
                    palette = {f"LM{i}": color for i, color in edit_palette.items()},
                    edgecolor='black', linewidth=0.5, saturation=1)
        sns.scatterplot(data = character_df_all[character_df_all['site'] == edit_sites[i]],
                        x = "allele",y = "percentage_norm",ax = axs[i],color = "black",s=5,linewidth=0)
        if i != 0:
            axs[i].set_yticks([])
            axs[i].set_ylabel('')
        if i == 0:
            axs[i].set_ylabel('Fraction of\nLMs (%)')
        axs[i].set_xlabel(edit_sites[i])
    for ax in axs:
        ax.tick_params(axis='x', rotation=90)
    save_plot(fig,plot_name,plots_path)


def plot_tree(plot_name,tdata,keys = None,cmaps = None,vmaxs = None,vmins = None,palettes = None,
              color_branches = True,polar = False, figsize = (2,2)):
    """Plot tree with annotations"""
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained",subplot_kw={"polar":polar})
    if color_branches:
        py.pl.branches(tdata, tree = "tree", depth_key="time",color="clade",polar = polar,linewidth = .3,ax = ax,palette = get_clade_palette(tdata))
    else:
        py.pl.branches(tdata,tree = "tree",depth_key="time",polar = polar,linewidth = .3,ax = ax)
    for i, key in enumerate(keys):
        if key == "characters":
            petracer.tree.plot_grouped_characters(tdata,width=.07,ax = ax,label = True)
        else:
            cmap = cmaps[i] if cmaps is not None else "viridis"
            vmax = vmaxs[i] if vmaxs is not None else None
            vmin = vmins[i] if vmins is not None else None
            palette = palettes[i] if palettes is not None else None
            py.pl.annotation(tdata,keys = keys[i],width = .2,cmap = cmap,vmax = vmax,vmin = vmin,palette = palette,ax = ax)
    save_plot(fig,plot_name,plots_path,rasterize = True,dpi = 2000)


def plot_tree_with_zoom(plot_name,clone_tdata,key = "clade",sample = "M1-S2",subset = None,
                        xlim = (4100, 4600),ylim = (1500, 2000),figsize = (2.5, 3)):
    """Plot clades in tree, tumor section, and zoomed-in section."""
    # Set up figure
    fig = plt.figure(figsize=figsize, dpi=600)
    gs = fig.add_gridspec(2, 2, height_ratios = (1.2,1))
    ax1 = fig.add_subplot(gs[0, :])  # Span all columns of the first row
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    # Plot section
    clade_palette = get_clade_palette(clone_tdata,key)
    sc.pl.spatial(clone_tdata[clone_tdata.obs['sample'] == sample].copy(),color = key,spot_size =40,legend_loc="none",
                        frameon=False,title = "",ax = ax2,palette=clade_palette,show = False,basis = "spatial_grid")
    ax2.invert_yaxis()
    ax2.add_artist(ScaleBar(dx = 1,units="um",location='lower right',
                        color = "black",box_alpha=.8))
    ax2.add_patch(mpl.patches.Rectangle((xlim[0],ylim[0]),xlim[1]-xlim[0],ylim[1]-ylim[0],fill = False,edgecolor = "black",linewidth = 0.5))
    # Plot zoom
    zoom_polygons = clone_tdata.obs.query("@xlim[0] <= x <= @xlim[1] and @ylim[0] <= y <= @ylim[1]").copy()
    petracer.plotting.plot_polygons(cells = zoom_polygons,color=key,edgecolor = "black",palette=clade_palette,linewidth=.05,ax = ax3)
    ax3.axis('off')
    # Plot tree
    if subset is not None:
        clone_tdata = clone_tdata[subset].copy()
    py.pl.branches(clone_tdata, depth_key="time",ax = ax1,linewidth=.3, tree = "tree"
                    ,color = key,palette=clade_palette)
    petracer.tree.plot_grouped_characters(clone_tdata,width=.07,ax = ax1,label = True)
    plt.tight_layout()
    save_plot(fig,plot_name,plots_path,rasterize = True,dpi = 1200)


def clades_and_subclades_with_zoom(clone_tdata,figsize = (2.5, 3)):
    """Plot clades and subclades with zoomed-in section."""
    # Load data
    polygons = gpd.read_file(data_path / "M1_polygons_grid.json").set_crs(None, allow_override=True).set_index("index")
    clone_tdata = clone_tdata.copy()
    clone_tdata.obs = clone_tdata.obs.merge(polygons[["geometry"]],left_index=True,right_index=True,how = "left")
    clone_tdata.obs = gpd.GeoDataFrame(clone_tdata.obs,geometry = clone_tdata.obs["geometry"],crs = None)
    clone_tdata.obs["x"] = clone_tdata.obsm["spatial_grid"][:,0]
    clone_tdata.obs["y"] = clone_tdata.obsm["spatial_grid"][:,1]
    # Clade
    plot_tree_with_zoom("M1-T1_clade_tree_with_zoom",clone_tdata,key = "clade",sample = "M1-S2",
                        xlim = (4100, 4600),ylim = (1500, 2000), figsize = figsize)
    # Subclade
    subset = ['17','18']
    subset_tdata = clone_tdata[clone_tdata.obs["clade"].isin(subset)].copy()
    subclades = py.tl.clades(subset_tdata, tree = 'tree', key_added = 'clades_at_depth', depth = 6.25, depth_key = "time", copy = True)
    sizes = subset_tdata.obs.groupby("clades_at_depth").size().sort_values(ascending = False)[0:15]
    subclades = subclades.query("clades_at_depth in @sizes.index").set_index("node")
    subclades['clade'] = (subclades.reset_index().index +1).astype(str)
    py.tl.clades(clone_tdata,clades = subclades["clade"].to_dict(),tree = "tree", key_added='subclade')
    plot_tree_with_zoom("M1-T1_subclade_tree_with_zoom",clone_tdata,key = "subclade",subset = subset_tdata.obs_names,
                        sample = "M1-S2",xlim = (4100, 4600),ylim = (1500, 2000), figsize = figsize)
    # Subsubclade
    subset = ['15']
    subset_tdata = clone_tdata[clone_tdata.obs["subclade"].isin(subset)].copy()
    subclades = py.tl.clades(subset_tdata, tree = 'tree', key_added = 'clades_at_depth', depth = 9.75, depth_key = "time", copy = True)
    sizes = subset_tdata.obs.groupby("clades_at_depth").size().sort_values(ascending = False)[0:15]
    subclades = subclades.query("clades_at_depth in @sizes.index").set_index("node")
    subclades['clade'] = (subclades.reset_index().index +1).astype(str)
    py.tl.clades(clone_tdata,clades = subclades["clade"].to_dict(),tree = "tree", key_added='subsubclade')
    plot_tree_with_zoom("M1-T1_subsubclade_tree_with_zoom",clone_tdata,key = "subsubclade",subset = subset_tdata.obs_names,
                        sample = "M1-S2",xlim = (4100, 4600),ylim = (1500, 2000), figsize = figsize)


def clade_extant_ribbon(plot_name,tdata,ylim = None,frac = False,figsize = (2,2)):
    """Plot cumulative number of extant cells by clade over time."""
    clade_extant = petracer.tree.n_extant(tdata,groupby = "clade",depth_key = "time")
    clade_colors = get_clade_palette(tdata)
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    cumcount = clade_extant.pivot_table(index='time', columns='clade', values='n_extant', aggfunc='sum', fill_value=0)
    cumcount = cumcount.loc[:,clade_colors.keys()].cumsum(axis=1)
    if frac:
        cumcount = cumcount.div(cumcount.max(axis=1), axis=0)
    for i, clade in enumerate(cumcount.columns):
        lower = cumcount.iloc[:, i - 1] if i > 0 else np.zeros_like(cumcount.index)
        upper = cumcount.iloc[:, i]
        ax.fill_between(cumcount.index, lower, upper, color = clade_colors[clade])
    plt.xlabel('Time (days)')
    plt.ylabel('Number of extant cells')
    # fit exponential curve
    if not frac:
        def exponential_func(x, a, b, c):
            return a * np.exp2(b * x) + c
        total_extant = clade_extant.groupby("time").sum().reset_index().query("time < 25")
        params, covariance = sp.optimize.curve_fit(exponential_func, total_extant["time"], total_extant["n_extant"], p0=(1, 0.5, 1))
        a, b, c = params
        x_fit = np.linspace(0, 39, 500)
        y_fit = exponential_func(x_fit, a, b, c)
        plt.plot(x_fit, y_fit, color = "black", linestyle = "--", linewidth = 1)
    # tick scientific notation
    if ylim is not None:
        plt.ylim(ylim)
        ax.set_yticks([0,1e4,2e4,3e4])
        ax.set_yticklabels(["0","1e4","2e4","3e4"])
    save_plot(fig, plot_name, plots_path)


def clade_reconstruction_3d(plot_name,clone_tdata,tumor_boundaries,figsize = (3,3)):
    """Plot 3D clade reconstruction."""
    # Get 3d positions
    sample_z = {"M1-S1":-1200,"M1-S2":-800,"M1-S3":400,"M1-S4":1200}
    clone_tdata.obsm["spatial_3d"] = (
        np.concatenate([clone_tdata.obsm["spatial_overlay"],
        clone_tdata.obs["sample"].map(sample_z).values[:,np.newaxis]],axis = 1))
    # Get initial tree
    initial_edges = [('B','1'),('B', '2'),('C', '3'),('C', '4'),('D', '5'),
        ('D', 'F'),('F', '6'),('F', '7'),('F', '8'),('F', '9'),
        ('E', '10'),('E', '11'),('E', '12'),('E', '13'),('E', '14'),
        ('E', '15'),('E', '16'),('A', 'B'),
        ('A', 'C'),('A', 'D'),('A', 'E'),('root','17'),('root','18'),('root','A')]
    initial_tree = nx.DiGraph()
    initial_tree.add_edges_from(initial_edges)
    # Infer clade positions
    clades = clone_tdata.obs["clade"].dropna().unique()
    for node in nx.dfs_postorder_nodes(initial_tree):
        mean_position = np.mean(clone_tdata.obsm["spatial_3d"],axis = 0)
        if node in clades:
            initial_tree.nodes[node]["position"] = np.mean(clone_tdata[clone_tdata.obs["clade"] == node].obsm["spatial_3d"],axis = 0)
            initial_tree.nodes[node]["size"] = clone_tdata.obs.query("clade == @node").shape[0]
        else:
            children = list(initial_tree.successors(node))
            positions = [initial_tree.nodes[child]["position"] for child in children]
            initial_tree.nodes[node]["position"] = np.mean([np.mean(positions,axis = 0),mean_position],axis = 0)
            initial_tree.nodes[node]["size"] = clone_tdata.obs.query("clade == @node").shape[0]
    # Get clade colors
    clade_colors = get_clade_palette(clone_tdata)
    fig = plt.figure(figsize=figsize, dpi=600, layout = "constrained")
    ax = fig.add_subplot(111, projection='3d')
    # Plot tumor boundary
    tumor_boundaries = tumor_boundaries.query("tumor == 'M1-T1'").copy()
    tumor_boundaries["z"] = tumor_boundaries["sample"].map(sample_z)
    verts = []
    for _, row in tumor_boundaries.iterrows():
        sample_verts = np.array(row.geometry.xy)
        sample_verts = np.stack([sample_verts[0],sample_verts[1],np.ones_like(sample_verts[0]) * row["z"]/1.5],axis = 1)
        verts.append(sample_verts)
    poly = m3d.art3d.Poly3DCollection(verts,facecolor="lightgray",alpha = .3,edgecolor = "black",linewidth = 0)
    ax.add_collection3d(poly)
    poly = m3d.art3d.Poly3DCollection(verts,facecolor=(0, 0, 0, 0),edgecolor = "darkgray",linewidth = .2)
    ax.add_collection3d(poly)
    # Plot early splits
    for edge in initial_tree.edges():
        start = initial_tree.nodes[edge[0]]["position"]
        end = initial_tree.nodes[edge[1]]["position"]
        c = "gray"
        if edge[1] in clade_colors.keys():
            c = clade_colors[edge[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                [start[2], end[2]],c = c,linewidth = 1.5)
        # plot sphere
        if c != "gray":
            sphere_radius = 50
            sphere_center = end
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + sphere_center[0]
            y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + sphere_center[1]
            z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sphere_center[2]
            ax.plot_surface(x, y, z, color=c, alpha=1)
    # add scale bar
    ax.plot([-2500, -2500],[7000, 6500],[-750, -750],c = "black",linewidth = 1.5)
    ax.plot([-2500, -2000],[7000, 7000],[-750, -750],c = "black",linewidth = 1.5)
    ax.plot([-2500, -2500],[7000, 7000],[-750, -250],c = "black",linewidth = 1.5)
    # Configure plot
    ax.set_aspect('equal')
    ax.view_init(elev=25, azim=240)
    ax.axis('off')
    save_plot(fig, plot_name, plots_path,rasterize=True)


def plot_edits_spatial(plot_name,tdata,edits = None,figsize = (1.2,1.5)):
    """Plot spatial distribution of selected edits."""
    tdata = tdata[tdata.obs.tree.notnull()].copy()
    previous_edit = None
    for edit, value in edits.items():
        tdata.obs[edit] = tdata.obsm["characters"].loc[:,edit].astype(str)
        tdata.obs.loc[(tdata.obs[edit] != value) & (tdata.obs[edit] != "-1"),edit] = "0"
        if previous_edit is not None:
            tdata.obs.loc[(tdata.obs[previous_edit].astype(str) < "1") & (tdata.obs[edit] != "-1"),edit] = "0"
        previous_edit = edit
        plot_spatial(f"{plot_name}_{edit}_spatial",tdata,edit,
                     palette = edit_palette,edgecolor = "black",figsize = figsize)


def plot_expansion(plot_name,tdata,tumor_boundaries,node = 'node32961',figsize = (2,2)):
    """Plot phylogenetic and spatial location of expansion."""
    py.tl.clades(tdata ,tree = "tree",clades = {'node32961':"high_fit"},key_added="high_fit")
    clade_palette = get_clade_palette(tdata)
    # Plot tree
    leaves = np.array(py.utils.get_leaves(tdata.obst["tree"]))
    subset_tdata = tdata[tdata.obs_names.isin(leaves[16284-800:16284+400])].copy()
    py.tl.clades(subset_tdata,tree = "tree",clades = {'node32961':True},key_added="high_fit")
    subset_tdata.obs["high_fit"] = subset_tdata.obs["high_fit"].fillna(False)
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    py.pl.branches(subset_tdata,tree = "tree",depth_key="time",linewidth=.4,color="clade",palette=clade_palette,ax = ax)
    py.pl.annotation(subset_tdata,keys = "fitness",cmap = "magma",vmin = 1,vmax = 5,width = 0.1,label = "",ax = ax)
    py.pl.annotation(subset_tdata,keys = "high_fit",label = "",ax = ax,palette = {True:colors[2],False:"lightgray"},width = 0.1)
    save_plot(fig,f"{plot_name}_tree",plots_path)
    # Plot spatial
    section_tdata = clones["M1-T1"][clones["M1-T1"].obs["sample"] == "M1-S1"].copy()
    section_boundary = tumor_boundaries.query("tumor == 'M1-T1' & sample == 'M1-S1'").copy()
    plot_spatial(f"{plot_name}_spatial",section_tdata,"high_fit",palette = {"high_fit":colors[2]},
             regions = section_boundary,basis = "spatial_overlay",figsize = figsize)


def scatter_with_density(plot_name,tdata,x,y,x_label,y_label,mm = True,
        deciles = False,sample_n = None,figsize = (1.8,1.8)):
    """Scatter plot colored by density with optional regression lines."""
    df = tdata.obs.copy()
    if "density" in x:
        tdata.obs[x] = tdata.obsm["subtype_density"][x.replace("_density","")]
    if "density" in y:
        tdata.obs[y] = tdata.obsm["subtype_density"][y.replace("_density","")]
    if "tumor_boundary_dist" in x and mm:
        df[x] = df[x] / 1000
    df = df.query(f"{y}.notnull() and {x}.notnull()").copy()
    if sample_n is not None:
        df = df.sample(sample_n).copy()
    df["density"] = sp.stats.gaussian_kde(df[[x,y]].T)(df[[x,y]].T)
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.scatterplot(data = df,x = x,y = y,alpha = .5,s = 10, hue="density", palette="viridis",legend = False, ax = ax)
    sns.regplot(data=df, x=x, y=y, scatter = False, line_kws={"color": "black", "linewidth": 1.5}, ax = ax)
    if deciles:
        df['bin'] = pd.cut(df[x], 10)
        df_20th = df.groupby('bin',observed = False).apply(lambda g: g[g[y] < np.percentile(g[y], 20)]).reset_index(drop=True)
        sns.regplot(data=df_20th, x=x, y=y, scatter = False, line_kws={"color": "black", "linestyle": "--", "linewidth": .75})
        df_80th = df.groupby('bin',observed = False).apply(lambda g: g[g[y] > np.percentile(g[y], 80)]).reset_index(drop=True)
        sns.regplot(data=df_80th, x=x, y=y, scatter = False, line_kws={"color": "black", "linestyle": "--", "linewidth": .75})
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    r, p = sp.stats.pearsonr(df[x],df[y])
    ax.text(.95,.95,f"r = {r:.2f}", ha='right', va='top', transform=ax.transAxes)
    save_plot(fig,plot_name,plots_path,rasterize = True)


def module_phase_barplot(plot_name,tdata,figsize = (1.8,1.4)):
    """Stacked barplot of cell cycle phase distribution in each module."""
    fig,ax = plt.subplots(figsize = figsize, dpi = 600,sharex=True, layout = "constrained")
    phase_counts = tdata.obs.groupby(['hotspot_module','phase'], observed = True).size().unstack().fillna(0)
    phase_counts = phase_counts.div(phase_counts.sum(axis = 1),axis = 0)*100
    phase_counts = phase_counts.loc[:,["G2/M","S","G0/G1"]]
    phase_counts.plot(kind="bar", stacked=True, ax=ax, color=[phase_palette[phase] for phase in phase_counts.columns], width=0.8)
    plt.xlabel("Hotspot module")
    plt.ylabel("Phase (%)")
    plt.xticks(rotation=0)
    plt.xticks(np.arange(4),["1","2","3","4"])
    save_plot(fig,plot_name,plots_path)


def library_expr_corr_heatmap(plot_name,libraries,figsize = (2.5,3)):
    """Correlation heatmap for cell subtype expression between libraries."""
    # Get mean expression values
    common_genes = np.intersect1d(libraries["M1M2"].var_names,libraries["M3"].var_names)
    common_subtypes = libraries["M1M2"].obs["cell_subtype"].cat.categories[
        libraries["M1M2"].obs["cell_subtype"].cat.categories.isin(
            libraries["M3"].obs["cell_subtype"].cat.categories)]
    mean_counts = sc.get.aggregate(libraries["M1M2"][:,common_genes], by=["cell_subtype"], func=["mean"], layer = 'counts')
    mean_counts_m3 = sc.get.aggregate(libraries["M3"][:,common_genes], by=["cell_subtype"], func=["mean"], layer = 'counts')
    mean_counts = ad.concat([mean_counts,mean_counts_m3],axis = 0,keys = ["M1M2","M3"],label = "library")
    # Calculate correlation
    corr = pd.DataFrame(np.corrcoef(mean_counts.layers["mean"]),index = mean_counts.obs_names,columns = mean_counts.obs_names)
    corr = corr.loc[mean_counts.obs.library == "M1M2",mean_counts.obs.library == "M3"]
    corr = corr.loc[common_subtypes,common_subtypes]
    corr.index = corr.index.map(subtype_abbr)
    corr.columns = corr.columns.map(subtype_abbr)
    # Plot
    subtype_colors = [subtype_palette[i] for i in common_subtypes]
    subtype_labels = [subtype_abbr[i] for i in common_subtypes]
    g = sns.clustermap(corr,cmap = "RdBu_r",center = 0, figsize = figsize,row_cluster=False,col_cluster=False,vmax = 1,
        row_colors = subtype_colors, col_colors = subtype_colors,xticklabels = subtype_labels, yticklabels = subtype_labels)
    save_plot(g,plot_name,plots_path)


def plot_module_summary(plot_name,clone_tdata,figsize = (1.8,5.5)):
    """Plot expression, clade fraction, heritability and fitness of modules."""
    heritability = pd.read_csv(results_path / "heritability.csv")
    fig,axes = plt.subplots(4,1,figsize = figsize, dpi = 600,sharex=False, layout = "constrained",gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1.5]})
    # Module expression
    expr = sc.get.aggregate(clone_tdata[clone_tdata.obs.cell_subtype == "Malignant"], by=["hotspot_module"], func=["mean"], layer = 'normalized')
    expr = pd.DataFrame(sp.stats.zscore(expr.layers['mean'],axis = 0), index =expr.obs['hotspot_module'], columns=expr.var.index)
    expr = expr.loc[:,['Nes','Kif2c','Cdca2','Foxm1','Snai1','Sox9','Slc2a1','Vegfa','Wnt7b','Inava','Cdh1','Cd274','Cldn4','Fgfbp1']]
    sns.heatmap(expr.T,cmap = "RdBu_r",vmin = -2,vmax = 2,cbar = False,ax = axes[0],xticklabels = ["1","2","3","4"])
    axes[0].set_yticks(np.arange(len(expr.columns)) + .5)
    axes[0].set_yticklabels(expr.columns)
    axes[0].set_xlabel("")
    # Clade fraction
    clade_counts = clone_tdata.obs.groupby(["hotspot_module","clade"],observed = False).size().unstack().fillna(0)
    clade_counts = clade_counts.div(clade_counts.sum(axis=1), axis=0)*100
    clade_counts.plot(kind='bar', stacked=True,width = .9, color = list(petracer.config.discrete_cmap[18]),ax = axes[1])
    axes[1].legend().remove()
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(["1","2","3","4"])
    axes[1].set_ylabel("Clades (%)")
    axes[1].tick_params(axis='x', rotation=0)
    # Heritability
    sns.barplot(data = heritability.query("feature_type == 'hotspot_module'"),x = "feature",y = "autocorr",
                hue = "feature",palette = module_palette,saturation = 1,order = [1,2,3,4],ax = axes[2],legend = False)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Heritability\n(autocorrelation)")
    # Fitness violin
    sns.violinplot(data = clone_tdata.obs,x = "hotspot_module",y = "fitness",hue = "hotspot_module",ax = axes[3],
                palette = module_palette,legend = False,saturation=1,linewidth=.5,order= [1,2,3,4])
    # Save p-values
    pvals = skp.posthoc_mannwhitney(clone_tdata.obs.query("fitness.notnull()"), val_col="fitness", group_col="hotspot_module",p_adjust='fdr_bh')
    pvals = pvals.stack().rename_axis(['module1', 'module2']).reset_index(name='pval')
    pvals.to_csv(results_path / "M3_module_fitness_pvals.csv",index = False)
    save_plot(fig,plot_name,plots_path)


def module_subtype_heatmap(plot_name,clone_tdata,figsize = (2.5, 2)):
    """Heatmap of cell subtype density in each module."""
    clone_tdata = clone_tdata[clone_tdata.obs["hotspot_module"].notnull()].copy()
    df = sc.get.aggregate(clone_tdata, by=["hotspot_module"], func=["mean"], obsm = 'subtype_density')
    df = pd.DataFrame(df.layers['mean'], index =df.obs['hotspot_module'], columns=df.var.index)
    subtype_order = ['cDC','NK','Cancer fibroblast','Neutrophil','ARG1 macrophage','Tumor endothelial',
        'Alveolar fibroblast 1','CD11c macrophage','Exhausted CD8 T cell','Treg','B cell']
    df = df.loc[:,subtype_order]
    g = sns.clustermap(df,cmap="RdBu_r", z_score =1, figsize = figsize, row_cluster=False, col_cluster=False,
                row_colors= [module_palette[i] for i in df.index],
                col_colors = [subtype_palette[i] for i in df.columns],xticklabels = df.columns.map(subtype_abbr))
    g.ax_heatmap.set(ylabel=None)
    save_plot(g,plot_name,plots_path)


def hotspot_corr_heatmap(plot_name,figsize = (3.5,3.5)):
    """Plot hotspot correlation heatmap."""
    # Load data
    hotspot_corr = pd.read_csv(results_path / "M3_hotspot_corr.csv",index_col=0)
    hotsot_modules = pd.read_csv(results_path / "M3_hotspot_modules.csv",index_col=0)
    with open(results_path / "hotspot_linkage.npy", "rb") as f:
        linkage = np.load(f)
    # Plot
    gene_color = hotsot_modules["Module"].astype(str).map(module_palette)
    g = sns.clustermap(hotspot_corr, cmap = "RdBu_r",center = 0,figsize = figsize,vmax = 30,vmin = -30,dendrogram_ratio=.05,
                col_linkage=linkage,row_linkage=linkage,cbar_pos=None,col_colors = gene_color,row_colors=gene_color)
    y_ticks = hotspot_corr.index[g.dendrogram_row.reordered_ind]
    g.ax_heatmap.set_yticks(np.arange(len(y_ticks)) + .5)
    g.ax_heatmap.set_yticklabels(y_ticks, fontsize=6)  # Adjust 'fontsize' as needed
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.xaxis.set_ticks([])
    save_plot(g,plot_name,plots_path)


def module_transition_heatmap(plot_name,clone_tdata,figsize = (2.2, 2)):
    """Plot transition heatmap of hotspot modules."""
    # Get transition probabilities
    clone_tdata = clone_tdata[clone_tdata.obs.tree.notnull()].copy()
    py.tl.tree_neighbors(clone_tdata, max_dist = 20,depth_key="time")
    neighbors = py.tl.compare_distance(clone_tdata, dist_keys = ["tree"])
    neighbors = neighbors.merge(clone_tdata.obs["hotspot_module"],left_on = "obs1",right_index = True).rename(columns = {"hotspot_module":"module1"})
    neighbors = neighbors.merge(clone_tdata.obs["hotspot_module"],left_on = "obs2",right_index = True).rename(columns = {"hotspot_module":"module2"})
    module_counts = neighbors.groupby(["module1","module2"],observed = False).size().unstack().fillna(0)
    module_counts = module_counts.div(module_counts.sum(axis = 1),axis = 0)
    # Plot
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.heatmap(module_counts,vmin = 0,vmax = .5,cmap = petracer.config.sequential_cmap)
    plt.xlabel("To Hotspot module")
    plt.ylabel("From Hotspot module")
    save_plot(fig,plot_name,plots_path)


def fgf1_cldn4_fitness_violin(plot_name,clone_tdata,figsize = (1.3, 1.8)):
    """Plot fitness distribution of cells binned by Fgf1 and Cldn4 expression."""
    # Get fitness data
    df = clone_tdata.obs.copy()
    df["Fgf1"] = clone_tdata[:,"Fgf1"].X.mean(axis = 1) > 1
    df["Cldn4"] = clone_tdata[:,"Cldn4"].X.mean(axis = 1) > 1
    df["expression"] = df["Fgf1"].map({True:"Fgf1+",False:"Fgf1-"}) + "/" + df["Cldn4"].map({True:"Cldn4+",False:"Cldn4-"})
    df = df.query("fitness.notnull()")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.violinplot(data = df,x = "expression",linecolor = "black",
        y = "fitness",color = colors[2],saturation=1,linewidth=.5)
    plt.xticks(rotation=90)
    plt.ylabel("Fitness")
    plt.xlabel("")
    save_plot(fig,plot_name,plots_path)
    # Save p-values
    pvals = skp.posthoc_mannwhitney(df, val_col="fitness", group_col="expression",p_adjust='fdr_bh')
    pvals = pvals.stack().rename_axis(['group1', 'group2']).reset_index(name='pval')
    pvals.to_csv(results_path / "M3_Fgf1_Cldn4_fitness_pvals.csv",index = False)


def subtype_fitness_corr_scatter(plot_name,clone = 'M1-T1',figsize = (2,2)):
    """Fitness corr vs distance corr scatter plot."""
    fitness_corr = pd.read_csv(results_path / "fitness_corr.csv")
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    clone_corr = fitness_corr.query("clone == @clone").copy()
    clone_corr["cell_subtype"] = clone_corr["feature"].str.replace("_density","")
    sns.regplot(data = clone_corr.query("feature_type == 'subtype_density'"),
                x = "dist_cor",y = "cor",scatter = False,color = "black",line_kws={"linewidth":1},ax = ax)
    sns.scatterplot(data = clone_corr.query("feature_type == 'subtype_density'"),x = "dist_cor",s = 10,
                    y = "cor",hue = "cell_subtype",palette = subtype_palette,legend = False,linewidth = 0,ax = ax)
    plt.xlabel("Dist. to tumor boundary\ncorrelation (Pearson)")
    plt.ylabel("Fitness correlation\n(Pearson)")
    save_plot(fig,plot_name,plots_path)


def fitness_corr_scatter(plot_name,clone = 'M3-T1',figsize = (2,2)):
    """Fitness corr vs distance corr scatter plot."""
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    fitness_corr = pd.read_csv(results_path / "fitness_corr.csv")
    clone_corr = fitness_corr.query("clone == @clone").copy()
    sns.scatterplot(data = clone_corr.query("feature_type.isin(['expression','subtype_density'])"),
                    x = "dist_cor",y = "cor",hue = "feature_type",legend = False, size = 10)
    plt.plot([.3,-.3],[-.3,.3],linestyle="--",color="black",linewidth = 1)
    for _, row in clone_corr.query("feature_type != 'distance'").iterrows():
        if abs(row["cor"]) > .1:
            label = row["feature"]
            label = subtype_abbr.get(label,label)
            plt.text(row["dist_cor"],row["cor"],label,size = 8)
    plt.xlabel("Dist. to tumor boundary\ncorrelation (Pearson)")
    plt.ylabel("Fitness correlation\n(Pearson)")
    plt.ylim(-.2,.2)
    save_plot(fig,plot_name,plots_path)


def fitness_corr_heatmap(plot_name,figsize = (1.2, 3)):
    """Heatmap of fitness predictors across clones."""
    fitness_corr = pd.read_csv(results_path / "fitness_corr.csv")
    fitness_corr["n_samples"] = fitness_corr.groupby("feature")["cor"].transform("count")
    fitness_corr["feature"] = fitness_corr["feature"].str.replace("_density","")
    mat = fitness_corr.pivot_table(index = "feature",columns = "clone",values = "cor")
    sig_mat = fitness_corr.pivot_table(index = "feature",columns = "clone",values = "q_value").map(lambda x: "*" if x < .05 else "")
    fig, axes = plt.subplots(3,1,figsize=figsize,dpi = 300, layout = "constrained",gridspec_kw={'height_ratios': [1, 2.2, .2]})
    for i, feature_type in enumerate(["subtype_density","expression","distance"]):
        use_features = fitness_corr.query("clone == 'M3-T1' & q_value < .05 & n_samples > 1 & feature_type == @feature_type").sort_values(
            "cor",ascending = False)["feature"].values
        sns.heatmap(mat.loc[use_features,["M3-T1","M1-T1","M1-T2","M1-T4","M2-T5"]], annot_kws={"ha": "center", "ma": "center","size": 6},cbar = False,
                    center = 0,ax = axes[i],cmap = "RdBu_r",vmax = .15,vmin = -.2,annot=sig_mat.loc[use_features,["M3-T1","M1-T1","M1-T2","M1-T4","M2-T5"]], fmt="",
                    yticklabels = False,xticklabels = False)
        for t in axes[i].texts:
            trans = t.get_transform()
            offs = mpl.transforms.ScaledTranslation(0, .3,
                            mpl.transforms.IdentityTransform())
            t.set_transform( offs + trans )
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")


def heritability_vs_corr_scatter(plot_name,clones,clone = "M3-T1",figsize = (2, 2)):
    """Heritability vs fitness correlation scatter plot."""
    # Load data
    clone_tdata = clones[clone].copy()
    fitness_corr = pd.read_csv(results_path / "fitness_corr.csv")
    heritability = pd.read_csv(results_path / "heritability.csv")
    clone_corr = pd.merge(heritability,fitness_corr[["clone","feature","cor"]],on = ["clone","feature"],
                      how = "left").query("clone == @clone & feature_type == 'expression'").copy()
    # Assign genes to hotspot modules
    module_corr = []
    for module in clone_tdata.obs.hotspot_module.dropna().unique():
        features = pd.DataFrame(clone_tdata.layers["counts"].toarray(),
                                columns = clone_tdata.var_names,index = clone_tdata.obs_names)
        corr = petracer.utils.pearson_corr(clone_tdata.obsm["module_scores"][module], features).assign(clone = clone)
        corr["hotspot_module"] = module
        module_corr.append(corr)
    module_corr = pd.concat(module_corr)
    module_corr["abs_cor"] = module_corr["cor"].abs()
    module_corr = module_corr.sort_values("abs_cor",ascending = False).drop_duplicates("feature")
    clone_corr = clone_corr.merge(module_corr[["feature","hotspot_module"]],on = "feature")
    # Plot
    fig, ax = plt.subplots(figsize=(2, 2),dpi = 600, layout = "constrained")
    sns.scatterplot(data = clone_corr,x = "autocorr",y = "cor",s = 10,
        hue = "hotspot_module",legend = False,palette = module_palette)
    for _, row in clone_corr.query("abs(cor) >.08").iterrows():
        plt.text(row["autocorr"], row["cor"], row["feature"], fontsize=8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Heritability\n(Moran's I)")
    plt.ylabel("Fitness corr. (Pearson)")
    save_plot(fig,plot_name,plots_path)


if __name__ == "__main__":
    # Load data
    print(f"Loading data from {data_path}")
    libraries, clones, use_clones = load_tdata()
    tumor_boundaries, module_boundaries = load_boundaries()
    print(f"Loaded {len(clones)} clones")
    # Run analysis
    generate_combined_umap(libraries["M1M2"])
    calculate_imaging_stats(libraries)
    calculate_clone_stats(clones)
    calculate_fitness_correlation(clones, use_clones)
    calculate_heritability(clones, use_clones)
    # Plot decoding thumbnails
    print("Plotting decoding thumbnails")
    decoding_thumbnail("M1-S1_spot_thumbnail", "M1-S1",40,"H0R1",[748, 637, 405],(1550, 1120, 2000, 1570))
    decoding_thumbnail("M1-S1_intBC_thumbnail", "M1-S1",40,"H1R2",[748, 637, 545],(1550, 1120, 2000, 1570))
    decoding_thumbnail("M1-S1_edit_thumbnail", "M1-S1",40,"H16R17",[748, 637, 545],(1550, 1120, 2000, 1570))
    # Plot UMAPs
    print("Plotting UMAPs")
    plot_umap("M1M2_subtype_umap", libraries["M1M2"], "cell_subtype")
    plot_marker_umaps("M1M2_celltype_marker_umaps",libraries["M1M2"])
    plot_umap("M3_subtype_umap", libraries["M3"], "cell_subtype")
    plot_umap("M3_module_umap", libraries["M3"], "hotspot_module")
    # Plot cell type distributions and counts
    print("Plotting cell type distributions and counts")
    subtype_barplots("M1M2_subtype_barplots", libraries["M1M2"])
    nhood_enrichment_heatmap("M1M2_nhood_enrichment_heatmap",libraries["M1M2"])
    subtype_position_ridgeplot("M1M2_subtype_position_ridgeplot",libraries["M1M2"])
    for section, figsize in [("M1-S2",(1,4)), ("M2-S1",(1.5, 1.2))]:
       section_tdata = libraries["M1M2"][libraries["M1M2"].obs["sample"] == section].copy()
       section_boundaries = tumor_boundaries.query("sample == @section").copy()
       plot_spatial(f"{section}_subtype_spatial",section_tdata,"cell_subtype",
                   regions = section_boundaries,basis = "spatial_overlay",bar_length = 1000,
                   palette = subtype_palette, spot_size = 20, figsize = figsize)
    for clone in ["M1-T1","M3-T1"]:
        plot_spatial(f"{clone}_subtype_spatial",clones[clone],"cell_subtype",basis = "spatial_grid",
            spot_size = 20, palette = subtype_palette, figsize = (5,2))
    plot_spatial_zoom("M2-T1_spatial_zoom",libraries["M1M2"],sample = "M2")
    # Plot expression by subtype
    print("Plotting expression by subtype")
    subtype_marker_dotplot("M1M1_expression_dotplot",libraries["M1M2"],
                           largest_dot = 30, swap_axes = True,figsize=(2.5, 13))
    subtype_marker_dotplot("M3_expression_dotplot",libraries["M3"],gene_subset = libraries["M1M2"].var_names,
                           largest_dot = 30, swap_axes = True,figsize=(2.5, 18))
    subtype_marker_dotplot("M1M2_subtype_marker_dotplot",libraries["M1M2"],
        gene_order= ["Wnt7b", "Shank3", "Chil1","Cyp4b1","Cxcr2","Col5a2","Cxcl14","Ebf1",'Ncr1', 'Siglech',"Flt3",
                    "Cd3g","Foxp3","Cd8a", "Cd4", "C1qb","Itgax","Gpr39", "Arg1","Alox15", "Cd22"],
        subtype_order = ['Malig.','Endo.','AT1/AT2','Club cell','Neu.','AF2','AF1','NK','pDC','cDC','Treg',
                        'CD8+ T','CD4+ T','CD11c Mac.','CAF','ARG1 Mac.','ALOX15 Mac.','B cell'])
    # Plot replicate and scRNA-seq comparisons
    print("Plotting replicate and scRNA-seq comparisons")
    replicate_corr_scatter("M1_replicate_corr_scatter",libraries["M1M2"])
    subtype_expr_corr_heatmap("M1M2_scRNAseq_corr_heatmap",libraries["M1M2"],figsize = (3.5, 3.5))
    bulk_corr_scatter("M1M2_bulk_corr_scatter",libraries["M1M2"])
    library_expr_corr_heatmap("M1M2_M3_corr_heatmap",libraries,figsize = (2.5,3))
    # Plot statistics for each clone
    print("Plotting clone tracing statistics")
    volume_vs_z_scatter("M1-T1_volume_vs_z_scatter",(1.5,1.5))
    edit_frac_stacked_barplot("M1-T1_edit_frac_barplot",clones["M1-T1"])
    detection_rate_hist("M1-T1_detection_hist", clones["M1-T1"], figsize = (1.7,1.5))
    clone_detection_violin("M1M2_clone_detection_violin",libraries["M1M2"],figsize = (2.4,1.6))
    for stat in ["edit_sites","site_edit_frac","n_cells","pct_cells"]:
        figsize = (2.4,1.6) if stat in ["edit_sites","site_edit_frac"] else (1.4,1.6)
        plot_clones = None if stat in ["edit_sites","site_edit_frac"] else use_clones
        clone_stats_barplot(f"M1M2_clone_{stat}_barplot",stat,plot_clones=plot_clones,figsize = figsize)
    lm_proportion_barplot("M1M2_LM_proportion_barplot",clones)
    # Plot phylogenetic trees
    print("Plotting phylogenetic trees")
    for clone in ["M1-T1","M1-T2","M1-T4","M2-T5","M3-T1"]:
        figsize = (5,2) if clone in ["M1-T1","M3-T1"] else (3,2.5)
        spot_size = 20 if clone == "M2-T5" else 40
        plot_spatial(f"{clone}_clades_spatial",clones[clone],"clade", spot_size = spot_size,
            palette = get_clade_palette(clones[clone]), figsize = figsize)
        plot_tree(f"{clone}_tree_with_characters",clones[clone],keys = ["characters"],figsize = (3,2))
        distance_comparison_scatter(f"{clone}_spatial_vs_phylo_scatter",plots_path,clones[clone],
            x = "tree",sample_n = 100000,y = "spatial",mm = True, figsize = (2.1,2))
    clades_and_subclades_with_zoom(clones["M1-T1"],figsize = (2.5, 3))
    # Plot fitness and correlates
    print("Plotting fitness and correlates")
    for clone in ["M1-T1","M3-T1"]:
        plot_spatial(f"{clone}_fitness_spatial",clones[clone],"fitness",cmap = "magma", figsize = (5,2))
    plot_tree("M1-T1_tree_with_fitness",clones["M1-T1"],keys = ["fitness"],cmaps = ["magma"],polar = True)
    plot_tree("M3-T1_tree_with_fitness",clones["M3-T1"],keys = ["fitness","hotspot_module"],polar = True,
        cmaps = ["magma",None],palettes=[None,module_palette],figsize = (2.3,2.3))
    subtype_fitness_corr_scatter("M1-T1_subtype_fitness_corr_scatter",clone = "M1-T1",figsize = (2,2))
    fitness_corr_scatter("M3-T1_fitness_corr_scatter",clone = "M3-T1",figsize = (2.6,2))
    fgf1_cldn4_fitness_violin("M3-T1_Fgf1_Cldn4_fitness_violin",clones["M3-T1"],figsize = (1.3, 1.8))
    fitness_corr_heatmap("fitness_corr_heatmap",figsize = (1.2, 3))
    heritability_vs_corr_scatter("M3-T1_heritability_vs_corr_scatter",clones,clone = "M3-T1",figsize = (2, 2))
    # M1-T1 spatial plots
    print("Plotting M1-T1 spatial distributions")
    for key, cmap, limits in [("local_character_diversity","viridis",(.4,.9)),
                              ("character_dist_of_relatives","magma_r",(0,.5)),
                              ("cDC_density","Blues",(0,750))]:
        plot_spatial(f"M1-T1_{key}_spatial",clones["M1-T1"],key,cmap = cmap,colorbar_loc = "right",
                    vmin = limits[0], vmax = limits[1],figsize = (5,2))
    # M1-T1 section 1
    section_tdata = clones["M1-T1"][clones["M1-T1"].obs["sample"] == "M1-S1"].copy()
    section_tdata.obs["malignant"] = section_tdata.obs["cell_subtype"] == "Malignant"
    for key, vmax in zip(["Cldn4","Fgfbp1"],[50,20],strict = True):
        plot_spatial(f"M1-T1_{key}_spatial", section_tdata, key,vmax = vmax,layer = "counts",
                    cmap = "Reds", mask_obs = "malignant", figsize=(1.7, 1.7))
    plot_edits_spatial("M1-T1",section_tdata,edits = {"intID2147-HEK3":"3","intID673-EMX1":"2","intID606-EMX1":"7"})
    plot_expansion("M1-T1_expansion",clones["M1-T1"],tumor_boundaries,node = "node32961")
    # M1-T1 section 2
    section_tdata = clones["M1-T1"][clones["M1-T1"].obs["sample"] == "M1-S2"].copy()
    section_boundary = tumor_boundaries.query("tumor == 'M1-T1' & sample == 'M1-S2'").copy()
    for key in ["Endothelial", "CD11c macrophage", "ARG1 macrophage", "Cancer fibroblast"]:
        name = key.replace(" ", "_").lower()
        plot_spatial(f"M1-T1_{name}_spatial", section_tdata, "cell_subtype",regions = section_boundary,
            palette=subtype_palette, spot_size=30, groups=[key],basis = "spatial_overlay", figsize=(1.7, 1.7))
    # M1-T1 scatter plots
    print("Plotting M1-T1 scatter plots")
    distance_comparison_scatter("M1-T1_spatial_vs_character_scatter",plots_path,clones["M1-T1"],
        x = "character",y = "spatial",mm = True, sample_n = 100000, figsize = (2.1,2))
    distance_comparison_scatter("M1-T1_phylo_vs_character_scatter",plots_path,clones["M1-T1"],
        x = "character",y = "tree",sample_n = 100000, figsize = (2.1,2))
    scatter_with_density("M1-T1_dist_vs_fitness_scatter",clones["M1-T1"],"tumor_boundary_dist","fitness",
        "Dist. to tumor boundary (mm)","Fitness",deciles = True)
    scatter_with_density("M1-T1_dist_vs_character_dist_scatter",clones["M1-T1"],"tumor_boundary_dist","character_dist_of_relatives",
        "Dist. to tumor boundary (mm)","Mean neighbor character distance (k = 20)",deciles = True)
    scatter_with_density("M1-T1_fitness_metric_scatter",clones["M1-T1"],"character_dist_of_relatives","fitness",
        "Mean neighbor character distance (k = 20)","Fitness",figsize = (1.77,1.8))
    scatter_with_density("M1-T1_DC_vs_dist_scatter",clones["M1-T1"],"tumor_boundary_dist","cDC_density",
        "Distance to tumor boundary (mm)","cDC density",figsize = (2,1.8))
    scatter_with_density("M1-T1_fitness_vs_DC",clones["M1-T1"],"cDC_density","fitness",
        "cDC density","Fitness",figsize = (1.8,1.8))
    # Plot M1-T1 clade analysis
    print("Plotting M1-T1 clade analysis")
    clade_extant_ribbon("M1-T1_clade_ribbon",clones["M1-T1"],ylim = (0,30000))
    clade_reconstruction_3d("M1-T1_3d_reconstruction",clones["M1-T1"],tumor_boundaries)
    # Plot module analysis
    print("Plotting hotspot module analysis")
    plot_spatial("M3_modules_spatial",clones["M3-T1"],"hotspot_module",basis = "spatial_grid", spot_size = 20,
        palette = module_palette, regions = module_boundaries, linestyle = "-", figsize = (5,2))
    plot_module_summary("M3-T1_module_summary",clones["M3-T1"])
    module_subtype_heatmap("M3-T1_module_subtype_density_heatmap",clones["M3-T1"],figsize = (2.5, 2))
    module_phase_barplot("M3-T1_phase_barplot",clones["M3-T1"],figsize = (1.8,1.4))
    hotspot_corr_heatmap("M3-T1_hotspot_corr",figsize = (3.5,3.5))
    module_transition_heatmap("M3-T1_module_transition_heatmap",clones["M3-T1"],figsize = (2.2, 2))
    # Plot M3-T1 section 1
    print("Plotting M3-T1 spatial distributions")
    section_tdata = libraries["M3"][libraries["M3"].obs["sample"] == "M3-S4"].copy()
    section_tdata = section_tdata[section_tdata.obsm["spatial"][:,0] > 7800].copy()
    section_tdata.obs["malignant"] = section_tdata.obs["cell_subtype"] == "Malignant"
    for key, module in zip(["Foxm1","Vegfa","Cldn4","Fgf1","Fgfbp1"],["1","2","4","4","4"],strict = True):
        vmax = 10 if key in ["Fgf1","Fgfbp1"] else 50
        mask_obs = None if key in ["Vegfa"] else "malignant"
        plot_spatial(f"M3-S4_{key}_spatial",section_tdata,key,cmap = "Reds",spot_size = 20,vmax = vmax,
                    regions = module_boundaries.query('hotspot_module == @module'),
                    linestyle = "-",mask_obs = mask_obs,layer = "counts",figsize = (1.7,1.7))
    # Plot M3-T1 phase and cell type distributions
    print("Plotting M3-T1 phase and cell type distributions")
    plot_spatial("M3-S4_phase_spatial",section_tdata,"phase",palette=phase_palette,spot_size = 20,
            regions = module_boundaries.query("hotspot_module == '1'"),mask_obs = "malignant",
            linestyle = "-",figsize = (1.7,1.7))
    plot_spatial("M3-S4_endothelial_spatial",section_tdata,"cell_subtype",palette=subtype_palette,spot_size = 20,
            groups = ["Tumor endothelial","ARG1 macrophage","CAP1 endothelial","CAP2 endothelial"],
            regions = module_boundaries.query("hotspot_module == '2'"),linestyle = "-",figsize = (1.7,1.7))






