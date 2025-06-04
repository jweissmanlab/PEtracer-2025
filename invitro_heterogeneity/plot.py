import ast

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petracer
import scanpy as sc
import scipy as sp
import seaborn as sns
from petracer.config import colors, phase_palette, sequential_cmap
from petracer.utils import save_plot

base_path, data_path, plots_path, results_path = petracer.config.get_paths("invitro_heterogeneity")
petracer.config.set_theme()
np.random.seed(42)

# Helper functions
def get_cell_order(adata,clone_order = ("1","3","4","6","2","5")):
    """Get order of cells clustering within clones."""
    X = adata.obsm["intBC_detected"].copy()
    X["clone"] = adata.obs["clone"].values
    cell_order = []
    for clone in clone_order:
        group = X[X["clone"] == clone].drop(columns="clone")
        Z = sp.cluster.hierarchy.linkage(group, method="ward")
        ordered_idx = sp.cluster.hierarchy.leaves_list(Z)
        reordered_group = group.iloc[ordered_idx]
        cell_order.extend(reordered_group.index.tolist())
    return cell_order


# Plotting functions
def plot_umap(plot_name, adata, color, figsize = (2,2)):
    """Plot UMAP with colored by specified column."""
    fig, ax = plt.subplots(figsize=figsize,layout='constrained',dpi = 600)
    if color == "phase":
        sc.pl.umap(adata, color=color, palette=phase_palette, legend_loc=None, title="", ax=ax)
    else:
        sc.pl.umap(adata, color=color, legend_loc=None, title="", ax=ax)
    save_plot(fig, plot_name, plots_path, rasterize=True)


def diff_expr_volcano(plot_name,adata,figsize = (2,2)):
    """Plot volcano plot of differentially expressed genes."""
    adata_subset = adata[adata.obs.sample(1000).index,adata.var["mean"] > .1].copy()
    sc.tl.rank_genes_groups(adata_subset, groupby="leiden_cluster", groups=["1"], reference="0",layer = "normalized",method = "t-test")
    diff_genes = sc.get.rank_genes_groups_df(adata_subset, group="1", key="rank_genes_groups")
    diff_genes["-log10(pval_adj)"] = -np.log10(diff_genes["pvals_adj"]+1e-300)
    emt_genes = pd.read_csv(base_path / "reference" / "emtome_genes.csv",header = None).squeeze().tolist()
    diff_genes["emt"] = diff_genes["names"].isin(emt_genes)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained', dpi=600)
    sns.scatterplot(
        data=diff_genes,
        x="logfoldchanges",
        y="-log10(pval_adj)",
        hue="emt",
        alpha=.8,
        s=10,
        ax=ax,
        palette={True: colors[2], False: "gray"},
        legend=False,
    )
    for _, row in diff_genes[diff_genes["-log10(pval_adj)"] > 78].iterrows():
        ax.text(
            row["logfoldchanges"],
            row["-log10(pval_adj)"],
            row["names"],
            fontsize=8,
            ha='left',
            va='bottom'
        )
    plt.xlabel("log2 fold change")
    plt.ylabel("-log10(q-value)")
    save_plot(fig, plot_name, plots_path, rasterize=True)

def tumor_to_cell_mapping(plot_name,adata,figsize = (8, 5)):
    """Plot tumor to cell mapping with intBCs and Jaccard similarity."""
    tumor_order = ["M1-T1","M1-T2","M1-T3","M1-T4","M2-T5","M2-T2-2","M2-T2-1","M2-T1","M2-T3","M2-T4", "M2-T6"]
    cell_order  = get_cell_order(adata)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=figsize,
        gridspec_kw={'width_ratios': [.1,2.3,1],"height_ratios": [1.2, 1]},
    )
    # Jaccard similarity
    sns.heatmap(
        adata.obsm["tumor_clone_similarity"].loc[cell_order, tumor_order],
        yticklabels=False,
        xticklabels=False,
        ax=axes[1, 2],
        cbar=False,
        cmap="magma",
        vmax = 1
    )
    plt.xticks(fontsize=8)
    # Tumor intBCs
    tumor_intBCs = pd.read_csv(results_path / "tumor_clone_intBCs.csv",index_col=0)
    tumor_intBCs['intBCs'] = tumor_intBCs['intBCs'].apply(ast.literal_eval)
    tumor_intBCs = tumor_intBCs.explode('intBCs')
    tumor_intBCs['value'] = 1
    tumor_intBCs = tumor_intBCs.pivot_table(index='tumor', columns='intBCs', values='value', fill_value=0)
    tumor_intBCs.columns = tumor_intBCs.columns.str.replace("intID", "")
    tumor_intBC_order = tumor_intBCs.sum().sort_values(ascending=False).index
    sns.heatmap(tumor_intBCs.loc[tumor_order, tumor_intBC_order].T,
                cbar=False,
                ax=axes[0, 2],
                cmap=sequential_cmap,
                yticklabels=tumor_intBC_order,
                xticklabels=tumor_order,
    )
    axes[0,2].set_xlabel("")
    axes[0,2].set_ylabel("")
    axes[0, 2].xaxis.set_ticks_position('top')
    axes[0, 2].xaxis.set_label_position('top')
    axes[0, 2].tick_params(axis='x', rotation=90)
    for spine in axes[0, 2].spines.values():
        spine.set_visible(True)
    axes[0, 2].tick_params(axis='y', labelsize=7)
    # Cell intBCs
    cell_intBC_order = adata.obsm["intBC_detected"].astype(int).sum().sort_values(ascending=False).index
    cell_intBCs = adata.obsm["intBC_counts"].loc[cell_order, cell_intBC_order]
    cell_intBCs.columns = cell_intBCs.columns.str.replace("intID", "")
    g = sns.heatmap(
        cell_intBCs,
        cmap=sequential_cmap,
        vmax=100,
        yticklabels=False,
        xticklabels=False,
        cbar=False,
        ax=axes[1, 1],
    )
    for spine in axes[1, 1].spines.values():
        spine.set_visible(True)
    # Cell leiden clusters
    sns.heatmap(np.matrix(adata[cell_order].obs["leiden_cluster"].astype(int)).T * -1 + 1,
        cmap = [colors[2], colors[1]],
        yticklabels=False,
        xticklabels=False,
        cbar=False,
        ax=axes[1, 0]
    )
    # remove axes
    axes[0,1].axis('off')
    axes[0,0].axis('off')
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    save_plot(fig, plot_name, plots_path, rasterize=True)


if __name__ == "__main__":
    # Load data
    print(f"Loading data from {data_path}")
    adata = ad.read_h5ad(data_path / "4T1_invitro.h5ad")
    print("Data loaded")
    print("Generating plots...")
    plot_umap("leiden_cluster_umap", adata, "leiden_cluster", figsize=(2.5, 2.5))
    diff_expr_volcano("leiden_cluster_diff_volcano", adata, figsize=(2.5, 2.5))
    tumor_to_cell_mapping("tumor_to_cell_mapping", adata, figsize=(10, 6.8))
