import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import petracer
import seaborn as sns
from petracer.config import colors, discrete_cmap, edit_ids, edit_palette, sequential_cmap, site_names
from petracer.utils import save_plot
from scipy.cluster.hierarchy import leaves_list, linkage

base_path, data_path, plots_path, results_path = petracer.config.get_paths("insert_selection")
petracer.config.set_theme()

# Helper functions
def organize_enrichment_data():
    """Organize enrichment data for plotting"""
    enrichment = pd.read_csv(data_path / "allele_matrix_normalized_stats_q30_20bpwindow.txt",sep = "\t")
    enrichment["edit"] = enrichment.index
    enrichment_long = pd.melt(enrichment[["edit","EMX1_log2FC","RNF2_log2FC","HEK3_log2FC"]],id_vars = ["edit"],var_name = "site",value_name = "log2FC")
    enrichment_long["site"] = enrichment_long["site"].str.replace("_log2FC","")
    enrichment_long.to_csv(results_path / "insert_screen_log2FC.csv",index = False)


# Plotting functions
def crosshyb_vs_length_lineplot(plot_name,y,y_label,figsize = (2,2)):
    """Plot the crosshyb vs length lineplot"""
    results = pd.read_csv(results_path / "crosshyb_vs_length.csv")
    results["correct_pct"] = results["correct_frac"] * 100
    fig, ax = plt.subplots(figsize=figsize, dpi=600,layout="constrained")
    sns.lineplot(data=results.query("length > 1"),x="length",y=y,color = colors[1],ax=ax)
    ax.set_xlabel("Insert length")
    ax.set_ylabel(y_label)
    plt.xticks(range(2,9));
    save_plot(fig, plot_name, plots_path)


def crosshyb_heatmap(plot_name = None,site = "HEK3", metric = "free_energy",subset = None,highlight = None,
    lower = True,vmax = -12,vmin = -25,ticklabels = False,ax = None,figsize = (2,2)):
    """Plot the crosshyb heatmap"""
    # Load crosshyb for site
    crosshyb = pd.read_csv(results_path / "top_insert_crosshyb.csv",keep_default_na=False).query("site == @site")
    if metric == "probe_frac":
        crosshyb["probe_frac"] = np.log10(crosshyb["probe_frac"])
    crosshyb = crosshyb.pivot(index = "probe",columns="target",values=metric)
    order = crosshyb.index[leaves_list(linkage(crosshyb, method='average'))]
    # Select inserts
    inserts = pd.read_csv(results_path / "top_inserts.csv",keep_default_na=False)
    if subset is not None:
        selected = inserts.query(f"site == @site & {subset}")["insert"].tolist()
        order = [insert for insert in order if insert in selected]
    if subset == "final_8":
        order = list(edit_ids[site].keys())
    crosshyb = crosshyb.loc[order,order]
    # Plot heatmap
    if lower:
        mask = np.triu(np.ones_like(crosshyb, dtype=bool),k = 1)
    else:
        mask = np.tril(np.ones_like(crosshyb, dtype=bool),k = -1) 
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,dpi=600,layout = "constrained")
    cmap = sequential_cmap.reversed() if metric == "free_energy_diff" else sequential_cmap
    sns.heatmap(crosshyb, cmap=cmap, mask=mask, square=True,vmax=vmax,vmin=vmin,
            cbar = False,xticklabels=ticklabels,yticklabels=ticklabels,ax = ax)
    # Highlight selected inserts
    if highlight is not None:
        if subset == "final_8":
            selected = inserts.query(f"site == @site & {highlight}")["insert"].tolist()
            for i, insert in enumerate(order):
                if insert in selected:
                    color = edit_palette[str(edit_ids[site][insert])]
                    ax.add_patch(plt.Rectangle((i + .1, i + .1), .8, .8, fill=False, edgecolor=color, lw=2))
        else:
            ax.set_xlim(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1)
            ax.set_ylim(ax.get_ylim()[0] + 1, ax.get_ylim()[1] - 1) 
            selected = inserts.query(f"site == @site & {highlight}")["insert"].tolist()
            for i, insert in enumerate(order):
                if insert in selected:
                    ax.add_patch(plt.Rectangle((i - .1, i - .1), 1.2, 1.2, fill=False, edgecolor=colors[2], lw=1))
    # Format plot
    plt.xlabel("DNA with LMs")
    plt.ylabel("LM probes")
    if lower:
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
    else:
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
    save_plot(fig, plot_name, plots_path)


def insert_logfc_rankplots(figsize = (2.8,2)):
    """Plot the rank of log2FC for each insert across sites"""
    # Load data
    results = pd.read_csv(results_path / "insert_screen_log2FC.csv")
    results.sort_values("log2FC",ascending = False,inplace = True)
    results["rank"] = results.groupby("site").cumcount() + 1
    results.index = results["edit"].values
    # Highlight inserts
    top_inserts = pd.read_csv(results_path / "top_inserts.csv")
    top_inserts.drop(columns = ["within_10%","final_20"],inplace = True)
    top_inserts.rename(columns = {"insert":"edit"},inplace = True)
    top_inserts["highlight"] = "top"
    results = results.merge(top_inserts,on = ["site","edit"],how = "left").fillna("")
    results.loc[results.edit == "GCTGC","highlight"] = "GCTGC"
    results.loc[results.edit == "GTCAG","highlight"] = "GTCAG"
    # Plot
    for site in ["EMX1","RNF2","HEK3"]:
        fig, ax = plt.subplots(figsize = (2.8,2),dpi = 600)
        site_results = results.query("site == @site").sort_values("highlight",ascending = True)
        label = site_results.query("rank <= 4 or rank >= 1021")["edit"].tolist()
        label = label + ["GCTGC","GTCAG"]
        palette = {"top":colors[5],"GCTGC":colors[2],"GTCAG":colors[1],"":"lightgrey"}
        sns.scatterplot(data = site_results,x = "rank",y = "log2FC",hue = "highlight",linewidth = 0,palette = palette,s = 10,legend = False)
        for edit in label:
            row = site_results.query("edit == @edit").iloc[0]
            plt.text(row["rank"],row["log2FC"],row["edit"],fontsize = 10,ha = "left",va = "bottom",color = palette[row["highlight"]])
        plt.xlabel("Ranked 5nt insertion sequences")
        plt.ylabel("Log$_2$-fold LM insertion\n efficiency to average")
        plt.ylim(-7.5,3.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        save_plot(fig,f"{site}_insert_rankplot",plots_path)


if __name__ == "__main__":
    crosshyb_vs_length_lineplot("free_energy_diff_vs_length","free_energy_diff","$\Delta G$ - on-target $\Delta G$")
    crosshyb_vs_length_lineplot("correct_frac_vs_length","correct_pct","Correct probe bound (%)")
    vmax =8
    vmin = 0
    metric = "free_energy_diff"
    for site in site_names.keys():
        crosshyb_heatmap(f"{site}_free_energy_diff",site,metric=metric,highlight="final_20",
                        lower=True,figsize = (2,2),vmax = vmax,vmin = vmin)
        crosshyb_heatmap(f"{site}_20_free_energy_diff",site,metric=metric,subset="final_20",
                        lower=False,figsize = (2,2),vmax = vmax,vmin = vmin)
        crosshyb_heatmap(f"{site}_8_free_energy_diff",site,metric=metric,subset="final_8",highlight="final_8",
                        lower=True,figsize = (1.2,1.2),vmax = vmax,vmin = vmin,ticklabels=False)
    insert_logfc_rankplots(figsize = (2.8,2))
