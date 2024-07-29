'''Generate plots for experiments with preedited clones'''

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
import geopandas as gpd
import ast
import networkx as nx
import treedata as td
import pycea
import shapely as shp
import pickle

# Configure
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
ref_path = base_path / "reference"
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

# Load source
from src.config import colors,sequential_cmap,site_names,edit_ids,min_edit_prob, edit_cmap
from src.config import img_paths, edit_palette, preedited_clone_colors, fov_size, discrete_cmap, default_decoder
from src.utils import save_plot
from src.tree_utils import alleles_to_characters, plot_grouped_characters
from src.img_utils import load_fov, get_rgb, unsharp_mask, plot_region
from src.legends import edit_legend, add_cbar

### Constants ###
experiment_names = {"merfish_invitro":"MERFISH in vitro",
                    "10x_invitro":"10x in vitro","merfish_invivo":
                    "MERFISH in vivo","merfish_zombie":
                    "Zombie MERFISH in vitro"}
experiment_fovs = {"merfish_invitro":[-50,60,125,235],
                   "merfish_invivo":[1070,2585,1270,2785],
                   "merfish_zombie":[-300,140,-100,340]}
experiment_cells = {"merfish_invitro":[26,141,46.5,166]}    

### Helper functions ###
def add_colored_border(image, border_thickness=1, border_color=(255, 0, 0)):
    '''Add colored border to image'''
    image[:border_thickness, :, :] = border_color
    image[-border_thickness:, :, :] = border_color
    image[:, :border_thickness, :] = border_color
    image[:, -border_thickness:, :] = border_color
    return np.clip(image,0,1)

def get_edit_accuracy(alleles,x,y,min_prob = None):
    correct = []
    for site in site_names:
        if min_prob:
            filtered_df = alleles.query(f"{site}_prob > @min_prob")
            correct.append((filtered_df[x.format(site = site)] == filtered_df[y.format(site = site)]).mean())
        else:
            correct.append((alleles[x.format(site = site)] == alleles[y.format(site = site)]).mean())
    return np.mean(correct)

def calculate_detection_stats(experiments,prefix):
    '''Calculate detection stats for preedited experiments'''
    clone_whitelist = pd.read_csv(data_path / "preedited_clone_whitelist.csv",
                            keep_default_na=False,dtype={"clone":str})
    stats = []
    for experiment in experiments:
        # Load data
        alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",
                            keep_default_na=False,dtype={"clone":str})
        alleles = alleles.merge(clone_whitelist.rename(
                columns = {"EMX1":"EMX1_actual","RNF2":"RNF2_actual","HEK3":"HEK3_actual"}),
                on = ["intID","clone"],how = "left").query("whitelist")
        # Get stats
        experiment_stats = []
        for site in site_names.keys():
            if f"{site}_prob" in alleles.columns:
                detected = alleles.query(f"{site}_prob > @min_edit_prob")
            else:
                detected = alleles 
            experiment_stats.append(detected[[site,f"{site}_actual","clone","intID"]].rename(
                    columns = {site:"edit",f"{site}_actual":"true_edit"}).assign(site = site))
        experiment_stats = pd.concat(experiment_stats)
        experiment_stats["cells"] = experiment_stats["clone"].map(alleles.drop_duplicates(["clone","cellBC"]).groupby("clone").size())
        experiment_stats = experiment_stats.assign(detected = True,correct = experiment_stats["edit"] == experiment_stats["true_edit"],
                                                alleles = experiment_stats["cells"] * 3,experiment = experiment)
        stats.append(experiment_stats)
    stats = pd.concat(stats)
    # Integration stats
    int_stats = stats.groupby(["experiment","clone","intID"]).agg({"cells":"first","alleles":"first","correct":"sum","detected":"count"}).reset_index()
    int_stats["accuracy"] = int_stats["correct"]/int_stats["detected"]*100
    int_stats["recall"] = int_stats["detected"]/(int_stats["alleles"])*100
    int_stats.to_csv(results_path / f"{prefix}_integration_stats.csv",index = False)
    # Clone stats
    clone_stats = int_stats.groupby(["experiment","clone"]).agg({"cells":"first","alleles":"sum","correct":"sum","detected":"sum"}).reset_index()
    clone_stats["accuracy"] = clone_stats["correct"]/clone_stats["detected"]*100
    clone_stats["recall"] = clone_stats["detected"]/(clone_stats["alleles"])*100
    clone_stats.to_csv(results_path / f"{prefix}_clone_stats.csv",index = False)
    # Experiment stats
    experiment_stats = clone_stats.groupby("experiment").agg({"cells":"sum","alleles":"sum","correct":"sum","detected":"sum"}).reset_index()
    experiment_stats["accuracy"] = experiment_stats["correct"]/experiment_stats["detected"]*100
    experiment_stats["recall"] = experiment_stats["detected"]/(experiment_stats["alleles"])*100
    experiment_stats.to_csv(results_path / f"{prefix}_experiment_stats.csv",index = False)


def layout_spot_images(experiment,crop = [26,141,46.5,166],fov = 31):
    '''Create spots vs rounds matirx for spot images with the given region'''
    # Load data
    img_path = img_paths[f"preedited_{experiment}"]
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",keep_default_na=False,dtype={"clone":str})
    cell_spots = alleles.query("@crop[0] < global_x < @crop[2] and @crop[1] < global_y < @crop[3]").copy()
    rounds = pd.read_csv(data_path / "imaging_rounds.csv",keep_default_na=False)
    spot_rounds = rounds.query("type.isin(['common','integration','edit'])").copy().reset_index()
    int_codebook = pd.read_csv(ref_path / "integration_codebook.csv")
    int_codebook["bits"] = int_codebook["bits"].apply(lambda x: ast.literal_eval(x))
    fov_drift = pd.read_csv(img_path["analysis"]+f"/{fov}_drift.csv")
    cell_spots = cell_spots.merge(int_codebook,on = "intID",how = "left")
    # Create a blank canvas
    n_spots = len(cell_spots)
    n_rounds = len(spot_rounds)
    img_size = 20  
    margin = 2   
    canvas_height = n_spots * (img_size + margin) - margin
    canvas_width = n_rounds * (img_size + margin) - margin
    canvas = np.ones((canvas_height, canvas_width,3))
    # Process and position each image
    i = 0
    site_to_color = {site:mcolors.to_rgb(colors[3+i]) for i,site in enumerate(site_names.keys())}
    for series, series_rounds in spot_rounds.groupby("series", sort=False):
        series_imgs, _ = load_fov(fov, series_rounds, img_path["path"],img_path["file_pattern"], z_project=False)
        j = 0
        for _, spot in cell_spots.iterrows():
            z = int(spot.z)
            y = int(spot.y - fov_drift.query("series == @series")["y_drift"].values[0])
            x = int(spot.x - fov_drift.query("series == @series")["x_drift"].values[0])
            for k, round in series_rounds.reset_index().iterrows():
                img = unsharp_mask(series_imgs[k, z, y - 50:y + 50, x - 50:x + 50], 10)
                img = np.clip(img.astype(float)/(spot["intBC_intensity"]),0,1)
                img_section = np.stack([img[40:60, 40:60]]*3,axis = -1)
                row_start = j * (img_size + margin)
                col_start = (i + k) * (img_size + margin)
                bit = round["bit"]
                if bit in ["r52","r53"]:
                    img_section = add_colored_border(img_section,3,mcolors.to_rgb(colors[1]))
                elif bit in spot["bits"]:
                    img_section = add_colored_border(img_section,3,mcolors.to_rgb(colors[2]))
                elif round["type"] == "edit":
                    for site in list(site_names.keys()):
                        if round["site"] == site and spot[site] == round["edit"]:
                            img_section = add_colored_border(img_section,3,site_to_color[site])
                canvas[row_start:row_start + img_size, col_start:col_start + img_size,:] = img_section
            j += 1
        i += len(series_rounds)
    # save as numpy array
    np.save(results_path / f"{experiment}_spot_images.npy", canvas)

### Plotting functions ###
def plot_character_heatmap(plot_name,experiment,figsize = (4,3)):
    '''Plot character matrix annotated with clone for preedited experiment'''
    # Load data
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",
                        keep_default_na=False,dtype={"clone":str})
    clone_whitelist = pd.read_csv(data_path / "preedited_clone_whitelist.csv",keep_default_na=False)
    int_order = clone_whitelist.drop_duplicates("intID")["intID"]
    alleles = alleles.query("intID in @int_order")
    characters = alleles_to_characters(alleles,edit_ids,min_prob=.5,
                                    order = int_order,index = ["clone","cellBC"]).reset_index("clone")
    # Make tdata
    tree = nx.DiGraph()
    tree.add_node("root",time = 0)
    for leaf in reversed(characters.index):
        tree.add_edge("root",leaf)
        tree.add_node(leaf,time = 1)
    tdata = td.TreeData(obs = characters[["clone"]],obst = {"tree":tree},
                        obsm = {"characters":characters.drop(columns = "clone")})
    # Plot
    fig, ax = plt.subplots(figsize = figsize,dpi = 600,layout = "constrained")
    pycea.pl.branches(tdata,linewidth=0,depth_key="time")
    pycea.pl.annotation(tdata,keys=["clone"],width=4,palette = preedited_clone_colors,label = "Clone")
    plot_grouped_characters(tdata,width = 1,label = True,offset = 1)
    plt.legend(handles = edit_legend,ncol = 5,loc = "center",bbox_to_anchor = (0.5,-.32),columnspacing=.8,)
    plt.xticks(fontsize=9)
    ax.tick_params(axis='x', pad=0)
    plt.ylabel("Cells")
    fig.text(0.5, .15, 'Lineage cassette intBCs', fontsize=10, ha='center')
    save_plot(fig,plot_name,plots_path,svg = True,rasterize =True)  
    plt.close(fig)


def detection_stats_barplot(plot_name,experiment,figsize = (1.5,1.5)):
    '''Plot detection rate and accuracy for preedited experiment'''
    # Load data 
    stats = pd.read_csv(results_path / "preedited_experiment_stats.csv")
    stats = stats.query("experiment == @experiment")[["recall","accuracy"]].T.reset_index() #.set_index("experiment").T
    stats.columns = ["metric","value"]
    # Plot
    fig, ax = plt.subplots(figsize = figsize,layout = "constrained",dpi = 600)
    sns.barplot(data = stats,x = "metric",y = "value",hue = "metric",err_kws={"linewidth":1.5,"color":"black"},
                palette = {"recall":colors[1],"accuracy":colors[2]},ax = ax,saturation=1)
    # Set y tick labels on both sides
    ax.yaxis.tick_left()
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_label_position("left")
    ax.set_ylabel("LM detection rate (%)", color=colors[1])
    ax.tick_params(axis='y', colors=colors[1])
    # Create a twin y-axis for the right side
    ax_right = ax.twinx()
    ax_right.set_yticks([0, 25, 50, 75, 100])
    ax_right.set_ylabel("LM accuracy (%)", color=colors[2], rotation=270, labelpad=10)
    ax_right.tick_params(axis='y', colors=colors[2])
    ax_right.spines['right'].set_visible(True)
    # Set limit and label
    ax.set_ylim(0, 100)
    ax.xaxis.set_ticks([])
    ax.xaxis.labelpad = 10
    ax.set_xlabel(experiment_names[experiment])
    save_plot(fig,plot_name,plots_path)
    plt.close(fig)

def cells_per_clone(plot_name,experiment,figsize = (1.5,1.5)):
    '''Plot the number of cells per clone for a preedited experiment'''
    if experiment == "10x_invitro":
        cells = pd.read_csv(data_path / f"{experiment}_cells.csv")
    else:
        cells = gpd.read_file(data_path / f"{experiment}_cells.json")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained",dpi=600)
    n_cells = cells.groupby("clone").size().reset_index(name = "n_cells")
    sns.barplot(data=n_cells,x = "clone",y = "n_cells",palette = preedited_clone_colors,saturation=1)
    plt.ylabel("Nunber of cells")
    plt.xlabel("Clone")
    save_plot(fig,plot_name,plots_path)


def integration_confusion_matrix(plot_name,experiment,figsize=(1.2, 1.2)):
    '''Plot integration confusion matrix for preedited experiment'''
    # Load data
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",keep_default_na=False,dtype={"clone":str})
    # Generate confusion matrix
    total_ints = alleles.query("whitelist").groupby("clone").agg({"intID":"nunique","cellBC":"nunique"})
    total_ints["clone_total"] = total_ints["intID"] * total_ints["cellBC"]
    total_ints = total_ints["clone_total"].sum()
    whitelist_counts = alleles.whitelist.value_counts().values
    confusion = np.array([[whitelist_counts[0],whitelist_counts[1]],[total_ints - whitelist_counts[0],0]])
    # Plot
    fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=600)
    sns.heatmap(confusion, annot=True, fmt = 'd', cmap=sequential_cmap, square=True, cbar = False, annot_kws={"size": 9})
    plt.xticks([0.5, 1.5], ["True", "False"]);
    plt.yticks([0.5, 1.5], ["True", "False"]);
    plt.xlabel("Expected intBC");
    plt.ylabel("Detected intBC");
    ax.tick_params(axis='both', which='both', length=0)
    save_plot(fig,plot_name,plots_path)
    plt.close(fig)


def edit_confusion_matrix(plot_name,experiment,figsize=(6.5, 2.5)):
    '''Plot edit confusion matrix for preedited experiment'''
    # Load data
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",
                        keep_default_na=False,dtype={"clone":str})
    clone_whitelist = pd.read_csv(data_path / "preedited_clone_whitelist.csv",
                            keep_default_na=False,dtype={"clone":str})
    alleles = alleles.merge(clone_whitelist.rename(
            columns = {"EMX1":"EMX1_actual","RNF2":"RNF2_actual","HEK3":"HEK3_actual"}),
            on = ["intID","clone"],how = "left").query("whitelist")

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained", dpi=600, sharey=True)
    # Plot
    for i, site in enumerate(site_names):
        ax = axes[i]
        if f"{site}_prob" in alleles.columns:
            filtered_alleles = alleles.query(f"{site}_prob > @min_edit_prob").copy()
        else:
            filtered_alleles = alleles.copy()
        y = filtered_alleles[site+"_actual"].map(edit_ids[site]).values
        y_pred = filtered_alleles[site].map(edit_ids[site]).values
        accuracy = np.mean(y_pred == y) * 100
        cm = confusion_matrix(y, y_pred)
        cm = (cm / cm.sum(axis=1)[:, np.newaxis] * 100)
        sns.heatmap(cm.T, annot=True, ax=ax, fmt = '.1f', cmap=sequential_cmap, square=True, cbar = False, annot_kws={"size": 7})
        ax.set_title(f"{site_names[site]} ({accuracy:.1f}% accuracy)")
        if i == 0:
            ax.set_ylabel('Decoded LM')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xlabel('Actual LM')
    # Add colorbar
    cbar_ax = fig.add_axes([1, 0.2, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=sequential_cmap, 
        norm=mpl.colors.Normalize(vmin=0, vmax=100), orientation='vertical')
    cbar.outline.set_visible(False)
    cbar.set_label('Proportion of actual LM (%)', labelpad = 0)
    cbar.set_ticks(np.arange(0, 101, 20))
    cbar.set_ticklabels([f'{x}' for x in range(0, 101, 20)]) 
    save_plot(fig,plot_name,plots_path)
    plt.close(fig)


def intensity_vs_probability_scatterplot(plot_name,experiment,figsize = (2,2)):
    '''Plot intensity vs probability scatterplot for preedited experiment'''
    # Load data
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",
                        keep_default_na=False,dtype={"clone":str})
    clone_whitelist = pd.read_csv(data_path / "preedited_clone_whitelist.csv",
                            keep_default_na=False,dtype={"clone":str})
    alleles = alleles.merge(clone_whitelist.rename(
            columns = {"EMX1":"EMX1_actual","RNF2":"RNF2_actual","HEK3":"HEK3_actual"}),
            on = ["intID","clone"],how = "left").query("whitelist")
    # Pivot alleles
    alleles_long = []
    for site in site_names.keys():
        alleles["correct"] = alleles[site] == alleles[f"{site}_actual"]
        alleles_long.append(alleles[["intBC_intensity",f"{site}_prob","correct"]].rename(
            columns = {f"{site}_prob":"prob"}).assign(site = site))
    alleles_long = pd.concat(alleles_long)
    # Plot
    g = sns.JointGrid(data=alleles_long, x="intBC_intensity", y="prob", hue="correct", marginal_ticks=True, 
                  height=figsize[0], ratio=3, space=0.5)
    g.plot_joint(sns.scatterplot, s=5, alpha=.3, palette=[colors[2], "darkgray"], legend=False)
    g.ax_joint.set_xscale('log')
    g.plot_marginals(sns.histplot, element="step", palette=[colors[2], "darkgray"],bins = 20,alpha = .8,stat="probability")
    g.ax_joint.axhline(0.5, color='black', linestyle='--')
    handles = [mpatches.Patch(color="darkgray", label='Correct LM'),
            mpatches.Patch(color=colors[2], label='Incorrect LM')]
    g.figure.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), columnspacing=1.0)
    g.ax_joint.set_xlabel("Spot intensity");
    g.ax_joint.set_ylabel("LM probability");
    save_plot(g.figure,plot_name,plots_path,svg = True,rasterize=True)


def clone_edit_whitelist(plot_name,figsize = (2,1)):
    '''Plot heatmap with LMs for each preedited clone'''
    whitelist = pd.read_csv(data_path / "preedited_clone_whitelist.csv", keep_default_na=False)
    fig, ax = plt.subplots(1,3,figsize = figsize,dpi = 600,layout = "constrained")
    for i, site in enumerate(site_names.keys()):
        name = site_names[site]
        df = whitelist[["clone","intID",site]].copy()
        df["value"] = df[site].map(edit_ids[site])
        df["column"] = df[site].map(edit_ids[site])
        df = df[["clone","intID","column","value"]].pivot_table(
            index = "clone",columns = "column",values= "value",aggfunc = "first").fillna(-1)
        sns.heatmap(df,cmap = edit_cmap,ax = ax[i],cbar = False)
        for j in range(1, df.shape[0]):
            ax[i].axhline(j, color='black', linewidth=1)
        ax[i].set_xticks([])
        if i > 0:
            ax[i].set_yticks([])
            ax[i].set_ylabel("")
        else:
            ax[i].set_ylabel("Clone")
            ax[i].tick_params(axis='y', length = 0)
            ax[i].set_yticklabels(ax[i].get_yticklabels(), rotation = 0)
        ax[i].set_xlabel(name.replace("Edit site","Site"))
        for spine in ax[i].spines.values():
            spine.set_visible(True)
    fig.suptitle("LM representation",y = 1.1,fontsize = 10, x = .6)
    plt.subplots_adjust(wspace=0.1)
    save_plot(fig,plot_name,plots_path,svg = True)
    plt.close(fig)


def cell_masks(plot_name,experiment,crop = None,highlight = None,figsize = (3.5,3.5)):
    '''Plot cell masks for preedited experiment'''
    # Load data
    cells = gpd.read_file(data_path / f"{experiment}_cells.json")
    cells.crs = None
    # plot
    fig, ax = plt.subplots(figsize = figsize,layout = "constrained",dpi = 600)
    plot_region(img = None,crop = crop,cells = cells,palette=preedited_clone_colors,color = "clone",ax = ax)
    if highlight:
        ax.add_patch(mpatches.Rectangle((highlight[0],highlight[1]),highlight[2]-highlight[0],highlight[3]-highlight[1],
                                        fill = False,edgecolor = "black",linewidth = 2))
    ax.axis('off');
    save_plot(fig,f"{experiment}_slide",plots_path,svg = True,rasterize = True)
    plt.close(fig)


def image_with_masks(plot_name,experiment,crop,highlight = None,label_spots = False,linewidth = 1.5,figsize = (3.5,3.5)):
    '''Plot image with cell masks overlayed for preedited experiment'''
    # Load data
    img_path = img_paths[f"preedited_{experiment}"]
    cells = gpd.read_file(data_path / f"{experiment}_cells.json")
    cells.crs = None
    rounds = pd.read_csv(data_path / "imaging_rounds.csv")
    # Load and process image
    fov = cells[cells.within(shp.geometry.box(*crop))].iloc[0,:]
    extent = [fov.x_offset,fov.y_offset,fov.x_offset+fov_size,fov.y_offset+fov_size]
    fov_img, _ = load_fov(fov.fov,rounds.query("series == 'H0M1'"),img_path["path"],img_path["file_pattern"],z_project = False)
    z_projection = get_rgb(np.stack([
        unsharp_mask(fov_img[0].max(axis = 0),10),
        unsharp_mask(fov_img[1].max(axis = 0),10),   
        fov_img[4].max(axis = 0)],axis = 0),[99.9,99.9,100])
    # Plot
    fig, ax = plt.subplots(figsize = figsize,layout = "constrained",dpi = 600)
    plot_region(img = z_projection,extent=extent,crop = crop,cells = cells,
                palette=preedited_clone_colors,edgecolor = "clone",ax = ax,)
    if highlight is not None:
        ax.add_patch(mpatches.Rectangle((highlight[0],highlight[1]),highlight[2]-highlight[0],highlight[3]-highlight[1],
                                        fill = False,edgecolor = "white",linewidth=linewidth ))
    if label_spots:
        alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",keep_default_na=False,dtype={"clone":str})
        alleles = alleles.query("@crop[0] < global_x < @crop[2] and @crop[1] < global_y < @crop[3]")
        for _, row in alleles.iterrows():
            plt.text(row['global_x'], row['global_y'], str(row['intID']).replace("intID",""), 
                     fontsize=9, color='white', ha='right', va='bottom')
    ax.axis('off');
    save_plot(fig,plot_name,plots_path,svg = True,rasterize = True)
    plt.close(fig)


def plot_spot_images(plot_name,experiment,crop,figsize = (8.2,4)):
    '''Plot spot images within a given region'''
    spot_images = np.load(results_path / f"{experiment}_spot_images.npy")
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",keep_default_na=False,dtype={"clone":str})
    cell_spots = alleles.query("@crop[0] < global_x < @crop[2] and @crop[1] < global_y < @crop[3]").copy()
    xlabels = ["1","2"] + [f"{i}" for i in range(1,22)] + [f"{i}" for i in range(1,9)] * 3 + ["ES3","ES2","ES1"]
    xtick_colors = [colors[1]] * 2 + [colors[2]] * 21 + [colors[4]] * 8 + [colors[5]] * 8 + ["black"] * 8 + [colors[5],colors[4],"black"]
    xticks = np.linspace(10, spot_images.shape[1]-10, num=len(xlabels), dtype=int)
    fig, ax = plt.subplots(figsize = (8.2,4),layout="constrained",dpi=600)
    plt.imshow(spot_images)
    plt.xticks(xticks, xlabels,size = 10,rotation = 90);
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), xtick_colors):
        ticklabel.set_color(tickcolor)
    yticks = np.linspace(8, spot_images.shape[0]-12, num=len(cell_spots), dtype=int)
    plt.yticks(yticks, cell_spots.intID.str.replace("intID",""),size = 10);
    plt.ylabel("Lineage intBCs")
    plt.tick_params(axis='both', which='both', length=0)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    save_plot(fig,plot_name,plots_path,svg = True)
    plt.close(fig)


def umi_histogram(plot_name,experiment,figsize = (2.5,2.5)):
    '''Plot UMI histogram for preedited experiment'''
    # Load data
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",keep_default_na=False,dtype={"clone":str})
    # Plot
    fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=600,layout="constrained")
    sns.histplot(alleles, x="UMI", color = colors[1],linewidth=.3,bins = 20,log_scale = True,alpha = 1)
    plt.xlabel("UMI count")
    plt.ylabel("Lineage intBC count")
    plt.axvline(alleles["UMI"].median(), color="black", linestyle='--')
    plt.ylim(0, 12000)
    plt.text(6, 11000, f'Median = {alleles["UMI"].median():.0f}', color = 'black')
    save_plot(fig,f"{experiment}_umi_histplot",plots_path)
    plt.close(fig)


def barcode_mapping_heatmap(plot_name,figsize = (4,2.5)):
    '''Plot heatmap mapping 30nt barcodes to 183nt barcodes'''
    # Load data
    mapping = pd.read_csv(data_path / "preedited_barcode_mapping.csv")
    # Reshape data
    mapping["intID"] = mapping["intID"].str.replace("intID","").astype(int)
    mapping = mapping.sort_values("intID")
    mapping_wide = mapping.pivot(index='intID', columns='intBC', values='cell_frac').fillna(0)
    mapping_wide = mapping_wide.loc[mapping.intID, mapping.intBC]
    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=600,layout="constrained")
    sns.heatmap(mapping_wide.T, cmap=sequential_cmap, cbar=False,vmin = 0, vmax = 1,square=True)
    ax.set_xticks(np.arange(0.5, len(mapping.intID), 1),mapping.intID,size = 6,rotation = 90);
    ax.set_yticks(np.arange(0.5, len(mapping.intBC), 1),mapping.intBC,size = 6);
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='both', which='both', length=0)
    plt.xlabel("183nt MERFISH intBC");
    plt.ylabel("30nt sequencing intBC");
    # Add colorbar
    cbar_ax = fig.add_axes([1, 0.3, 0.03, 0.6])
    add_cbar(cbar_ax,sequential_cmap,[0,20,40,60,80,100],"Detection rate (%)")
    save_plot(fig,plot_name,plots_path)
    plt.close(fig)

def decoding_vs_intensity_lineplot(plot_name,experiments,figsize = (1.7,1.7)):
    # Load data
    spots = []
    for experiment in experiments:
        path = img_paths[f"preedited_{experiment}"]["analysis"].replace("Analysis/","")
        spots.append(pd.read_csv(f"{path}decoded_spots.csv",keep_default_na=False,index_col=0).assign(experiment = experiment))
    spots = pd.concat(spots)
    spots["decoded"] = spots["intBC_dist"] < 1.2
    # Get fraction decoded
    bins = np.logspace(np.log10(200), np.log10(15000), num=20, base=10).astype(int)
    spots["intensity_bin"] = pd.cut(spots["intBC_intensity"],bins=bins,labels = bins[:-1])
    frac_decoded = spots.groupby(["experiment","intensity_bin"],observed=True)["decoded"].mean().reset_index(name="frac_decoded")
    frac_decoded["frac_decoded"] = frac_decoded["frac_decoded"] * 100
    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=600, layout="constrained")
    sns.lineplot(data=frac_decoded, x="intensity_bin", y="frac_decoded", hue="experiment", ax=ax,palette = colors[1:3],legend=False)
    ax.set_xscale("log")
    ax.set_xlabel("Spots intensity")
    ax.set_ylabel("intBCs decoded (%)")
    save_plot(fig,plot_name,plots_path)

def decoder_accuracy_barplot(plot_name,experiment,figsize=(2.5, 2.5)):
    '''Plot decoder accuracy barplot for preedited experiment'''
    # Load data
    alleles = pd.read_csv(data_path / f"{experiment}_alleles.csv",
                        keep_default_na=False,dtype={"clone":str})
    clone_whitelist = pd.read_csv(data_path / "preedited_clone_whitelist.csv",
                            keep_default_na=False,dtype={"clone":str})
    alleles = alleles.merge(clone_whitelist.rename(
            columns = {"EMX1":"EMX1_actual","RNF2":"RNF2_actual","HEK3":"HEK3_actual"}),
            on = ["intID","clone"],how = "left").query("whitelist")
    # Calculate accuracy  
    brightest_round = get_edit_accuracy(alleles,"{site}_actual","{site}_brightest") * 100
    classifier = get_edit_accuracy(alleles,"{site}_actual","{site}") * 100
    threshold = get_edit_accuracy(alleles,"{site}_actual","{site}",min_prob = min_edit_prob) * 100
    accuracy = pd.DataFrame({"Max intensity":brightest_round,"LR classifier":classifier,"LR (p > 0.5)":threshold},index = [0])
    # Plot
    fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=600)
    sns.barplot(data = accuracy.T.reset_index(),x = "index",y = 0,color = colors[1])
    plt.ylim(95,100)
    plt.xticks(rotation = 90)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("")
    save_plot(fig,plot_name,plots_path)

def decoding_example_heatmaps(figsize = (2.1,2.1)):
    # weight matrix
    with open(ref_path / default_decoder, "rb") as f:
        decoder = pickle.load(f)
    weights = decoder["EMX1"].coef_[[7,0,4,1,5,6,2,3,8],:]
    # vectors
    intensities = np.array([327,33,17,434,51,14,1534,789,22])
    corrected_intensities = np.array([327,13,17,434,21,14,1534,624,22])
    norm_intensities = corrected_intensities / 2497
    logits = norm_intensities @ weights
    probs = np.exp(logits)/sum(np.exp(logits))
    # plot vectors
    fig, axes = plt.subplots(3,1,figsize=figsize,dpi = 600,layout = "constrained")
    green_cmap = mcolors.LinearSegmentedColormap.from_list('two_color_cmap', ['white', '#009E73'])
    for i, values in enumerate([intensities,norm_intensities,probs]):
        ax = axes[i]
        fmt = "d" if values.dtype == np.int64 else ".2f"
        sns.heatmap(values[:,np.newaxis].T, cmap=green_cmap, annot=True, fmt = fmt,vmin = 0,
                    cbar = False, ax = ax, annot_kws={"size": 8},square=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
    save_plot(fig, "decoding_example_intensities",plots_path)
    # plot weight matrix
    fig, ax = plt.subplots(figsize=figsize,dpi = 600,layout = "constrained")
    blue_red_cmap = mcolors.LinearSegmentedColormap.from_list('three_color_cmap', ['#1874CD','white', '#CD2626'])
    sns.heatmap(weights,annot=True,cmap=blue_red_cmap,center=0,ax = ax,annot_kws={"size": 8},cbar=False,fmt=".1f",square=True)
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
            spine.set_visible(True)
    save_plot(fig, "decoding_example_weights",plots_path)

# Generate plots
if __name__ == "__main__":
    clone_edit_whitelist("clone_edit_whitelist",(1.5,1))
    barcode_mapping_heatmap("preedited_barcode_mapping_heatmap",(4,2.5))
    decoding_example_heatmaps((2.1,2.1))
    experiments = ["10x_invitro","merfish_invitro","merfish_invivo","merfish_zombie"]
    calculate_detection_stats(experiments,"preedited")
    # All experiments
    for experiment in experiments:
        plot_character_heatmap(f"{experiment}_characters",experiment,figsize = (4,3))
        detection_stats_barplot(f"{experiment}_detection_stats_barplot",experiment,figsize = (1.5,1.5))
        integration_confusion_matrix(f"{experiment}_integration_confusion_matrix",experiment,figsize=(1.2, 1.2))
        edit_confusion_matrix(f"{experiment}_edit_confusion_matrix",experiment,figsize=(6.5, 2.5))
    # MERFISH experiments
    for experiment in ["merfish_invitro","merfish_invivo","merfish_zombie"]:
        intensity_vs_probability_scatterplot(f"{experiment}_intensity_vs_probability",experiment,figsize = (2.8,2.8))
        cell_masks(f"{experiment}_slide",experiment,highlight=experiment_fovs[experiment],figsize = (3.5,3.5))
        image_with_masks(f"{experiment}_fov",experiment,experiment_fovs[experiment],figsize = (2.5,2.5))
    # MERFISH invitro
    experiment = "merfish_invitro"
    decoder_accuracy_barplot(f"{experiment}_accuracy_barplot",experiment,figsize=(1.5, 1.7))
    decoding_vs_intensity_lineplot("decoding_vs_intensity_lineplot",
                                   experiments = ["merfish_invitro","merfish_zombie"],figsize = (1.7,1.7))
    image_with_masks(f"{experiment}_fov",experiment,experiment_fovs[experiment], 
                     highlight=experiment_cells[experiment],figsize = (2.5,2.5))
    image_with_masks(f"{experiment}_cell",experiment,experiment_cells[experiment],label_spots=True,linewidth=2,figsize = (1.7,1.7))
    #layout_spot_images("merfish_invitro",crop = experiment_cells[experiment],fov = 31)
    experiment = "merfish_invitro"
    plot_spot_images(f"{experiment}_spot_images",experiment,experiment_cells[experiment],figsize = (8.2,4))
    # 10x invitro
    experiment = "10x_invitro"
    umi_histogram(f"{experiment}_umi_histplot",experiment,figsize = (2.5,2.5))