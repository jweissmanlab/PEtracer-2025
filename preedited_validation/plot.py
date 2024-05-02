import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from shapely.geometry import box
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
import geopandas as gpd
import ast

# Configure
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
ref_path = base_path / "reference"
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

# Load source
from src.config import colors,discrete_cmap,min_spot_intensity,sequential_cmap,site_names,edit_names
from src.config import img_paths, preedited_clone_names
from src.utils import save_plot
from src.img_utils import load_fov, get_rgb, unsharp_mask

# Clone colors
clone_to_color = {i:discrete_cmap[4][i-1] for i in range(1,5)}
clone_to_color[0] = "lightgrey"

def add_colored_border(image, border_thickness=1, border_color=(255, 0, 0)):
    image[:border_thickness, :, :] = border_color
    image[-border_thickness:, :, :] = border_color
    image[:, :border_thickness, :] = border_color
    image[:, -border_thickness:, :] = border_color
    return np.clip(image,0,1)

def cells_per_clone(experiment):

    cells =gpd.read_file(results_path / f"{experiment}_cells_with_clone.json")

    fig, ax = plt.subplots(figsize=(2,2), layout="constrained",dpi=300)

    n_cells = cells.groupby("clone").size().reset_index(name = "n_cells")
    sns.barplot(data=n_cells,x = "clone",y = "n_cells",palette = clone_to_color,saturation=1)
    plt.ylabel("Number of cells")
    plt.xlabel("Clone")

    save_plot(fig,experiment + "_merfish_n_cells",plots_path)


def integration_precision_recall(experiment):
    spots = pd.read_csv(data_path / f"{experiment}_decoded_spots.csv",keep_default_na=False)
    cells =gpd.read_file(results_path / f"{experiment}_cells_with_clone.json")
    clone_whitelist = pd.read_csv(results_path / "preedited_clone_whitelist.tsv",sep="\t",keep_default_na=False)
    clone_whitelist["intID"] = clone_whitelist["mfID"]

    spots = spots.merge(cells,on = ["cell"],how = "left")
    spots = spots.merge(clone_whitelist.rename(
        columns = {"EMX1":"EMX1_actual","RNF2":"RNF2_actual","HEK3":"HEK3_actual"}),
        on = ["intID","clone"],how = "left")
    spots["whitelist"] = ~spots["EMX1_actual"].isna()
    spots = spots.sort_values("intBC_intensity",ascending = False).groupby(["cell","intID"]).first().reset_index()

    total_expected = len(cells.merge(clone_whitelist,on = "clone",how = "left"))
    thresholded_spots = spots.sort_values("intBC_intensity",ascending = False)
    thresholded_spots["total_whitelist"] = thresholded_spots["whitelist"].cumsum()
    thresholded_spots["total"] = np.arange(len(thresholded_spots)) + 1
    thresholded_spots["precision"] = (thresholded_spots["total_whitelist"]/thresholded_spots["total"]).rolling(window=50).mean() * 100
    thresholded_spots["recall"] = (thresholded_spots["total_whitelist"]/total_expected).rolling(window=50).mean() * 100

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(2.5,2), layout="constrained",dpi=300)

    # Plot the first line on ax1
    data = thresholded_spots.sample(500)
    sns.lineplot(data=data, x="intBC_intensity", y="precision", ax=ax, color=colors[1])
    ax.set_ylabel("Decoding precision (%)", color=colors[1])
    ax.tick_params(axis='y', colors=colors[1])
    ax.set_xlabel("Integration barcode intensity threshold")
    ax.invert_xaxis()
    ax.axvline(min_spot_intensity, linestyle="--", color="black")
    ax.set_xscale("log")
    ax.set_ylim(75, 100)

    # Create a second y-axis for the detection rate
    ax2 = ax.twinx()
    sns.lineplot(data=data, x="intBC_intensity", y="recall", ax=ax2, color=colors[2])
    ax2.set_ylabel("Decoding recall (%)", color=colors[2])
    ax2.tick_params(axis='y', colors=colors[2])
    ax2.axvline(min_spot_intensity, linestyle="--", color="black")
    ax2.spines['right'].set_visible(True)
    ax2.set_ylim(0, 100)

    save_plot(fig,experiment + "_precision_recall_threshold",plots_path)


# Per site confusion matrix
def edit_confusion(experiment,plot_name):

    spots = pd.read_csv(results_path / f"{experiment}_filtered_spots.csv",keep_default_na=False)

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), layout="constrained", dpi=300, sharey=True)

    for i, site in enumerate(site_names):
        ax = axes[i]
        y = spots[site+"_actual"].map(edit_names[site]).values
        y_pred = spots[site].map(edit_names[site]).values
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

    # add colorbar
    cbar_ax = fig.add_axes([1, 0.2, 0.03, 0.7])
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=sequential_cmap, 
        norm=mpl.colors.Normalize(vmin=0, vmax=100), orientation='vertical')
    cbar.outline.set_visible(False)
    cbar.set_label('Proportion of actual LM (%)', labelpad = 0)
    cbar.set_ticks(np.arange(0, 101, 20))
    cbar.set_ticklabels([f'{x}' for x in range(0, 101, 20)]) 

    save_plot(fig, plot_name, plots_path)

def integration_detection_violin(experiment, plot_name):

    spots = pd.read_csv(results_path / f"{experiment}_filtered_spots.csv",keep_default_na=False)
    cells =gpd.read_file(results_path / f"{experiment}_cells_with_clone.json")

    int_recall = spots.groupby(["clone","intID"]).size().reset_index(name="n_spots")
    int_recall["n_cells"] = int_recall["clone"].map(cells.groupby("clone").size())
    int_recall["recall"] = int_recall["n_spots"]/int_recall["n_cells"]*100

    fig, ax = plt.subplots(figsize=(2,2.3),layout="constrained",dpi=300)

    sns.violinplot(data=int_recall.query("intID != 'intID2078'"), y="recall",ax=ax,color = colors[1],inner = None,saturation=1,width=.95,linewidth = .5, cut = 1)
    sns.swarmplot(data=int_recall, y="recall", color='black', size=2)
    if experiment == "invitro":
        plt.text(.05, int_recall.query("intID == 'intID2078'")['recall'], "2078", fontsize=10, ha='left', va='bottom')
    ax.set_ylim(0,100)
    ax.set_xlabel("Integrations")
    ax.set_ylabel("MERFISH detection rate (%)")

    save_plot(fig, plot_name, plots_path)

def example_fov(experiment,plot_name,fov):

    spots = pd.read_csv(results_path / f"{experiment}_filtered_spots.csv",keep_default_na=False)
    cells =gpd.read_file(results_path / f"{experiment}_cells_with_clone.json")
    cells.crs = None
    rounds = pd.read_csv(data_path / "imaging_rounds.csv")

    x_offset = cells.query("fov == @fov")["x_offset"].values[0]
    y_offset = cells.query("fov == @fov")["y_offset"].values[0]
    scale = 1/cells.query("fov == @fov")["micron_per_pixel"].values[0]
    fov_box = box(x_offset, y_offset, x_offset + 2304/scale, y_offset + 2304/scale)
    fov_cells = cells[cells.geometry.intersects(fov_box)].copy()
    fov_cells["geometry"] = fov_cells["geometry"].translate(xoff = -x_offset, 
            yoff = -y_offset).affine_transform([scale, 0, 0, scale, 0, 0])
    fov_spots = spots.query("cell in @fov_cells.cell").copy()
    fov_spots["x"] = (fov_spots["global_x"]- x_offset) * scale
    fov_spots["y"] = (fov_spots["global_y"] - y_offset) * scale
    fov_spots = fov_spots.query("(30 < x < 2274) & (30 < y < 2254)")

    fig, ax = plt.subplots(figsize = (2.5,2.5),layout="constrained",dpi=300)

    clone_to_color = {i:discrete_cmap[4][i-1] for i in range(1,5)}
    clone_to_color[0] = "lightgrey"

    cells.plot(column = "clone",color = cells["clone"].map(clone_to_color),ax = ax)
    gpd.GeoDataFrame(geometry = [fov_box],crs = cells.crs).plot(ax = ax,facecolor = "none", edgecolor = "black",linewidth=2)

    xmin, ymin, xmax, ymax = cells.total_bounds  # Assumes cells is a GeoDataFrame
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.gca().axis('off')

    ax.add_artist(ScaleBar(
        dx=1,
        scale_formatter=lambda value, unit: f'{value} μm',
        location='lower left',
        color = "black",
        box_alpha = .8))
    
    # add legend with clone colors
    present_clones = cells["clone"].unique()
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, color=color, label=preedited_clone_names[clone])
                    for clone, color in clone_to_color.items() if clone in present_clones]
    height_ratio = (ymax - ymin)/(xmax - xmin)
    fig.legend(handles=legend_handles,loc = 'upper left', 
               bbox_to_anchor=(min(1/height_ratio + .1, 1), min(height_ratio + .15, 1)),ncol=1)
    
    save_plot(fig, plot_name + "_box", plots_path)

    img_path = img_paths[f"preedited_{experiment}"]
    fov_img, _ = load_fov(fov,rounds.query("series == 'H0M1'"),img_path["path"],img_path["file_pattern"],z_project = False)

    fig, ax = plt.subplots(figsize = (8,8),layout="constrained",dpi=300)

    z_projection = get_rgb(np.stack([
        unsharp_mask(fov_img[0].max(axis = 0),10),
        unsharp_mask(fov_img[1].max(axis = 0),10),   
        unsharp_mask(fov_img[4].max(axis = 0),100)],axis = 0),[99.5,99.5,98])
    ax.imshow(z_projection)
    fov_cells.plot(ax = ax,edgecolor = fov_cells["clone"].map(clone_to_color),facecolor = "none")
    for _, row in fov_spots.iterrows():
        plt.text(row['x'], row['y'], str(row['intID']).replace("intID",""), fontsize=7, color='white', ha='left', va='bottom')

    ax.add_artist(ScaleBar(
        dx=1/scale,
        scale_formatter=lambda value, unit: f'{value} μm',
        location='lower right',
        color = "white",
        box_alpha = 0))
    
    plt.xlim(0,2304)
    plt.ylim(0,2304)
    plt.gca().axis('off')

    save_plot(fig, plot_name + "_labeled_spots", plots_path)

def example_cell_with_spots(experiment,plot_name,fov,x,y,x_size,y_size):

    spots = pd.read_csv(results_path / f"{experiment}_filtered_spots.csv",keep_default_na=False)
    cells =gpd.read_file(results_path / f"{experiment}_cells_with_clone.json")
    cells.crs = None
    rounds = pd.read_csv(data_path / "imaging_rounds.csv",keep_default_na=False)
    spot_rounds = rounds.query("type.isin(['common','integration','edit'])").copy().reset_index()
    int_codebook = pd.read_csv(ref_path / "integration_codebook.csv")
    int_codebook["bits"] = int_codebook["bits"].apply(lambda x: ast.literal_eval(x))
    spots = spots.merge(int_codebook, on = "intID")

    x_offset = cells.query("fov == @fov")["x_offset"].values[0]
    y_offset = cells.query("fov == @fov")["y_offset"].values[0]
    scale = 1/cells.query("fov == @fov")["micron_per_pixel"].values[0]
    global_x = x/scale + x_offset
    global_y = y/scale + y_offset
    crop_box = box(global_x, global_y, global_x + x_size/scale, global_y + y_size/scale)

    crop_cells = cells[cells.geometry.intersects(crop_box)].copy()
    crop_cells.geometry = crop_cells.buffer(5).buffer(-5)
    rescaled_crop_cells = crop_cells.copy()
    rescaled_crop_cells["geometry"] = rescaled_crop_cells["geometry"].translate(xoff = -global_x, 
            yoff = -global_y).affine_transform([scale, 0, 0, scale, 0, 0])
    crop_spots = spots.query("cell in @crop_cells.cell").copy()
    crop_spots = spots.query("fov == @fov & @x < x < @x + @x_size & @y < y < @y + @y_size").copy()
    crop_spots["crop_x"] = crop_spots["x"] - x
    crop_spots["crop_y"] = crop_spots["y"] - y

    img_path = img_paths[f"preedited_{experiment}"]
    fov_drift = pd.read_csv(img_path["analysis"]+f"/{fov}_drift.csv")
    fov_img, _ = load_fov(fov,rounds.query("series == 'H0M1'"),img_path["path"],img_path["file_pattern"],z_project = False)
    crop_img = fov_img[:,:,y:y+y_size,x:x+x_size]

    # Box
    fig, ax = plt.subplots(figsize = (1.8,1.8),layout="constrained",dpi=300)

    fov_cells = cells.query("fov == @fov")
    fov_cells.plot(column = "clone",color = fov_cells["clone"].map(clone_to_color),ax = ax)
    gpd.GeoDataFrame(geometry = [crop_box],crs = cells.crs).plot(ax = ax,facecolor = "none", edgecolor = "black",linewidth=2)

    xmin, ymin, xmax, ymax = fov_cells.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.gca().axis('off')

    # add legend with clone colors
    present_clones = fov_cells["clone"].unique()
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, color=color, label=preedited_clone_names[clone])
                    for clone, color in clone_to_color.items() if clone in present_clones]
    fig.legend(handles=legend_handles,loc = 'upper left', bbox_to_anchor=(1, 1),ncol=1)

    save_plot(fig,"invitro_cell_box",plots_path)

    # Z projection
    fig, ax = plt.subplots(figsize = (2,2),layout="constrained",dpi=300)
    z_projection = get_rgb(np.stack([
        unsharp_mask(crop_img[0].max(axis = 0),10),
        unsharp_mask(crop_img[1].max(axis = 0),10),   
        unsharp_mask(crop_img[4].max(axis = 0),100)],axis = 0),[99.5,99.5,98])
    rescaled_crop_cells.plot(ax = ax,edgecolor = rescaled_crop_cells["clone"].map(clone_to_color),facecolor = "none",linewidth = 2)
    ax.imshow(z_projection)
    for _, row in crop_spots.iterrows():
        plt.text(row['crop_x'], row['crop_y'], str(row['intID']).replace("intID",""), fontsize=8, color='white', ha='left', va='bottom')
    ax.add_artist(ScaleBar(
        dx=1/scale,
        scale_formatter=lambda value, unit: f'{value} μm',
        location='lower right',
        color = "white",
        box_alpha = 0,
    ))
    plt.xlim(0,x_size)
    plt.ylim(0,y_size)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().tick_params(axis='both', which='both', length=0)  # Hides the tick marks
    plt.gca().xaxis.set_tick_params(labelbottom=False)  # Hides x-axis tick labels
    plt.gca().yaxis.set_tick_params(labelleft=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    save_plot(fig, plot_name + "_z_projection", plots_path)

    # Y projection
    fig, ax = plt.subplots(figsize = (2,2),layout="constrained",dpi=300)
    y_projection = get_rgb(np.stack([
        unsharp_mask(crop_img[0].max(axis = 2),10),
        unsharp_mask(crop_img[1].max(axis = 2),10),
        unsharp_mask(crop_img[4].max(axis = 2),100)],axis = 0),[99.7,99.7,98])
    plt.imshow(np.swapaxes(y_projection,0,1),aspect = .4)
    for _, row in crop_spots.iterrows():
        plt.text(row['z'],row['crop_y'], str(row['intID']).replace("intID",""), fontsize=10, color='white', ha='left', va='bottom')
    ax.add_artist(ScaleBar(
        dx=1/scale/4,
        scale_formatter=lambda value, unit: f'{value} μm',
        location='lower right',
        color = "white",
        box_alpha = 0,
    ))
    plt.gca().invert_yaxis()
    plt.xlabel("z")
    plt.gca().tick_params(axis='both', which='both', length=0)  # Hides the tick marks
    plt.gca().xaxis.set_tick_params(labelbottom=False)  # Hides x-axis tick labels
    plt.gca().yaxis.set_tick_params(labelleft=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    save_plot(fig, plot_name + "_y_projection", plots_path)

    # Create a blank canvas
    n_spots = len(crop_spots)
    n_rounds = len(spot_rounds)
    img_size = 20  # Size of each cropped image
    margin = 2     # Margin between images
    canvas_height = n_spots * (img_size + margin) - margin
    canvas_width = n_rounds * (img_size + margin) - margin
    canvas = np.ones((canvas_height, canvas_width,3))

    # Process and position each image
    i = 0
    site_to_color = {site:mcolors.to_rgb(colors[3+i]) for i,site in enumerate(site_names.keys())}
    for series, series_rounds in spot_rounds.groupby("series", sort=False):
        series_imgs, _ = load_fov(fov, series_rounds, img_path["path"],img_path["file_pattern"], z_project=False)
        j = 0
        for _, spot in crop_spots.iterrows():
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
                    img_section = add_colored_border(img_section,2,mcolors.to_rgb(colors[1]))
                elif bit in spot["bits"]:
                    img_section = add_colored_border(img_section,2,mcolors.to_rgb(colors[2]))
                elif round["type"] == "edit":
                    for site in list(site_names.keys()):
                        if round["site"] == site and spot[site] == round["edit"]:
                            img_section = add_colored_border(img_section,2,site_to_color[site])
                canvas[row_start:row_start + img_size, col_start:col_start + img_size,:] = img_section
            j += 1
        i += len(series_rounds)

    # Integration spots
    fig, ax = plt.subplots(figsize = (6,4),layout="constrained",dpi=300)

    integration_canvas = np.concatenate([canvas[:,:(img_size + margin)*23],np.ones((canvas_height, (img_size + margin)*5,3))],axis = 1)
    yticks = np.linspace(8, integration_canvas.shape[0]-12, num=len(crop_spots), dtype=int)
    plt.yticks(yticks, crop_spots.intID.str.replace("intID",""),size = 10)
    xlabels = ["Common","Common"] + [f"{i}" for i in range(1,22)] + [""] * 5
    xticks = np.linspace(10, integration_canvas.shape[1]-10, num=len(xlabels), dtype=int)
    plt.xticks(xticks, xlabels,size = 10,rotation = 90)
    plt.imshow(integration_canvas, cmap='gray')
    plt.tick_params(axis='both', which='both', length=0)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.ylabel("Decoded integration")

    # Add legend
    bit_type_colors = {"Common":colors[1],"Integration":colors[2]}
    for site, color in site_to_color.items():
        bit_type_colors[site_names[site]] = color
    legend_handles = [mpatches.Rectangle((0, 0), 1, 1, color=color, label=bit_type) for bit_type, color in bit_type_colors.items()]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.92, .68),ncol=1)

    save_plot(fig, plot_name +"_integration_spots",plots_path)
    plt.show()

    # Edit spots
    fig, ax = plt.subplots(figsize = (6,4),layout="constrained",dpi=300)

    edit_canvas = np.concatenate([canvas[:,(img_size + margin)*23:],np.ones((canvas_height, (img_size + margin)*1,3))],axis = 1)
    yticks = np.linspace(8, edit_canvas.shape[0]-12, num=len(crop_spots), dtype=int)
    plt.yticks(yticks, crop_spots.intID.str.replace("intID",""))
    xlabels = list(spot_rounds.query("type == 'edit'").apply(lambda x: edit_names[x.site][x.edit],axis = 1).values) + [""] * 1
    xticks = np.linspace(10, edit_canvas.shape[1]-10, num=len(xlabels), dtype=int)
    plt.xticks(xticks, xlabels,rotation = 90)
    plt.imshow(edit_canvas, cmap='gray')
    plt.tick_params(axis='both', which='both', length=0)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.ylabel("Decoded integration")

    save_plot(fig,plot_name +"_edit_spots",plots_path)
    plt.show()

# Generate plots
if __name__ == "__main__":
    for experiment in ["invitro","invivo"]:
        #edit_confusion(experiment,f"{experiment}_merfish_confusion")
        #integration_detection_violin(experiment,f"{experiment}_merfish_detection_violin")
        #integration_precision_recall(experiment,f"{experiment}_precision_recall_threshold")
        pass
    #example_fov("invitro","invitro_fov",31)
    #example_fov("invivo","invivo_fov",71)
    #example_cell_with_spots("invitro","invitro_cell",31,1345,1330,180,220)
