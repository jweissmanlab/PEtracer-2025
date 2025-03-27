import re
from pathlib import Path

import fishtank as ft
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pycea as py
import scipy as sp
import seaborn as sns
import shapely as shp
import geopandas as gpd
import pandas as pd

from matplotlib_scalebar.scalebar import ScaleBar
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
from svglib.svglib import svg2rlg
from .tree import hamming_distance
from .utils import save_plot


def plot_polygons(img = None,cells = None,extent = None,crop = None,color = "none",vmax = None, vmin = None,
               edgecolor = "none",palette = None,cmap = "viridis",micron_per_pixel = 0.107,ax = None,**kwargs):
    if img is None:
        extent = np.array(cells.total_bounds)
    else:
        extent = np.array(extent)
    if ax is None:
        fig, ax = plt.subplots(figsize = (3,3),dpi = 600)
    if crop is not None:
        crop = np.array(crop)
        crop_box = shp.geometry.box(*crop)
        if img is not None:
            crop_pixels = ((crop - extent[[0,1,0,1]]) / micron_per_pixel).astype(int)
            if img.ndim == 3:
                img = img[:,crop_pixels[1]:crop_pixels[3],crop_pixels[0]:crop_pixels[2]]
            else:
                img = img[crop_pixels[1]:crop_pixels[3],crop_pixels[0]:crop_pixels[2]]
        if cells is not None:
            cells = cells[cells.intersects(crop_box)]
        extent = crop
    if img is not None:
        ft.pl.imshow(img,extent = extent[[0,2,1,3]],origin = "lower",ax = ax,vmax = vmax,vmin = vmin)
    if palette is not None:
        dtype = type(list(palette.keys())[0])
    if cells is not None:
        if edgecolor not in ["none","black"]:
            edgecolor = cells[edgecolor].astype(dtype).map(palette).fillna("lightgray")
        if color != "none":
            cells["na_color"] = cells[color].isna()
            cells = cells.sort_values("na_color",ascending = False)
            if palette is not None:
                color = cells[color].astype(dtype).map(palette).fillna("lightgray")
            else:
                vmin = cells[color].min() if vmin is None else vmin
                vmax = cells[color].max() if vmax is None else vmax
                norm = plt.Normalize(vmin = vmin, vmax = vmax)
                cmap = plt.get_cmap(cmap)
                color = cells[color].map(lambda x: "lightgray" if pd.isnull(x) else cmap(norm(x)))
        cells.plot(color = color,edgecolor = edgecolor,ax = ax,**kwargs)
        cells.drop(columns = "na_color",errors = "ignore",inplace = True)
    ax.set_xlim(extent[[0,2]])
    ax.set_ylim(extent[[1,3]])
    ax.add_artist(ScaleBar(dx = 1,units="um",location='lower right',
                           color = "white" if img is not None else "black"
                           ,box_alpha=0 if img is not None else .8))


def distance_comparison_scatter(plot_name,plots_path, clone_tdata, x = "tree", y = "spatial", mm = False, 
    groupby = "sample",sample_n = 10000,figsize = (1.8,1.7)):
    """Scatter plot comparing pairwise distances using two distance metrics."""
    metric_names = {"tree_distances": "Phylo. distance (days)",
                    "spatial_distances": "Spatial distance (um)",
                    "character_distances": "Character distance"}
    if mm:
        metric_names["spatial_distances"] = "Spatial distance (mm)"
    # Get distances
    clone_tdata = clone_tdata[clone_tdata.obs.tree.notnull()].copy()
    sample_n = min(clone_tdata.n_obs**2,sample_n)
    connect_key = None
    if x  == "spatial" or y == "spatial":
        py.tl.distance(clone_tdata,key = "spatial",metric = "euclidean",sample_n = sample_n,update=False)
        connect_key = "spatial"
    if x == "character" or y == "character":
        if connect_key is None:
            connect_key = "character"
            py.tl.distance(clone_tdata,key = "characters",metric = hamming_distance,key_added="character",sample_n = sample_n,update=False)
        else:
            py.tl.distance(clone_tdata,key = "characters",metric = hamming_distance,key_added="character",connect_key = connect_key,update=False)
    if x == "tree" or y == "tree":
        py.tl.tree_distance(clone_tdata,depth_key="time",connect_key=connect_key, tree = "tree", update = False)
    distances = py.tl.compare_distance(clone_tdata,dist_keys = [x,y],groupby = groupby)
    if mm and (x == "spatial" or y == "spatial"):
        distances["spatial_distances"] = distances["spatial_distances"]/1000
    if x == "tree" or y == "tree":
        distances["tree_distances"] = distances["tree_distances"]/2
    x = x + "_distances"
    y = y + "_distances"
    # Plot
    distances = distances.query("obs1 != obs2").copy()
    distances["density"] = sp.stats.gaussian_kde(distances[[x,y]].T)(distances[[x,y]].T)
    fig, ax = plt.subplots(figsize=figsize,dpi = 600, layout = "constrained")
    sns.scatterplot(data = distances,x = x,y = y,hue = "density",
                    palette = "viridis",alpha = .5,s = 10,legend = False)
    if distances[y].max() < 10:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', min_n_ticks=4))
    plt.ylabel(metric_names[y])
    plt.xlabel(metric_names[x])
    save_plot(fig,plot_name,plots_path,rasterize=True)
