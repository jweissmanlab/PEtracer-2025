import numpy as np
import matplotlib.pyplot as plt
import shapely as shp
import fishtank as ft
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
import re

def plot_polygons(img = None,cells = None,extent = None,crop = None,color = "none",vmax = None, vmin = None,
               edgecolor = "none",palette = None,micron_per_pixel = 0.107,ax = None,**kwargs):
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
            color = cells[color].astype(dtype).map(palette).fillna("lightgray")
        cells.plot(color = color,edgecolor = edgecolor,ax = ax,**kwargs)
    ax.set_xlim(extent[[0,2]])
    ax.set_ylim(extent[[1,3]])
    ax.add_artist(ScaleBar(dx = 1,units="um",location='lower right',
                           color = "white" if img is not None else "black"
                           ,box_alpha=0 if img is not None else .8))