import re
from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.image as mimage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.stats as sm
from reportlab.graphics import renderPDF
from reportlab.lib.utils import ImageReader
from svglib.svglib import svg2rlg


def convert_svg_text(svg_path):
    """Convert SVG text to Arial font."""
    with open(svg_path, 'r') as f:
        svg_content = f.readlines()
    with open(svg_path, 'w') as file:
        for line in svg_content:
            if '<text' in line or '<tspan' in line:
                style_match = re.search(r'style="([^"]*)"', line)
                if style_match:
                    style_content = style_match.group(1)
                    font_size_match = re.search(r'\s*(\d+)px', style_content)
                    font_family_match = re.search(r'\'([^\']+)\'', style_content)
                    font_size = font_size_match.group(1) if font_size_match else '10'
                    font_family = font_family_match.group(1) if font_family_match else 'Arial'
                    font_family = font_family.replace('DejaVu Sans', 'Arial')
                    line = line.replace('style=', f'font-family="{font_family}" font-size="{font_size}" style=')
            file.write(line)


def save_plot(fig, plot_name, plots_path='.',transparent=False,svg = True,rasterize = False, dpi = 600):
    """Save the plot in both PNG and SVG formats."""
    plt.rcParams['svg.fonttype'] = 'none'
    plots_path = Path(plots_path)
    fig.savefig(plots_path / f"{plot_name}.png", bbox_inches='tight', pad_inches=0, transparent=transparent,dpi = dpi)
    if svg:
        if rasterize:
            for ax in fig.axes:
                for artist in ax.get_children():
                    if isinstance(artist, (mcoll.PathCollection, mcoll.PolyCollection, mcoll.QuadMesh,  # noqa: UP038
                                           mcoll.PatchCollection, mcoll.LineCollection, mpatches.Patch, mpatches.Rectangle,
                                           mimage.AxesImage)):
                        artist.set_rasterized(True)
        fig.savefig(plots_path / f"{plot_name}.svg", bbox_inches='tight', pad_inches=0, transparent=transparent,dpi = dpi)
        convert_svg_text(plots_path / f"{plot_name}.svg")


def render_plot(c, label, path, x, y, scale=1, x_offset=10, y_offset=25):
    """Render plot onto PDF canvas."""
    x = x * 72
    y = (11 - y) * 72
    if path is not None:
        file_extension = str(path).split('.')[-1].lower()
        if file_extension == 'svg':
            scale = scale * .8
            drawing = svg2rlg(str(path))
            if scale != 1:
                drawing.scale(scale, scale)
            renderPDF.draw(drawing, c, x + x_offset, y - drawing.height * scale - y_offset)
        elif file_extension == 'png':
            image = ImageReader(str(path))
            iw, ih = image.getSize()
            iw = (iw / 8.275) * scale
            ih = (ih / 8.275) * scale
            c.drawImage(image, x + x_offset, y - ih - y_offset, width=iw, height=ih, mask='auto')
    if label is not None:
        c.drawString(x + 10, y - 20, label)


def pearson_corr(y, x, correction_method='fdr_bh'):
    mask = y[y.notnull()].index
    x = x.loc[mask,:].copy()
    y = y[ mask]
    r_values = []
    p_values = []
    for col in x.columns:
        r, p = sp.stats.pearsonr(y, x.loc[:, col])
        if np.isnan(r):
            r = 0
            p = 1
        r_values.append(r)
        p_values.append(p)
    corrected_p_values = sm.multitest.multipletests(p_values, method=correction_method)[1]
    results_df = pd.DataFrame({
        'feature': x.columns,
        'cor': r_values,
        'p_value': p_values,
        'q_value': corrected_p_values,
    })
    results_df["rank"] = results_df["cor"].rank(ascending = True)
    return results_df