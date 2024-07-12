import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

import matplotlib.patches as mpatches
from .config import edit_ids, edit_names, site_names, discrete_cmap, edit_palette

edit_labels = [name for name in edit_names["EMX1"].values()] + ["Other"]
edit_legend = [mpatches.Rectangle((0, 0), 1, 1, color=color, label=label)
                    for color, label in zip(list(edit_palette.values())[1:10],edit_labels)]

edit_legend_with_other = [mpatches.Rectangle((0, 0), 1, 1, color=color, label=label)
                     for color, label in zip(list(edit_palette.values())[1:11],edit_labels)]

barcode_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Blast BC'),
                  plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Puro BC')]

barcoding_clone_legend = [mpatches.Rectangle((0, 0), 1, 1, color=color, label=label)
                    for color, label in zip(discrete_cmap[6], range(1,7))]

def add_cbar(cbar_ax, cmap, ticks, label, ticklabels=None):
    vmin = min(ticks)
    vmax = max(ticks)
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), orientation='vertical')
    cbar.outline.set_visible(False)
    cbar.set_label(label, labelpad=3)
    cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels)
    else:
        cbar.set_ticklabels([f'{x}' for x in ticks])
    cbar.ax.yaxis.set_tick_params(pad=2)