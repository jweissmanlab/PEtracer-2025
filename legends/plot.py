import matplotlib.pyplot as plt
import sys
from pathlib import Path

plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

from petracer.utils import save_plot
from petracer.legends import edit_legend, edit_legend_with_other, add_cbar, barcode_legend, barcoding_clone_legend
from petracer.config import sequential_cmap

def plot_legend(handels, name, title= None):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([], [])
    fig.legend(handles=handels,loc='upper right',bbox_to_anchor=(.995,1),ncol=1,title=title)
    ax.axis('off')
    save_plot(fig, name, plots_path)

def plot_cbar(cmap, ticks, label, name, ticklabels=None):
    fig, cbar_ax = plt.subplots(figsize=(.2, 2))
    add_cbar(cbar_ax, cmap, ticks, label, ticklabels=ticklabels)
    save_plot(fig, name, plots_path)

if __name__ == "__main__":
    plot_legend(edit_legend, "edit_legend")
    plot_legend(edit_legend_with_other, "edit_legend_with_other")
    plot_cbar(sequential_cmap, [.3,.4,.5,.6,.7], "Robinson-Foulds distance", "rf_cbar")
    plot_cbar(sequential_cmap, [0,-1,-2,-3,-4,-5], "Probe molecule fraction", "crosshyb_frac_cbar", 
              ticklabels= ["1","$10^{-1}$","$10^{-2}$","$10^{-3}$","$10^{-4}$","$10^{-5}$"])
    plot_cbar(sequential_cmap.reversed(), [0,2,4,6,8,10], "$\Delta G$ - on-target $\Delta G$", "crosshyb_free_energy_cbar")
    plot_legend(barcode_legend, "barcode_legend")
    plot_legend(barcoding_clone_legend, "barcoding_clone_legend","Clone")

