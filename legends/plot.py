import matplotlib.pyplot as plt
import sys
from pathlib import Path

plots_path = Path(__file__).parent / "plots"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))
plt.style.use(base_path / 'plot.mplstyle')

from src.utils import save_plot
from src.legends import edit_legend, add_cbar
from src.config import sequential_cmap

def plot_legend(handels, name):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([], [])
    fig.legend(handles=handels,loc='upper right',bbox_to_anchor=(.995,1),ncol=1)
    ax.axis('off')
    save_plot(fig, name, plots_path)

def plot_cbar(cmap, ticks, label, name, ticklabels=None):
    fig, cbar_ax = plt.subplots(figsize=(.2, 2))
    add_cbar(cbar_ax, cmap, ticks, label, ticklabels=ticklabels)
    save_plot(fig, name, plots_path)

if __name__ == "__main__":
    plot_legend(edit_legend, "edit_legend")
    plot_cbar(sequential_cmap, [.3,.4,.5,.6,.7], "Robinson-Foulds distance", "rf_cbar")
    plot_cbar(sequential_cmap, [0,-2,-4,-6], "Probe molecule fraction", "crosshyb_cbar", 
              ticklabels= ["1","$10^{-2}$","$10^{-4}$","$10^{-6}$"])

