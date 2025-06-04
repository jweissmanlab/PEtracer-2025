import matplotlib.pyplot as plt
import sys
from pathlib import Path
import matplotlib as mpl

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

def plot_cbar(cmap, ticks, label, name, ticklabels=None,center=None):
    fig, cbar_ax = plt.subplots(figsize=(.2, 2))
    add_cbar(cbar_ax, cmap, ticks, label, ticklabels=ticklabels,center=center)
    save_plot(fig, name, plots_path)

if __name__ == "__main__":
    #plot_legend(edit_legend, "edit_legend")
    #plot_legend(edit_legend_with_other, "edit_legend_with_other")
    #plot_cbar(sequential_cmap, [.3,.4,.5,.6,.7], "Robinson-Foulds distance", "rf_cbar")
    #plot_cbar(sequential_cmap, [0,-1,-2,-3,-4,-5], "Probe molecule fraction", "crosshyb_frac_cbar", 
    #          ticklabels= ["1","$10^{-1}$","$10^{-2}$","$10^{-3}$","$10^{-4}$","$10^{-5}$"])
    #plot_cbar(sequential_cmap.reversed(), [0,2,4,6,8,10], "$\Delta G$ - on-target $\Delta G$", "crosshyb_free_energy_cbar")
    #plot_legend(barcode_legend, "barcode_legend")
    #plot_legend(barcoding_clone_legend, "barcoding_clone_legend","Clone")
    #plot_cbar(mpl.colormaps["viridis"], [0,.2,.4,.6,.8,1], "Density", "density_cbar")
    #plot_cbar(sequential_cmap, [0,.2,.4,.6,.8,1], "Normalized UMIs", "reads_cbar")
    #plot_cbar(mpl.colormaps["magma"], [0,1,2,3,4,5,6], "Fitness", "fitness_cbar")
    #plot_cbar(mpl.colormaps["Greens_r"], [0,200,400,600,800], "Boundary distance", "boundary_dist_cbar")
    #plot_cbar(mpl.colormaps["viridis"], [.2,.4,.6,.8,1.0], "Detection rate", "detection_rate_cbar")
    #plot_cbar(mpl.colormaps["Blues"], [0,1500,3000], "Arg1_Macrophage_density", "arg1_density_cbar")
    #plot_cbar(mpl.colormaps["Reds"], [0,3,6], "Cldn4_expression", "Cldn4_cbar")
    #plot_cbar(mpl.colormaps["Reds"], [0,50,100], "Cldn4 expression", "expression_cbar")
    #plot_cbar(mpl.colormaps["RdBu_r"], [-1.5,0,1.5], "Mean expression (z-score)", "mean_expression_cbar")
    #plot_cbar(mpl.colormaps["RdBu_r"], [-2,0,1.5], "Fitness correlation (Pearson)", "fitness_corr_cbar",center = 0)
    plot_cbar(mpl.colormaps["Purples"], [0,3,6], "Fitness", "purple_fitness_cbar")


