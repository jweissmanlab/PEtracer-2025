from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

threads = 30
log_path = None
default_colors = ["black","blue","red","green","orange"]

# Configure colormaps
cmap_colors = [(1, 1, 1)] + [plt.cm.Reds(i / (256 - 1)) for i in range(256)]
sequential_cmap = mcolors.LinearSegmentedColormap.from_list("Reds", cmap_colors, N=256)
