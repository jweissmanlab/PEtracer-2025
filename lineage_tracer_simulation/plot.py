import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configure paths
results_path = Path(__file__).parent / "results"
plots_path = Path(__file__).parent / "plots"
module_path = Path(__file__).parent.parent
sys.path.append(str(module_path))

# Load results
state_dist_results = pd.read_csv(
    results_path / "state_distribution_simulation.tsv",sep="\t")

# RF line plot for entropy vs. number of states
def rf_vs_state_distribution_lineplot():
    sns.lineplot(data=state_dist_results, x="states", y="rf", 
        hue="entropy")
    plot_name = "rf_vs_state_distribution_lineplot"
    plt.savefig(plots_path / f"{plot_name}.png")
    plt.savefig(plots_path / f"{plot_name}.svg")

# Generate plots
if __name__ == "__main__":
    rf_vs_state_distribution_lineplot()

