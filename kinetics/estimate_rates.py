import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import curve_fit

# Configure
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"

def exp_growth_saturation(x, a, b):
    return a * (1 - np.exp(-b * x))

def estimate_rates(output,cell_lines = ["4T1","B16F10"]):
    '''Estimate editing rates for each pegRNA'''
    cell_line_rates = []
    for cell_line in cell_lines:
        # Load data
        rates = []
        cells = pd.read_csv(data_path / f"{cell_line}_kinetics_cells.csv",index_col=0)
        # Get mean
        mean_edit_frac = cells.groupby(["day","peg_site","peg"]).agg({"edit_frac":"mean"}).reset_index()
        day0 = mean_edit_frac.groupby(["peg_site","peg"]).first().reset_index().assign(day=0,edit_frac=0)
        mean_edit_frac = pd.concat([mean_edit_frac,day0])
        # Fit curve
        for peg, peg_site in mean_edit_frac[["peg","peg_site"]].drop_duplicates().values:
            data = mean_edit_frac.query("peg == @peg & peg_site == @peg_site")
            if len(data) > 2:
                params, covariance = curve_fit(exp_growth_saturation, data["day"], data["edit_frac"], 
                                        p0=[1, 0.2],bounds=([.8,.0001],[1,0.6]))
                rates.append({"peg":peg,"peg_site":peg_site,f"{cell_line}_rate":params[1]})
        cell_line_rates.append(pd.DataFrame(rates))
    # Merge and save
    rates = cell_line_rates[0].merge(cell_line_rates[1],on=["peg","peg_site"])
    pegs = pd.read_csv(data_path / "pegRNAs.csv")
    rates["mean_rate"] = rates[["4T1_rate","B16F10_rate"]].mean(axis=1)
    rates = pegs.merge(rates,on=["peg","peg_site"],how = "right")
    rates.to_csv(results_path / output,index=False)

if __name__ == "__main__":
    estimate_rates("edit_rates.csv")