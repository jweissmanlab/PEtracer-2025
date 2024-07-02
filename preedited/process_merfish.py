"""Code for processing pre-edited validatation data."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import geopandas as gpd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# Configure paths
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

# Load source
from src.config import threads, log_path, min_spot_intensity, site_names

# Helper functions
def max_jaccard(x, groups, fill, min_jaccard = .5):
    x_set = set(x)
    best_group, best_jaccard = fill, min_jaccard
    for group, y in groups.items():
        y_set = set(y)
        jaccard = len(x_set.intersection(y_set)) / len(x_set.union(y_set))
        if jaccard > best_jaccard:
            best_group, best_jaccard = group, jaccard
    return best_group

def get_alleles(df,sites = ["EMX1","HEK3","RNF2"]):
    alleles = []
    for i, row in df.iterrows():
        for site in sites:
            alleles.append(row["intID"]+site+str(row[site]))
    return set(alleles)

# Assign imaging cells to clones
def assign_clones(spots, cells, clone_whitelist, fill = 1, min_jaccard = 0):
    clone_alleles = clone_whitelist.groupby("clone").apply(get_alleles).to_dict()
    cell_to_clone = spots.groupby("cell").apply(get_alleles).reset_index(name = "alleles")
    cell_to_clone["clone"] = cell_to_clone["alleles"].apply(max_jaccard, 
        groups = clone_alleles, fill = fill, min_jaccard = min_jaccard)
    cells = cells.merge(cell_to_clone[["cell","clone"]], on = "cell", how = "left")
    cells["clone"] = cells["clone"].fillna(fill).astype(int)
    return cells, cell_to_clone

def cv_decode_edits(spots,rounds):
    spots = spots.copy()
    intensities = np.array(spots["intensity"].to_list())
    for site in list(site_names.keys()):
        clf = LogisticRegression(max_iter = 1000,class_weight='balanced')
        site_intensities = intensities[:,rounds.site == site] + 1e-4
        site_intensities = site_intensities/site_intensities.sum(axis = 1,keepdims = True)
        spots[site] = cross_val_predict(clf,site_intensities,spots[site],cv = 10)
    return spots


# Process merfish data
if __name__ == "__main__":

    for experiment in ["invitro","invivo"]:
        # Load data
        spots = pd.read_csv(data_path / f"{experiment}_decoded_spots.csv",keep_default_na=False)
        spots["intensity"] = spots["intensity"].apply(ast.literal_eval)
        cells = gpd.read_file(data_path / f"{experiment}_cells.json")
        cells.crs = None
        clone_whitelist = pd.read_csv(results_path / "preedited_clone_whitelist.tsv",
                                    keep_default_na=False, sep = "\t")
        clone_whitelist["intID"] = clone_whitelist["mfID"]
        rounds = pd.read_csv(data_path / "imaging_rounds.csv",keep_default_na=False)
        rounds = rounds.query("type.isin(['common','integration','edit'])").reset_index(drop = True)

        # Assign clones
        if experiment == "invitro":
            cells, cell_to_clone = assign_clones(spots, cells, clone_whitelist, fill = 4, min_jaccard = 0)
        else:
            cells, cell_to_clone = assign_clones(spots, cells, clone_whitelist, fill = 0, min_jaccard = .2)


        # Filter spots
        spots = spots.merge(cell_to_clone[["cell","clone"]],on = "cell",how = "left")
        spots = spots.merge(clone_whitelist.rename(
            columns = {"EMX1":"EMX1_actual","RNF2":"RNF2_actual","HEK3":"HEK3_actual"}),
            on = ["intID","clone"],how = "left")
        filtered_spots = spots.sort_values("intBC_intensity",ascending = False).groupby(["cell","intID"]).first().reset_index()
        filtered_spots = filtered_spots.query("intBC_intensity > @min_spot_intensity")
        filtered_spots["whitelist"] = ~filtered_spots["EMX1_actual"].isna()

        # Decode edits
        if experiment == "invitro":
            filtered_spots = cv_decode_edits(filtered_spots,rounds)

        # Save
        cells.to_file(results_path / f"{experiment}_cells_with_clone.json", driver = "GeoJSON")
        filtered_spots.to_csv(results_path / f"{experiment}_filtered_spots.csv",index = False)
