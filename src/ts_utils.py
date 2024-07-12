import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import NMF
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def sigma_threshold(x, max_sigma = 3, n_components = 2, min_threshold = 1,log = False):
    """Estimate threshold based on Gaussian Mixture Model."""
    # If more than 5000 values sample
    if len(x) > 5000:
        x = x.sample(5000)
    if log:
        x = np.log1p(x)
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(x.values.reshape(-1, 1)) 
    high_component = np.argmax(gmm.means_)
    threshold = gmm.means_[high_component,0] - max_sigma * np.sqrt(gmm.covariances_[high_component,0,0])
    if log:
        threshold = np.expm1(threshold)
    return max(min_threshold, threshold)

def bimodal_threshold(x, default_cutoff = 5, min_cutoff = 2, max_cutoff = 20):
    """Calculates the cutoff between two components of a bimodal distribution."""
    # If less than 100 values remove
    if len(x) < 100:
        return x.max()
    # If more than 1000 values sample
    if len(x) > 5000:
        x = x.sample(5000)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(x.values.reshape(-1, 1)) 
    # Return default cutoff if only one component is found
    if len(gmm.means_) < 2:
        return default_cutoff
    # If two components are found, return the cutoff between them
    y = gmm.predict([[0]])
    for cutoff in np.linspace(min_cutoff, max_cutoff, 100):
        if gmm.predict([[cutoff]]) != y:
            return cutoff
    return max_cutoff

def max_jaccard(x, groups, fill = -1, min_jaccard = 0):
    x_set = set(x)
    best_group, best_jaccard = fill, min_jaccard
    for group, y in groups.items():
        y_set = set(y)
        jaccard = len(x_set.intersection(y_set)) / len(x_set.union(y_set))
        if jaccard > best_jaccard:
            best_group, best_jaccard = group, jaccard
    return best_group

def pairwise_combs(clones):
    keys = list(clones.keys())
    combs = list(itertools.combinations(keys, 2))
    key_combs= [f"{i},{j}" for i, j in combs]
    list_combs = [sorted(set(list(clones[i]) + list(clones[j]))) for i, j in combs]
    return dict(zip(key_combs, list_combs))

def plot_whitelist_alleles(alleles, top_n = 100, plot_title = None):
    colored_ints = alleles.copy()
    if top_n is not None:
        top_intes = alleles.groupby("intID")["whitelist"].mean().sort_values(ascending = False).head(top_n).index
        colored_ints = colored_ints[colored_ints["intID"].isin(top_intes)]
    if colored_ints["cellBC"].nunique() > 500:
        use_cells = np.random.choice(colored_ints["cellBC"].unique(),500)
        colored_ints = colored_ints[colored_ints["cellBC"].isin(use_cells)]
    if "UMI" in colored_ints.columns:
        colored_ints["color"] = (colored_ints["whitelist"] * 2 - 1) * colored_ints["UMI"]
        colored_ints = pd.pivot_table(colored_ints,index=["clone","cellBC"], 
                                    columns="intID", values="color", aggfunc="sum").fillna(0)
    else:
        colored_ints["color"] = (colored_ints["whitelist"] * 2 - 1) * np.log10(colored_ints["intBC_intensity"])
        colored_ints = pd.pivot_table(colored_ints,index=["clone","cellBC"], 
                                    columns="intID", values="color", aggfunc="max").fillna(0)
    colored_ints.index = colored_ints.index.droplevel(1)
    cmap = mcolors.LinearSegmentedColormap.from_list('three_color_cmap', ["#1874CD", 'white', "#CD2626"])
    vmax = np.percentile(np.abs(colored_ints.values),98)
    sns.clustermap(colored_ints,figsize = (6,6),vmax = vmax,vmin = -vmax,cmap = cmap,cbar_pos = None,dendrogram_ratio=(0,0))
    plt.title(plot_title)

def call_clones(alleles,model = None,min_frac = .5,min_jaccard = .5,
                doublets = True,top_n = None,plot = True,plot_title = None):
    # Identify clones using clustering algorithm
    int_counts = pd.pivot_table(alleles.assign(present = 1),index=["cellBC"],columns="intID", values="present", aggfunc="first").fillna(0)
    int_counts[int_counts > 1] = 1
    if isinstance(model, NMF):
        clone = model.fit_transform(int_counts).argmax(axis = 1)
    else:
        clone = model.fit_predict(int_counts)
    cell_to_clone = pd.DataFrame({"cellBC":int_counts.index,"clone":clone.astype(str)})
    alleles = alleles.merge(cell_to_clone, on = "cellBC")
    # Determine set of integrations for each clone
    frac = alleles.groupby(["intID", "clone"]).size() / alleles.groupby("clone")["cellBC"].nunique()
    int_whitelist = frac.reset_index(name="frac").query("frac > @min_frac").drop(columns = "frac")
    clone_ints = int_whitelist.groupby("clone")["intID"].agg(list).to_dict()
    # Assign cells to clones using jaccard similarity
    if doublets:
        clone_ints.update(pairwise_combs(clone_ints))
    cell_to_clone = alleles.query("intID.isin(@int_whitelist.intID)").groupby(["cellBC"]).agg({"intID":list})
    cell_to_clone["clone"] = cell_to_clone["intID"].apply(lambda x: max_jaccard(x, clone_ints,min_jaccard=min_jaccard)).astype(str)
    cell_to_clone["whitelist"] = (~cell_to_clone["clone"].str.contains(",")) & (cell_to_clone["clone"] != "-1")
    cell_to_clone.drop(columns = "intID",inplace = True)
    # Whitelist alleles
    alleles = alleles.drop(columns = "clone").merge(cell_to_clone["clone"], on = "cellBC")
    int_whitelist["whitelist"] = True
    alleles = alleles.merge(int_whitelist, on = ["intID","clone"], how = "left")
    alleles["whitelist"] = alleles["whitelist"].fillna(0).astype(bool)
    # Plot
    if plot:
        plot_whitelist_alleles(alleles, top_n = top_n, plot_title = plot_title)
    return alleles, cell_to_clone.reset_index(drop = False)

def get_alleles(df,sites = ["EMX1","HEK3","RNF2"]):
    alleles = []
    for i, row in df.iterrows():
        for site in sites:
            alleles.append(row["intID"]+site+str(row[site]))
    return set(alleles)

def assign_clones(alleles,clone_whitelist,min_jaccard = .5,doublets = True,fill = -1,
                  top_n = 100,plot = True,plot_title = None):
    alleles = alleles.copy()
    # Assign cells to clones using jaccard similarity
    clone_alleles = clone_whitelist.groupby("clone").apply(get_alleles).to_dict()
    if doublets:
        clone_alleles.update(pairwise_combs(clone_alleles))
    cell_to_clone = alleles.groupby("cellBC").apply(get_alleles).reset_index(name = "alleles")
    cell_to_clone["clone"] = cell_to_clone["alleles"].apply(max_jaccard, 
        groups = clone_alleles, fill = fill, min_jaccard = min_jaccard).astype(str)
    cell_to_clone["whitelist"] = (~cell_to_clone["clone"].str.contains(",")) & (cell_to_clone["clone"] != "-1")
    # Whitelist alleles
    alleles = alleles.merge(cell_to_clone[["clone","cellBC"]], on = "cellBC")
    int_whitelist = clone_whitelist[["clone","intID"]].assign(whitelist = True).copy()
    alleles = alleles.merge(int_whitelist, on = ["intID","clone"], how = "left")
    alleles["whitelist"] = alleles["whitelist"].fillna(0).astype(bool)
    # Plot
    if plot:
        plot_whitelist_alleles(alleles, top_n = top_n, plot_title = plot_title)
    return alleles, cell_to_clone.reset_index(drop = False)

def select_allele(allele, sites=["RNF2", "HEK3", "EMX1"]):
    agg_funcs = {col: 'sum' if col in ["UMI", "readCount", "frac"] else 'first' for col in allele.columns}
    aggregated = allele.groupby("n_alleles").agg(agg_funcs)
    n_edits = 0
    for site in sites:
        values = list(allele[site].dropna().unique())
        if len(values) == 1:
            continue
        elif len(values) == 2 and 'None' in values:
            aggregated[site] = values[0] if values[1] == 'None' else values[1]
            n_edits += 1
        elif len(values) > 1:
            return allele
    if n_edits == 1:
        aggregated["n_alleles"] = 1
        return aggregated
    else:
        return allele