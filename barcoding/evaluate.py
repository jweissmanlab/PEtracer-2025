"""Code for evaluating barcoded trees."""
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
import treedata as td
import pycea
from pathlib import Path
from tqdm.auto import tqdm

# Configure paths
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

# Load source
from src.config import threads
from src.tree_utils import reconstruct_tree, get_edit_frac, collapse_mutationless_edges
from src.barcode_utils import get_barcode_clades

def eval_fmi(param):
    np.random.seed(param["iteration"])
    tdata = td.read_h5ad(data_path / f"barcoding_clone_{param['clone']}.h5td")
    # Sample characters
    n_characters = tdata.obsm["characters"].shape[1]
    if param["characters"] > n_characters:
        param["characters"] = n_characters
    else:
        use_characters = np.random.choice(tdata.obsm["characters"].columns,param["characters"],replace=False)
        tdata.obsm["characters"] = tdata.obsm["characters"].loc[:,use_characters]
    # dropout data
    detection_rate = np.mean(tdata.obsm["characters"] != -1)
    if param["detection_rate"] > detection_rate:
        param["detection_rate"] = detection_rate
    else:
        n_detected = int(tdata.obsm["characters"].size * param["detection_rate"])
        idx = np.where(tdata.obsm["characters"].values != -1)
        drop_idx = np.random.choice(len(idx[0]),len(idx[0]) - n_detected,replace=False)
        tdata.obsm["characters"].values[(idx[0][drop_idx],idx[1][drop_idx])] = -1
    # Permute barcodes
    if param["permute"]:
        tdata.obs["blast"] = np.random.permutation(tdata.obs["blast"])
        tdata.obs["puro"] = np.random.permutation(tdata.obs["puro"])
    # Reconstruct tree
    reconstruct_tree(tdata,solver = param["solver"],reconstruct_characters=True,estimate_lengths=False,collapse_edges=True)
    # Get barcode FMI
    puro_clades = get_barcode_clades(tdata,"puro")
    blast_clades = get_barcode_clades(tdata,"blast")
    puro_fmi = np.average(puro_clades["fmi"],weights=puro_clades["n"])
    blast_fmi = np.average(blast_clades["fmi"],weights=blast_clades["n"])
    fmi = np.average([puro_fmi,blast_fmi],weights=[puro_clades["n"].sum(),blast_clades["n"].sum()])
    # Format results
    result = pd.DataFrame({"puro_fmi":puro_fmi,"blast_fmi":blast_fmi,"fmi":fmi},index=[0])
    for key in param.keys():
        result[key] = param[key]
    return result

def clone_fmi(threads = 30):
    """Compare NJ and UPGMA trees for each clone."""
    # Define parameters
    params = {"solver":["upgma","nj"],
        "clone":range(1,7),
        "characters":[100],
        "detection_rate":[1],
        "permute":[True,False],
        "iteration":[0]} 
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_fmi, parallel_list), 
                                total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "clone_fmi.csv",index=False)

def fmi_vs_characters(threads = 30):
    """Test how the number of characters affects the FMI."""
    # Define parameters
    params = {"solver":["upgma","nj"],
        "clone":range(1,7),
        "characters":[5,10,15,20,25,30,35,40,45],
        "detection_rate":[1],
        "permute":[False],
        "iteration":range(10)} 
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_fmi, parallel_list), 
                                total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "fmi_vs_characters.csv",index=False)

def fmi_vs_detection(threads = 30):
    """Test how the number of characters affects the FMI."""
    # Define parameters
    params = {"solver":["upgma","nj"],
        "clone":range(1,7),
        "characters":[100],
        "detection_rate":[1,.9,.8,.7,.6,.5,.4,.3,.2,.1],
        "permute":[False],
        "iteration":range(10)} 
    params = pd.DataFrame(list(product(*params.values())), columns=params.keys())
    # Test parameters in parallel
    with mp.Pool(processes=threads) as pool:
        parallel_list = [row.to_dict() for index,row in params.iterrows()]
        results = list(tqdm(pool.imap_unordered(eval_fmi, parallel_list), 
                                total=len(parallel_list)))
        results = pd.concat(results)
    results.to_csv(results_path / "fmi_vs_detection_rate.csv",index=False)

def get_mean_barcode_time(tdata,barcode):
    times = []
    for node in tdata.obst["tree"].nodes:
        node_attrs = tdata.obst["tree"].nodes[node]
        if f"{barcode}_lca" in node_attrs:
            times.append(node_attrs["time"])
    return np.mean(times) * 16

def calculate_clone_stats():
    """Calculate the number of cells, edit fraction, and detection rate for each clone."""
    clone_stats = []
    for clone in range(1,7):
        tdata = td.read_h5ad(data_path / f"barcoding_clone_{clone}.h5td")
        pycea.pp.add_depth(tdata)
        leaf_depth = pycea.utils.get_keyed_leaf_data(tdata,"depth")
        site_edit_frac = get_edit_frac(tdata.obsm["characters"]) * 100
        detection_rate = (tdata.obsm["characters"] != -1).values.mean() * 100
        puro_mean_time = get_mean_barcode_time(tdata,"puro")
        blast_mean_time = get_mean_barcode_time(tdata,"blast")
        clone_stats.append({"clone":clone,"n_cells":tdata.n_obs,"site_edit_frac":site_edit_frac,
                            "avg_depth":np.mean(leaf_depth),
                            "site_edit_frac":site_edit_frac,"detection_rate":detection_rate,
                            "puro_mean_time":puro_mean_time,"blast_mean_time":blast_mean_time})
    # Add fmi scores
    fmi_scores = pd.read_csv(results_path / "clone_fmi.csv")
    fmi_score = fmi_scores.query("solver == 'nj' & ~permute")[["clone","puro_fmi","blast_fmi"]]
    clone_stats = pd.DataFrame(clone_stats).merge(fmi_score,on="clone")
    clone_stats.to_csv(results_path / "clone_stats.csv",index=False)

# Run simulations
if __name__ == "__main__":
    print("Evaluating barcoded trees")
    #clone_fmi(threads = threads)
    print("Testing how the number of characters affects the FMI")
    #fmi_vs_characters(threads = threads)
    print("Testing how the detection rate affects the FMI")
    #fmi_vs_detection(threads = threads)
    print("Calculating clone stats")
    calculate_clone_stats()
