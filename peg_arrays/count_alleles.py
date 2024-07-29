"""Code to aggregate crispresso results into an allele count table."""

import sys
import pandas as pd
from pathlib import Path
import pandas as pd
import os

# Configure
results_path = Path(__file__).parent / "results"
data_path = Path(__file__).parent / "data"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

# Helper function to extract edit
def get_edit(seq, ref, pos = 22, window = 5):
    seq = seq[pos-window:pos+window]
    ref = ref[pos-window:pos+window]
    insertion = ""
    deletion = ""
    mutation = ""
    for s, r in zip(seq, ref):
        if r == '-':
            insertion += s
        elif s == '-':
            deletion += r
        elif s != r:
            mutation += s
    if deletion != "":
        deletion = "D" + deletion
    if mutation != "":
        mutation = "M" + mutation
    if insertion == "" and deletion == "" and mutation == "":
        return "None"
    else:
        return insertion + deletion + mutation
    
# Process peg array data
if __name__ == "__main__":
    # Load peg arrays
    peg_arrays = pd.read_csv(results_path / "peg_arrays.csv")
    # For each sample
    alleles = []
    for sample in os.listdir(data_path):
        [line,day,guides,array,version,rep] = sample.split("_")
        for site in ["HEK3","EMX1","RNF2"]:
            # Load crispresso output
            sample_alleles = pd.read_csv(data_path / sample / f"{site}_crispresso.tsv",sep = "\t").rename(
                columns = {"#Reads":"readCount"})
            # Call edits
            sample_alleles['edit'] = sample_alleles .apply(lambda row: get_edit(
                row['Aligned_Sequence'], row['Reference_Sequence']), axis=1)
            if site == "EMX1":
                sample_alleles["edit"] = sample_alleles["edit"].replace("CTTGGG","None")
            sample_alleles = sample_alleles.groupby(["edit"]).agg({"readCount":"sum"}).reset_index().sort_values(
                "readCount",ascending=False)
            sample_alleles = sample_alleles.assign(site=site,array=array,version=version)
            # Set unexpected edits to "Other"
            sample_alleles = sample_alleles.merge(peg_arrays,on=["site","edit","array","version"],how="left")
            sample_alleles.loc[(sample_alleles["edit"] != "None") & sample_alleles.position.isna(),"edit"] = "Other"
            sample_alleles = sample_alleles.groupby(["edit"]).agg({"readCount":"sum"}).reset_index()
            # Add metadata
            sample_alleles = sample_alleles.assign(site=site,line=line,guides=guides,array=array,
                                                   version=version,rep=rep,sample = sample)
            sample_alleles = sample_alleles.merge(peg_arrays,on=["site","edit","array","version"],how="left")
            alleles.append(sample_alleles)
    alleles = pd.concat(alleles)
    # Filter and save
    alleles = alleles.loc[(alleles["site"] == alleles["guides"]) | (alleles["array"] == "24mer")]
    alleles.to_csv(results_path / "peg_array_allele_counts.csv",index=False)