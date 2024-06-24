"""Code for modeling cross hybridization between 5mer readout probes."""

import sys
import numpy as np
import pandas as pd
import nupack
from pathlib import Path

# Configure paths
results_path = Path(__file__).parent / "results"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    complement_sequence = ''.join(complement[base] for base in sequence)
    reverse_complement_sequence = complement_sequence[::-1]
    return reverse_complement_sequence

def simulate_crosshyb(v):
    """Given a list of sequences, simulate the cross-hybridization matrix using NUPACK"""
    model = nupack.Model(material='dna', celsius=43, sodium=0.3)
    nvars = len(v)
    strands = [nupack.Strand(s,name='s'+str(i)) for i,s in enumerate(v)]
    strands_rc = [nupack.Strand(reverse_complement(s),name='s*'+str(i)) for i,s in enumerate(v)]
    strands_conc = {s:1e-8 for s in strands+strands_rc}
    t1 = nupack.Tube(strands=strands_conc, complexes=nupack.SetSpec(max_size=2), name='Tube t1')
    tube_result = nupack.tube_analysis(tubes=[t1], compute=['pairs'], model=model)
    concs = {c.name : conc for c, conc in tube_result.tubes[t1].complex_concentrations.items()}
    concs_pairwise = []
    for s1 in range(nvars):
        for s2 in range(nvars):
            curr_name = '(s' + str(s1) +'+s*' + str(s2)+')'
            curr_name_rev = '(s' + str(s2) +'+s*' + str(s1)+')'
            curr_name_rev_swap = '(s' + str(s1) +'+s*' + str(s2)+')'
            curr_name_swap = '(s*' + str(s1) +'+s' + str(s2)+')'
            if curr_name in concs:
                concs_pairwise.append(concs[curr_name]/1e-8)
            elif curr_name_rev in concs:
                concs_pairwise.append(concs[curr_name_rev]/1e-8)
            elif curr_name_rev_swap in concs:
                concs_pairwise.append(concs[curr_name_rev_swap]/1e-8)
            elif curr_name_swap in concs:
                concs_pairwise.append(concs[curr_name_swap]/1e-8)
    return np.array(concs_pairwise).reshape((nvars,nvars))

# Get cross hybridization matrix for each site
if __name__ == "__main__":
    inserts = pd.read_csv(results_path / "top_inserts.tsv",sep="\t")
    site_seqs = {"HEK3":"GCCAAGT{insert}CGTGCTCA",
                 "EMX1":"ATGGGAG{insert}TTCTTCTG",
                 "RNF2":"ACCTGTC{insert}GTAATGAC"}
    site_crosshyb = {}
    for site in site_seqs.keys():
        site_inserts = inserts.loc[inserts["site"]==site,"insert"].values
        insert_seqs = [site_seqs[site].format(insert=insert) for insert in site_inserts]
        crosshyb = simulate_crosshyb(insert_seqs)
        crosshyb = pd.DataFrame(crosshyb,index=site_inserts,columns=site_inserts)
        crosshyb.to_csv(results_path / f"{site}_crosshyb.tsv",sep="\t")