"""Code for modeling cross hybridization between 5mer readout probes."""

import sys
import numpy as np
import pandas as pd
import nupack
from itertools import product
import random
from pathlib import Path

# Configure paths
results_path = Path(__file__).parent / "results"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

from src.config import site_names

#Define constants
site_seqs = {"HEK3":"CTTGCCAAGT{insert}CGTGCTCACTG",
            "EMX1":"GTGATGGGAG{insert}TTCTTCTGGAG",
             "RNF2":"GCTACCTGTC{insert}GTAATGACAGA"}

unedited_seqs = {"HEK3":"TGGGAGCCCAAGTTCTTCTG",
                 "EMX1":"TCTATGGGAGTTCTTCTGAGT",
                 "RNF2":"AACACCTGTCGTAATGACTA"}

def reverse_complement(sequence):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    complement_sequence = ''.join(complement[base] for base in sequence)
    reverse_complement_sequence = complement_sequence[::-1]
    return reverse_complement_sequence

def simulate_crosshyb(inserts, seq, unedited_seq, celsius = 43, sodium = 0.3):
    # Setup simulaiton
    model = nupack.Model(material='dna', celsius=celsius, sodium=sodium)
    seqs = []
    for insert in inserts:
        seq_with_insert = seq.format(insert=insert)
        if "None" in seq_with_insert:
            seq_with_insert = unedited_seq
        seqs.append(trim_to_length(seq_with_insert, 20))
    probes = [nupack.Strand(reverse_complement(seq),name=f'probe:{insert}') for seq, insert in zip(seqs,inserts)]
    targets = [nupack.Strand(seq,name=f'target:{insert}') for seq, insert in zip(seqs,inserts)]
    conc = {s:1e-8 for s in probes}
    conc.update({s:1e-10 for s in targets})
    t1 = nupack.Tube(strands=conc, complexes=nupack.SetSpec(max_size=2), name='Tube t1')
    tube_result = nupack.tube_analysis(tubes=[t1], compute=['pairs'], model=model)
    # Get results
    results = pd.DataFrame(list(product(inserts,inserts)),columns=["probe","target"])
    results["complex"] = results.apply(lambda x: f'(probe:{x.probe}+target:{x.target})',axis=1)
    concs = {c.name : conc for c, conc in tube_result.tubes[t1].complex_concentrations.items()}
    energies = {c.name : info.free_energy for c, info in tube_result.complexes.items()}
    results["concentration"] = results["complex"].map(concs) / 1e-10
    results["free_energy"] = results["complex"].map(energies)
    results["probe_frac"] = results.groupby("target")["concentration"].transform(lambda x: x / x.sum())
    return results

def random_inserts(N, length):
    possible_sequences = [''.join(seq) for seq in product('ACGT', repeat=length)]
    if N > len(possible_sequences):
        N = len(possible_sequences)
    sequences = random.sample(possible_sequences, N)
    return sequences

def trim_to_length(seq, length):
    if len(seq) <= length:
        return seq
    excess_length = len(seq) - length
    remove_from_start = excess_length // 2
    remove_from_end = excess_length - remove_from_start
    trimmed_seq = seq[remove_from_start:len(seq)-remove_from_end]
    return trimmed_seq

def crosshyb_vs_length():
    results = []
    for length in range(1,9):
        for site in site_names.keys():
            for i in range(10):
                np.random.seed(i)
                crosshyb = simulate_crosshyb(random_inserts(8,length), site_seqs[site], unedited_seqs[site])
                crosshyb["correct"] = crosshyb["probe"] == crosshyb["target"]
                free_energy_diff = crosshyb.groupby("correct").aggregate({"free_energy":"mean"}).diff().iloc[1,0]
                correct_frac = crosshyb.query("correct").probe_frac.mean()
                results.append({"length":length,
                                "site":site,
                                "iteration":i,
                                "free_energy_diff":free_energy_diff,
                                "correct_frac":correct_frac})
    results = pd.DataFrame(results)
    results.to_csv(results_path / "crosshyb_vs_length.csv",index=False)

def top_insert_crosshyb():
    inserts = pd.read_csv(results_path / "top_inserts.tsv",sep="\t")
    inserts = inserts[inserts["within_10%"]].copy()
    results = []
    for site in site_names.keys():
        crosshyb = simulate_crosshyb(inserts.query("site == @site")["insert"].tolist() + ["None"], 
                                    site_seqs[site], unedited_seqs[site])
        crosshyb["site"] = site
        results.append(crosshyb) 
    results = pd.concat(results)
    results.to_csv(results_path / "top_insert_crosshyb.csv",index=False)

# Get cross hybridization matrix for each site
if __name__ == "__main__":
    print("Simulating cross hybridization for different insert lengths")
    crosshyb_vs_length()
    print("Simulating cross hybridization for top inserts")
    top_insert_crosshyb()
