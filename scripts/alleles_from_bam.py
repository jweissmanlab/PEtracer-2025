import numpy as np
import sys
import pandas as pd
import argparse
import pysam
import multiprocessing as mp
from tqdm.auto import tqdm
from collections import defaultdict
import ast
from pathlib import Path

# Configure
__file__ = "/lab/solexa_weissman/PEtracing_shared/PETracer_Paper/kinetics/dev.ipynb"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

# Load helper functions
from petracer.seq import insertion_from_alignment, barcode_from_alignment

def call_alleles(param):
    # Setup
    intID, args, lock = param
    sites = ast.literal_eval(args.site_positions)
    if len(sites) > 0:
        end = max(sites.values())
    else:
        end = args.barcode_position
    # Get iterator
    bamfile = pysam.AlignmentFile(args.bam, "rb")
    if args.extract_barcode:
        total_reads = bamfile.count(contig=intID)
        read_iter = tqdm(bamfile.fetch(intID), total=total_reads, mininterval=60, desc="TS")
    else:
        read_iter = bamfile.fetch(intID)
    # Process reads
    umi_counts = defaultdict(int)
    for read in read_iter:
        if (read.mapping_quality < 30 or 
            read.reference_start > args.barcode_start or
            read.reference_end < end or 
            'N' in read.cigarstring or
            not read.has_tag('CB') or 
            not read.has_tag('UB')):
            continue
        # Get integration
        if args.extract_barcode:
            intID = barcode_from_alignment(read.query_sequence, read.cigarstring, args.barcode_start, args.barcode_end, read.reference_start)
        else:
            intID = read.reference_name
        # Get allele
        alleles = []
        for name, pos in sites.items():
            allele = insertion_from_alignment(read.query_sequence, read.cigarstring, pos, read.reference_start)
            if name == "EMX1" and allele == "CTTGGG":
                allele = "None"
            alleles.append(allele)
        key = (read.get_tag('UB'),read.get_tag('CB'), intID, *alleles)
        umi_counts[key] += 1
    bamfile.close()
    # Correct and aggregate UMIs
    if len(umi_counts) > 0:
        site_names = list(sites.keys())
        allele_counts = pd.DataFrame(umi_counts.keys(), columns=["UMI","cellBC","intID"] + site_names)
        allele_counts["readCount"] = umi_counts.values()
        del umi_counts
        # correct UMIs
        allele_counts = allele_counts.groupby(["intID","cellBC","UMI"] + site_names).agg(
            {"readCount":"sum"}).sort_values("readCount", ascending=False).reset_index()
        agg_dict = {site: 'first' for site in site_names} 
        agg_dict["readCount"] = "sum"
        allele_counts = allele_counts.groupby(["intID","cellBC","UMI"]).agg(agg_dict).reset_index()
        # collapse UMIs
        allele_counts = allele_counts.groupby(["intID","cellBC"] + site_names).agg(
        {"UMI":"size","readCount":"sum"}).reset_index()
        # filter alleles
        allele_counts = allele_counts.query(f"readCount >= {args.min_reads}")
        with lock:
            allele_counts.to_csv(args.out, mode='a', header=False, index=False)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Call alleles from bam file")
    # Add arguments
    parser.add_argument("--bam", type=str,help="Bam file")
    parser.add_argument("--out", type=str, help="Output csv file")
    parser.add_argument("--barcode_start", type=int, help="Start of integration barcode in reference")
    parser.add_argument("--barcode_end", type=int, help="End of integration barcode in reference")
    parser.add_argument("--site_positions", type=str, help="Dictionary mapping site names to positions")
    parser.add_argument("--min_reads", type=int, help="Minimum number of reads to call allele")
    parser.add_argument("--extract_barcode", type = bool, help="Extract barcode sequence from reads instead of using alignment")
     # Parse the arguments
    args = parser.parse_args()
    sites = ast.literal_eval(args.site_positions)
    bamfile = pysam.AlignmentFile(args.bam, "rb")
    intIDs = [ref for ref in bamfile.references if "intID" in ref]
    bamfile.close()
    # Make output file
    lock = mp.Manager().Lock()
    pd.DataFrame(columns=["intID", "cellBC"] + list(sites.keys()) + ["UMI", "readCount"]).to_csv(args.out, index=False)
    # Process
    if args.extract_barcode:
        call_alleles((intIDs[0],args,lock))
    # Process in parallel
    else:    
        with mp.Pool(processes=8) as pool:
            _ = list(tqdm(pool.imap_unordered(call_alleles,[(intID,args,lock) for intID in intIDs]), 
                                        total=len(intIDs),mininterval=60, desc="TS"))

if __name__ == "__main__":
    main()