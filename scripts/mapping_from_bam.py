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
from petracer.seq import barcode_from_alignment

def call_mapping(param):
    """Gets mapping between barcodes for a given intID"""
    # Setup
    intID, args, lock = param
    args.other_barcode_aligned = ast.literal_eval(args.other_barcode_aligned)
    # Process reads
    bamfile = pysam.AlignmentFile(args.bam, "rb", threads=8)
    umi_counts = defaultdict(int)
    for read in bamfile.fetch(intID):
        if (read.mapping_quality < 30 or 
            read.reference_start > args.barcode_position or
            'N' in read.cigarstring or
            not read.has_tag('CB') or 
            not read.has_tag('UB')):
            continue
        if args.other_barcode_aligned and read.reference_end < args.other_barcode_end:
            continue
        elif read.reference_end + 5 < args.other_barcode_start:
            continue
        # Get integration
        barcode = barcode_from_alignment(read.query_sequence, read.cigarstring,
                                         args.other_barcode_start, args.other_barcode_end, read.reference_start)
        key = (read.get_tag('UB'),read.get_tag('CB'), read.reference_name, barcode)
        umi_counts[key] += 1
    bamfile.close()
    # Correct and aggregate UMIs
    if len(umi_counts) > 0:
        barcode_counts = pd.DataFrame(umi_counts.keys(), columns=["UMI","cellBC","intID",args.other_barcode])
        barcode_counts["readCount"] = umi_counts.values()
        del umi_counts
        # correct UMIs
        barcode_counts = barcode_counts.groupby(["cellBC","UMI","intID",args.other_barcode]).agg(
            {"readCount":"sum"}).sort_values("readCount", ascending=False).reset_index()
        barcode_counts = barcode_counts.groupby(["cellBC","UMI","intID"]).agg({args.other_barcode:"first","readCount":"sum"}).reset_index()
        # collapse UMIs
        barcode_counts = barcode_counts.groupby(["cellBC","intID",args.other_barcode]).agg(
        {"UMI":"size","readCount":"sum"}).reset_index()
        # filter barcodes
        barcode_counts = barcode_counts.query(f"readCount >= {args.min_reads}")
        # Save the results
        with lock:
            barcode_counts.to_csv(args.out, mode='a', header=False, index=False)

def main():
    """Gets mapping between barcodes given a bam file"""
    # Create the parser
    parser = argparse.ArgumentParser(description="Call alleles from bam file")
    # Add arguments
    parser = argparse.ArgumentParser(description="Call alleles from bam file")
    parser.add_argument("--bam", type=str,help="Bam file")
    parser.add_argument("--out", type=str, help="Output file")
    parser.add_argument("--barcode_position", type=int, help="Position of barcode in reference")
    parser.add_argument("--other_barcode", type=str, help="Name of other barcode")
    parser.add_argument("--other_barcode_start", type=int, help="Start of other barcode in reference")
    parser.add_argument("--other_barcode_end", type=int, help="End of other barcode in reference")
    parser.add_argument("--other_barcode_aligned", type=str, default="False")
    parser.add_argument("--min_reads", type=int, help="Minimum number of reads to call allele")
     # Parse the arguments
    args = parser.parse_args()
    bamfile = pysam.AlignmentFile(args.bam, "rb")
    intIDs = [ref for ref in bamfile.references if "intID" in ref]
    bamfile.close()
    # Process in parallel
    lock = mp.Manager().Lock()
    pd.DataFrame(columns=["cellBC","intID", args.other_barcode, "UMI", "readCount"]).to_csv(args.out, index=False)
    # Process
    with mp.Pool(processes=8) as pool:
        _ = list(tqdm(pool.imap_unordered(call_mapping,[(intID,args,lock) for intID in intIDs]), 
                                    total=len(intIDs),mininterval=60, desc="T3"))

if __name__ == "__main__":
    main()