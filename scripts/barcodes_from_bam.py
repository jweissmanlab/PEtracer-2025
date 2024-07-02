import numpy as np
import sys
import pandas as pd
import argparse
import pysam
from tqdm.auto import tqdm
from collections import defaultdict
from pathlib import Path

# Configure
__file__ = "/lab/solexa_weissman/PEtracing_shared/PETracer_Paper/kinetics/dev.ipynb"
base_path = Path(__file__).parent.parent
sys.path.append(str(base_path))

# Load helper functions
from src.seq_utils import barcode_from_alignment

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Call alleles from bam file")

    # Add arguments
    parser.add_argument("--bam", type=str,help="Bam file")
    parser.add_argument("--out", type=str, help="Output file")
    parser.add_argument("--barcode", type=str, help="Barcode name")
    parser.add_argument("--barcode_start", type=int, help="Start position on reference")
    parser.add_argument("--barcode_end", type=int, help="End position on reference")
    parser.add_argument("--min_reads", type=int, help="Minimum number of reads to call allele")

    # Parse the arguments
    args = parser.parse_args()

    # Process reads
    bamfile = pysam.AlignmentFile(args.bam, "rb", threads=8)
    total_reads = bamfile.count(contig=args.barcode)
    umi_counts = defaultdict(int)
    for read in tqdm(bamfile.fetch(args.barcode), total=total_reads, mininterval=60, desc=args.barcode): 
        if (read.mapping_quality < 30 or 
            read.reference_start > args.barcode_start or
            read.reference_end < args.barcode_end or 
            'N' in read.cigarstring or
            not read.has_tag('CB') or 
            not read.has_tag('UB')):
            continue
        barcode = barcode_from_alignment(read.query_sequence, read.cigarstring, 
                                         args.barcode_start, args.barcode_end, read.reference_start)
        key = (read.get_tag('UB'),read.get_tag('CB'), barcode)
        umi_counts[key] += 1
    bamfile.close()
    barcode_counts = pd.DataFrame(umi_counts.keys(), columns=["UMI","cellBC",args.barcode])
    barcode_counts["readCount"] = umi_counts.values()
    del umi_counts
    # correct UMIs
    barcode_counts = barcode_counts.groupby(["cellBC","UMI",args.barcode]).agg(
        {"readCount":"sum"}).sort_values("readCount", ascending=False).reset_index()
    barcode_counts = barcode_counts.groupby(["cellBC","UMI"]).agg({args.barcode:"first","readCount":"sum"}).reset_index()
    # collapse UMIs
    barcode_counts = barcode_counts.groupby(["cellBC",args.barcode]).agg(
    {"UMI":"size","readCount":"sum"}).reset_index()
    # filter barcodes
    barcode_counts = barcode_counts.query(f"readCount >= {args.min_reads}")
    # Save the results
    barcode_counts.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()