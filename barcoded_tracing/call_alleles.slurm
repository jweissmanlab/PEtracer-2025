#!/bin/bash
#SBATCH --job-name=call_alleles         # friendly name for job.
#SBATCH --nodes=1                       # ensure cpus are on one node
#SBATCH --ntasks=1                      # run three tasks in parallel
#SBATCH --cpus-per-task=8               # number of cpus/threads requested for each task.
#SBATCH --mem=32gb                     # memory requested.
#SBATCH --partition=20                  # partition (queue) to use
#SBATCH --output=call_alleles-%j.out    # name of output file.  %j is jobid
#SBATCH --array=1-3

# List of samples
samples=(
  "4T1_Pool1"
  "4T1_Pool2"
  "4T1_Pool3"
)

sample=${samples[($SLURM_ARRAY_TASK_ID-1)]}
echo "Processing sample: ${sample}"

# TS alleles
python ../scripts/alleles_from_bam.py \
  --bam "./bam/${sample}.bam" \
  --out "./data/"${sample}"/${sample}"_allele_counts.csv \
  --barcode_start 270 \
  --barcode_end 300 \
  --site_positions  "{'RNF2':332,'HEK3':380,'EMX1':448}" \
  --min_reads 2

# Puro barcodes
barcode="PuroBC"
python ../scripts/barcodes_from_bam.py \
    --bam "./bam/${sample}.bam" \
    --out "./data/${sample}/${sample}_${barcode}_counts.csv" \
    --barcode "${barcode}" \
    --barcode_start 22 \
    --barcode_end 32 \
    --min_reads 2

# Blast barcodes in the background
barcode="BlastBC"
python ../scripts/barcodes_from_bam.py \
    --bam "./bam/${sample}.bam" \
    --out "./data/${sample}/${sample}_${barcode}_counts.csv" \
    --barcode "${barcode}" \
    --barcode_start 23 \
    --barcode_end 33 \
    --min_reads 2