#!/bin/bash
#SBATCH --job-name=call_alleles         # friendly name for job.
#SBATCH --nodes=1                       # ensure cpus are on one node
#SBATCH --ntasks=1                      # run three tasks in parallel
#SBATCH --cpus-per-task=8              # number of cpus/threads requested for each task.
#SBATCH --mem=128gb                     # memory requested.
#SBATCH --partition=20                  # partition (queue) to use
#SBATCH --output=call_alleles-%j.out    # name of output file.  %j is jobid
#SBATCH --array=1-1

# List of samples
samples=(
  "4T1_preedited"
)

sample=${samples[($SLURM_ARRAY_TASK_ID-1)]}
echo "Processing sample: ${sample}"

# TS alleles
python ../scripts/alleles_from_bam.py \
  --bam "./bam/${sample}.bam" \
  --out "./data/"${sample}"/${sample}"_allele_counts.csv \
  --barcode_start 272 \
  --barcode_end 302 \
  --site_positions  "{'RNF2':332,'HEK3':380,'EMX1':448}" \
  --min_reads 2

# Barcode mapping
python ../scripts/barcode_mapping_from_bam.py \
    --bam "./bam/${sample}.bam" \
    --out "./data/"${sample}"/${sample}"_mapping_counts.csv \
    --barcode_position 60 \
    --other_barcode "intBC" \
    --other_barcode_start 272 \
    --other_barcode_end 302 \
    --min_reads 4
