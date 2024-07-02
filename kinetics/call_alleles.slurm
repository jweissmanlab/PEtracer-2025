#!/bin/bash
#SBATCH --job-name=call_alleles         # friendly name for job.
#SBATCH --nodes=1                       # ensure cpus are on one node
#SBATCH --ntasks=1                      # run three tasks in parallel
#SBATCH --cpus-per-task=1               # number of cpus/threads requested for each task.
#SBATCH --mem=32gb                     # memory requested.
#SBATCH --partition=20                  # partition (queue) to use
#SBATCH --output=call_alleles-%j.out    # name of output file.  %j is jobid
#SBATCH --array=1-15%10

# List of samples
samples=(
  "4T1_D6_rep1"
  "4T1_D10_rep2"
  "4T1_D14_rep2"
  "4T1_D14_rep1"
  "4T1_D14_rep3"
  "4T1_D14_rep4"
  "4T1_D21_rep1"
  "4T1_D28_rep1"
  "B16F10_D10_rep1"
  "B16F10_D14_rep1"
  "B16F10_D21_rep1"
  "B16F10_D4_rep1"
  "B16F10_D6_rep1"
  "B16F10_D28_rep1"
  "B16F10_D28_rep2"
)

sample=${samples[($SLURM_ARRAY_TASK_ID-1)]}
echo "Processing sample: ${sample}"

python ../scripts/alleles_from_bam.py \
  --bam "./bam/${sample}.bam" \
  --out "./data/"${sample}"/${sample}"_allele_counts.csv \
  --barcode_start 21 \
  --barcode_end 31 \
  --extract_barcode True \
  --site_positions  "{'RNF2':61,'HEK3':109,'EMX1':177}" \
  --min_reads 2
