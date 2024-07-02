#!/bin/bash
#SBATCH --job-name=cellranger          # friendly name for job.
#SBATCH --nodes=1                      # ensure cpus are on one node
#SBATCH --ntasks=1                     # run a single task
#SBATCH --cpus-per-task=32             # number of cpus/threads requested.
#SBATCH --mem=128gb                    # memory requested.
#SBATCH --partition=20                 # partition (queue) to use
#SBATCH --output=cellranger-%j.out     # name of output file.  %j is jobid
#SBATCH --array=1-14%4                 # array of jobs to submit

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

# Directory containing the fastq files
fastq_dir="./fastq"
fastq_dir=$(realpath "$fastq_dir")

# Path to the transcriptome reference
transcriptome="../reference/cellranger/mm10_PETS_v0"
transcriptome=$(realpath "$transcriptome")

# Path to the feature reference
feature_reference="../reference/cellranger/pegRNA_feature_reference.csv"
feature_reference=$(realpath "$feature_reference")

# Template for the libraries.csv content
libraries_template="fastqs,sample,library_type
{fastq_dir},{sample}_GEX,Gene Expression
{fastq_dir},{sample}_TS,Gene Expression
{fastq_dir},{sample}_CROP,CRISPR Guide Capture"

#Make output directories
mkdir -p ./bam
mkdir -p ./data

# Sample
sample=${samples[($SLURM_ARRAY_TASK_ID-1)]}
echo "Processing sample: ${sample}"

# Run cellranger count 
libraries_csv="${sample}_libraries.csv"
echo "$libraries_template" | sed "s/{sample}/$sample/g" | sed "s|{fastq_dir}|$fastq_dir|g" > "$libraries_csv"
cellranger count \
  --id="$sample" \
  --libraries="$libraries_csv" \
  --transcriptome="$transcriptome" \
  --feature-ref="$feature_reference" \
  --localmem=120 \
  --nosecondary

# Clean up
rm "$libraries_csv"
output_dir="./$sample/outs"
data_dir="./data/${sample}"
mkdir -p $data_dir
# Bam
bam_file="${output_dir}/possorted_genome_bam.bam"
mv "$bam_file" "./bam/${sample}.bam"
bai_file="${output_dir}/possorted_genome_bam.bam.bai"
mv "$bai_file" "./bam/${sample}.bai"
# Counts
filtered_h5="${output_dir}/filtered_feature_bc_matrix.h5"
mv "$filtered_h5" "${data_dir}/${sample}_filtered_counts.h5"
raw_h5="${output_dir}/raw_feature_bc_matrix.h5"
mv "$raw_h5" "${data_dir}/${sample}_raw_counts.h5"
# Summary
summary="${output_dir}/web_summary.html"
mv "$summary" "${data_dir}/${sample}_summary.html"
# CRISPR analysis
crispr="${output_dir}/crispr_analysis"
mv "$crispr" "${data_dir}/${sample}_crispr_analysis"
# Remove output directory
rm -r "./${sample}"