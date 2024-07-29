#!/bin/bash
#SBATCH --job-name=crispresso          # friendly name for job.
#SBATCH --nodes=1                      # ensure cpus are on one node
#SBATCH --ntasks=1                     # run a single task
#SBATCH --cpus-per-task=1             # number of cpus/threads requested.
#SBATCH --mem=8gb                    # memory requested.
#SBATCH --partition=20                 # partition (queue) to use
#SBATCH --output=crispresso-%j.out     # name of output file.  %j is jobid
#SBATCH --array=1-40%40                 # array of jobs to submit

# Guides
guides=("GGCCCAGACTGAGCACGACT" "GTCATCTTAGTCATTACGAC" "GAGTCCGAGCAGAAGAACTC")
guide_names=("HEK3" "RNF2" "EMX1")

# Amplicon
amplicon="GAATCCAGCTAGCTGTGCAGCNNNNNNNNNNTTAATTAATGGCAGTCATCTTAGTCATTACGACAGGTGTTCGTTGTAACTCATATACTTGGGGCCCAGACTGAGCACGACTTGGCAGAGGAAAGGAAGCCCTGCTTCCTCCAGAGGGCGTCGCAGGCCTGAGTCCGAGCAGAAGAACTCCCATCACATCAACCGGTGGATCGTTAACTGCTCACCTATTAGCGGCTAAGG"

# Fastqs
fastq_list=($(ls ./test/*.fastq.gz))

# Run crispresso
fastq=${fastq_list[($SLURM_ARRAY_TASK_ID-1)]}
sample=$(basename "$fastq" .fastq.gz)
echo "Processing: ${sample}"
CRISPResso --fastq_r1 "$fastq" \
  -g $(IFS=,; echo "${guides[*]}") \
  -gn $(IFS=,; echo "${guide_names[*]}") \
  -a "$amplicon" \
  -n "$sample" --suppress_plots 

# Clean up
output_dir="./CRISPResso_on_${sample}"
data_dir="./data/${sample}"
mkdir -p $data_dir
for name in "${guide_names[@]}"; do
  mv "${output_dir}/Alleles_frequency_table_around_${name}.txt" "${data_dir}/${name}_crispresso.tsv"
done
rm -rf "$output_dir"
rm "CRISPResso_on_${sample}.html"
