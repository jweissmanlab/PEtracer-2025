#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=align_experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=weissman
#SBATCH --account=weissman
#SBATCH --output ../log/%x_%j.out

RNA_PATH=$(head -n 1 ../image_paths.txt | sed 's/\r//g' | tr -d '\n')
LINEAGE_PATH=$(tail -n 1 ../image_paths.txt | sed 's/\r//g' | tr -d '\n')
echo "Aligning ${LINEAGE_PATH} to ${RNA_PATH}"

source activate fishtank

fishtank align-experiments \
    -r ${RNA_PATH} \
    -m ${LINEAGE_PATH} \
    -o ../alignment.json \
    --ref_series "H0M1" \
    --moving_series "H0R1" \
    --color 405 \
    --z_offset -3 \
    --rotation -0.9902 \