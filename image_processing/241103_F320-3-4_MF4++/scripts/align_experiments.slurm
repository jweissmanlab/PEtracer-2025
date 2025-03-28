#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=align_experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=20
#SBATCH --account=weissman
#SBATCH --output ../log/%x_%j.out

REF_PATH=$(head -n 1 ../image_paths.txt | sed 's/\r//g' | tr -d '\n')
MOVING_PATH=$(tail -n 1 ../image_paths.txt | sed 's/\r//g' | tr -d '\n')
echo "Aligning ${MOVING_PATH} to ${REF_PATH}"

source activate fishtank

fishtank align-experiments \
    -r ${REF_PATH} \
    -m ${MOVING_PATH} \
    -o ../alignment.json \
    --ref_series "H0M1" \
    --moving_series "H0R1" \
    --color 405 \
    --z_offset -3 