#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=detect_spots
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --partition=20
#SBATCH --output ../log/%x_%j.out
#SBATCH --array=0-59%30

LINEAGE_PATH=$(tail -n 1 ../image_paths.txt | sed 's/\r//g' | tr -d '\n')
COLOR_USAGE="/lab/weissman_imaging/4T1/reference/lineage_color_usage.csv"

source activate fishtank

fishtank detect-spots \
    --fov  ${SLURM_ARRAY_TASK_ID} \
    --input ${LINEAGE_PATH} \
    --output ../spots/ \
    --color_usage ${COLOR_USAGE} \
    --common_bits r52,r53 \
    --reg_bit beads \
    --ref_series H0R1 \
    --filter unsharp_mask \
    --filter_args sigma=10 \
    --spot_min_sigma 2 \
    --spot_max_sigma 10 \
    --spot_threshold 200 \
    --spot_radius 5