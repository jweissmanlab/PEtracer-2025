#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=detect_spots
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --partition=20
#SBATCH --output ../log/%x_%j.out
#SBATCH --array=0-262%50

IMG_PATH=$(head -n 1 ../image_path.txt | sed 's/\r//g' | tr -d '\n')

source activate fishtank

fishtank detect-spots \
    --fov ${SLURM_ARRAY_TASK_ID} \
    --input ${IMG_PATH} \
    --output ../spots/ \
    --color_usage ../color_usage.csv \
    --common_bits r52,r53 \
    --reg_bit beads \
    --ref_series H0M1 \
    --filter unsharp_mask \
    --filter_args sigma=10 \
    --spot_min_sigma 2 \
    --spot_max_sigma 10 \
    --spot_threshold 200 \
    --spot_radius 5