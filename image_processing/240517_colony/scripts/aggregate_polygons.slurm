#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=aggregate_polygons
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=20
#SBATCH --account=weissman
#SBATCH --output ../log/%x_%j.out

source activate fishtank

fishtank aggregate-polygons \
    -i ../cellpose_polygons/ \
    -o ../polygons.json \
    --min_size 100 \
    --z_column "global_z"