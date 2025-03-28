#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=proseg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=20
#SBATCH --account=weissman
#SBATCH --output ../log/%x_%j.out

MERLIN_PATH="/lab/weissman_imaging/puzheng/MERFISH_analysis/20240812-F319-12-0807_MF4dna-mCh/"

source activate fishtank

fishtank assign-spots \
    -i "${MERLIN_PATH}/ExportBarcodes/barcodes.csv" \
    -p ../polygons.json \
    -o ../transcripts.csv \
    --max_dist 0 \
    --z_column global_z \
    --polygons_z_column z \
    --cell_fill -1 \
    