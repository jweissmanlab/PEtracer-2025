#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=decode_spots
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24gb
#SBATCH --partition=20
#SBATCH --output ../log/%x_%j.out

COLOR_USAGE="/lab/weissman_imaging/4T1/reference/lineage_color_usage.csv"
STRATEGY="/lab/weissman_imaging/4T1/reference/tumor_decoding_strategy.csv"

source activate fishtank

echo "Decoding spots"
fishtank decode-spots \
    --input ../spots/ \
    --output ../decoded_spots_unfiltered.csv \
    --color_usage ${COLOR_USAGE} \
    --strategy ${STRATEGY} \
    --max_dist 2 \
    --filter_output True \

echo "Assigning spots to polygons"
fishtank assign-spots \
    -i "../decoded_spots_unfiltered.csv" \
    -p ../polygons.json \
    -o ../decoded_spots_unfiltered.csv \
    --max_dist 4 \
    --z_column global_z \
    --cell_fill 0 \
    --alignment ../alignment.json \
    --subset ../all_malignant_cells.txt 