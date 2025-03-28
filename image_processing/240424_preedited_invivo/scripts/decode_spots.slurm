#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=decode_spots
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24gb
#SBATCH --partition=20
#SBATCH --output ../log/%x_%j.out

STRATEGY="/lab/weissman_imaging/4T1/reference/lineage_decoding_strategy.csv"

source activate fishtank

echo "Aggregating polygons"
# fishtank aggregate-polygons \
#     -i ../cellpose_polygons/ \
#     -o ../polygons.json \
#     --min_size 500 \
#     --z_column "global_z" \
#     --save_union True \

echo "Decoding spots"
fishtank decode-spots \
    --input ../spots/ \
    --output ../decoded_spots.csv \
    --color_usage ../color_usage.csv \
    --strategy ${STRATEGY} \
    --max_dist 1.7 \

echo "Assigning spots to polygons"
fishtank assign-spots \
    -i "../decoded_spots.csv" \
    -p ../polygons.json \
    -o ../decoded_spots.csv \
    --max_dist 4 \
    --z_column global_z \
    --cell_fill 0 \