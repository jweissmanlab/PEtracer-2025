#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=proseg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --partition=20
#SBATCH --account=weissman
#SBATCH --output ../log/%x_%j.out

MERLIN_PATH="/lab/weissman_imaging/puzheng/MERFISH_analysis/20241103-F320-3-4-1016-MF9-4plusplus/"
CODEBOOK=$(ls ${MERLIN_PATH}*codebook.csv)

source activate fishtank

# echo "Aggregating polygons"
# fishtank aggregate-polygons \
#     -i ../cellpose_polygons/ \
#     -o ../polygons.json \
#     --min_size 100 \
#     --z_column "global_z" \
#     --save_union True \

# echo "Assigning transcripts to polygons"
# fishtank assign-spots \
#     -i "${MERLIN_PATH}/ExportBarcodes/barcodes.csv" \
#     -p ../polygons.json \
#     -o ../transcripts.csv \
#     --max_dist 0 \
#     --z_column global_z \
#     --cell_fill 0 \
#     --map_z True \

# mkdir -p proseg
# cd proseg

# echo "Running ProSeg"
# proseg \
#     -x global_x \
#     -y global_y \
#     -z global_z \
#     --gene-column barcode_id \
#     --cell-id-column cell \
#     --cell-id-unassigned 0 \
#     --detect-layers \
#     --voxel-layers 3 \
#     --ncomponents 10 \
#     --nbglayers 10 \
#     --enforce-connectivity \
#     --max-transcript-nucleus-distance 10 \
#     --nuclear-reassignment-prob .2 \
#     ../transcripts.csv \

# cd ..

echo ${CODEBOOK}

fishtank assign-proseg \
    -t ../transcripts.csv \
    -p ../proseg/transcript-metadata.csv.gz \
    -o ../proseg_counts.csv \
    --codebook ${CODEBOOK} \
    --barcode_column barcode_id \
    --cell_missing 0 \
    --min_jaccard 0.4 \