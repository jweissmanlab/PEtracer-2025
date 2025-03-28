#!/bin/bash
# Configuration values for SLURM job submission.
#SBATCH --job-name=cellpose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia-t4-20
#SBATCH --account=weissman
#SBATCH --output ../log/%x_%j.out
#SBATCH --array=0-262%1


IMG_PATH=$(head -n 1 ../image_path.txt | sed 's/\r//g' | tr -d '\n')
CORRECTIONS="/lab/weissman_imaging/puzheng/Corrections/20240401-Merscope01_s11_n1200"

source activate fishtank

fishtank cellpose \
        --fov $SLURM_ARRAY_TASK_ID \
        --input $IMG_PATH \
        --output ../cellpose_polygons \
        --file_pattern {series}/Conv_zscan_{fov}.dax \
        --color_usage ../color_usage.csv \
        --corrections $CORRECTIONS \
        --model nuclei \
        --diameter 25 \
        --channels DAPI \
        --downsample 4 \
        --min_size 30000 \
        --do_3D True \
        --gpu True