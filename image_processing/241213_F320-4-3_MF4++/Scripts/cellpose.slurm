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
#SBATCH --output ../log_/%x_%j.out
#SBATCH --array=0-93%2


IMG_PATH=$(head -n 1 ../image_paths.txt | sed 's/\r//g' | tr -d '\n')
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
        --diameter 17 \
        --channels DAPI \
        --filter deconwolf \
        --filter_args tilesize=900 \
        --downsample 4 \
        --min_size 1000 \
        --do_3D True \
        --gpu True