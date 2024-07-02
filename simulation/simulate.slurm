#!/bin/bash
#SBATCH --job-name=simulate
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128gb
#SBATCH --partition=20
#SBATCH --output simulate-%j.out

python ./simulate.py