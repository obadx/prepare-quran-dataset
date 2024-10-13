#!/bin/bash

#SBATCH --job-name=QuranDatasetApp
#SBATCH --account=shams035
#SBATCH --output=app.out
#SBATCH --time=3-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

echo 'Starting QuranDataset App #########'
bash ~/.bashrc
conda activate prepare-ds
cd ../frontend/
