#!/bin/bash

#SBATCH --job-name=LoadSplit
#SBATCH --account=shams035
#SBATCH --output=ds-splits.out
#SBATCH --time=3-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


echo 'Starting QuranDataset App #########'
bash ~/.bashrc
source ~/data/miniconda3/bin/activate
conda activate prepare-ds

cd ..
python load_all_splits.sh

