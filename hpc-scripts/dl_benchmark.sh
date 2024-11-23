#!/bin/bash

#SBATCH --job-name=DlBenchmark
#SBATCH --account=shams035
#SBATCH --output=dl_benchmark.out
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24


echo 'Starting Downloading #########'
bash ~/.bashrc
source ~/data/miniconda3/bin/activate
conda activate prepare-ds
cd ../

python tests/download_benchmark.py
echo 'End Benchmarking!'
