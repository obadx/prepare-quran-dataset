#!/bin/bash

#SBATCH --job-name=ContAnnotate
#SBATCH --account=shams035
#SBATCH --output=ds-splits.out
#SBATCH --time=0-10:00:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16


echo 'Starting '
bash ~/.bashrc
source ~/data/miniconda3/bin/activate
conda activate prepare-ds

cd ..
python segment_transcribe_hf.py --dataset-dir /cluster/users/shams035u1/data/quran-dataset --out-dataset-dir /cluster/users/shams035u1/data/cont-ds-out --device cpu --vllm-endpoint https://9e03-34-82-174-137.ngrok-free.app/v1

