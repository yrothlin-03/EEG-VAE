#!/bin/bash
#SBATCH -p H100,A100,L40S,A40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --job-name=full_downstream_eegmamba
#SBATCH --output=logs/full_downstream/%x_%j.out
#SBATCH --error=logs/full_downstream/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

mkdir -p logs/full_downstream

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG-VAE

datasets=(BCI2A PHYSIONETMI FACED SEEDV)
seeds=(8 42 100 123 2024)
model=EEGMAMBA

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "======================================"
        echo "Dataset: $dataset | Seed: $seed"
        echo "======================================"

        python main_downstream.py \
            --use_parsing \
            --dataset_name "$dataset" \
            --seed "$seed" \
            --num_epochs 50 \
            --model_name "$model" \
            --mode linear_probing 
    done
done
