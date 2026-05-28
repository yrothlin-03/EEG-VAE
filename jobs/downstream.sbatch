#!/bin/bash
#SBATCH -p H100,A100,L40S,A40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=downstream_eegvae
#SBATCH --output=logs/downstream/%x_%j.out
#SBATCH --error=logs/downstream/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

mkdir -p logs/downstream

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG-VAE

python main_downstream.py \
    --use_parsing \
    --dataset_name BCI2A \
    --model EEG-VAE \
    --mode channel_training \

