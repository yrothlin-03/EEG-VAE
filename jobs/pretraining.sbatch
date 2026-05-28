#!/bin/bash
#SBATCH -p H100,A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=pretraining_eegvae
#SBATCH --output=logs/pretraining/%x_%j.out
#SBATCH --error=logs/pretraining/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG-VAE

python main_pretraining.py \
    --use_parsing \
    --num_epochs 20 \
    --max_step_train 3000 \
    --max_step_val 500 \
    --sequence_block mamba 
    