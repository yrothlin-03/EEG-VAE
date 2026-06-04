#!/bin/bash
#SBATCH -p H100,A100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=jepa_eegvae
#SBATCH --output=/home/infres/yrothlin-24/EEG-VAE/logs/jepa/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/EEG-VAE/logs/jepa/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG-VAE

mkdir -p logs/jepa

CKPT_ROOT=/projects/EEG-foundation-model/RECH202/EEG-VAE/checkpoints/pretraining
VAE_CHECKPOINT=$CKPT_ROOT/vq/mamba/q8/pretraining_final.pt

python main_jepa.py \
    --use_parsing \
    --num_epochs 20 \
    --max_step_train 5000 \
    --max_step_val 500 \
    --vae_checkpoint "$VAE_CHECKPOINT" 
