#!/bin/bash
#SBATCH -p H100,A100,L40S,A40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=bci2a_loso
#SBATCH --output=logs/bci2a_loso/%x_%j.out
#SBATCH --error=logs/bci2a_loso/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

mkdir -p logs/bci2a_loso

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG-VAE

subjects=(A01 A02 A03 A04 A05 A06 A07 A08 A09)

for test_subject in "${subjects[@]}"; do
    train_subjects=()

    for subject in "${subjects[@]}"; do
        if [[ "$subject" != "$test_subject" ]]; then
            train_subjects+=("$subject")
        fi
    done

    echo "======================================"
    echo "Test subject: $test_subject"
    echo "Train subjects: ${train_subjects[*]}"
    echo "======================================"

    python main_downstream.py \
        --use_parsing \
        --dataset_name BCI2A \
        --num_epochs 50 \
        --model_name EEG-VAE \
        --mode linear_probing \
        --train_ids "${train_subjects[@]}" \
        --test_ids "$test_subject" \
        --early_stopping_patience null \
        --restore_best_checkpoint False 
done