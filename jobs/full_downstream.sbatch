#!/bin/bash
#SBATCH -p H100,A100,L40S,A40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=vq_transformers_wo_bp
#SBATCH --output=/home/infres/yrothlin-24/EEG-VAE/logs/full_downstream/%x_%j.out
#SBATCH --error=/home/infres/yrothlin-24/EEG-VAE/logs/full_downstream/%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

mkdir -p logs/full_downstream

eval "$(micromamba shell hook --shell=bash)"
micromamba activate rech202

cd /home/infres/yrothlin-24/EEG-VAE

datasets=(BCI2A PHYSIONETMI FACED)
seeds=(8 42 100 123 2024 3025)

CKPT_ROOT=/projects/EEG-foundation-model/RECH202/EEG-VAE/checkpoints/pretraining
JEPA_ROOT=/projects/EEG-foundation-model/RECH202/EEG-VAE/checkpoints/jepa/mamba/q8

declare -A RUN_MODEL=(
    [kl_mamba]=EEGVAE
    [vq_mamba]=EEGVAE
    [vq_transformers]=EEGVAE
    [jepa]=EEGVAE
    [cbramod]=CBRAMOD
    [eegmamba]=EEGMAMBA
    [luna]=LUNA
    [femba]=FEMBA
    [eegpt]=EEGPT
)
declare -A RUN_WEIGHTS=(
    [kl_mamba]=$CKPT_ROOT/kl/mamba/pretraining_final.pt
    [vq_mamba]=$CKPT_ROOT/vq/mamba/q8/pretraining_final.pt
    [vq_transformers]=$CKPT_ROOT/vq/transformers/q8/pretraining_final.pt
    [jepa]=$JEPA_ROOT/jepa_final.pt
)

declare -A RUN_MODE=(
    [kl_mamba]=channel_training
    [vq_mamba]=channel_training
    [vq_transformers]=channel_training
    [jepa]=channel_training
    [cbramod]=linear_probing
    [eegmamba]=linear_probing
    [luna]=linear_probing
    [femba]=linear_probing
    [eegpt]=linear_probing
)

targets=(jepa)

OUT_ROOT=/projects/EEG-foundation-model/RECH202/EEG-VAE/checkpoints/full_downstream
LOG_ROOT=/home/infres/yrothlin-24/EEG-VAE/logs/full_downstream

for target in "${targets[@]}"; do
    model=${RUN_MODEL[$target]}
    weights=${RUN_WEIGHTS[$target]:-}
    mode=${RUN_MODE[$target]:-linear_probing}
    for dataset in "${datasets[@]}"; do
        if [[ "$dataset" == "TUAB" ]]; then
            channel_mode=mapped
        else
            channel_mode=raw
        fi
        for seed in "${seeds[@]}"; do
            echo "======================================"
            echo "Target: $target ($model, $mode) | Dataset: $dataset ($channel_mode) | Seed: $seed"
            echo "======================================"

            args=(
                --use_parsing
                --dataset_name "$dataset"
                --seed "$seed"
                --num_epochs 50
                --model_name "$model"
                --mode "$mode"
                --channel_mode "$channel_mode"
                --loss_dir "$LOG_ROOT/$target/$dataset/seed$seed"
                --model_checkpoint_dir "$OUT_ROOT/$target/$dataset/seed$seed"
            )
            if [[ -n "$weights" ]]; then
                args+=(--model_weights "$weights")
            fi

            python main_downstream.py "${args[@]}"
        done
    done
done
