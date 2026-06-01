#!/bin/bash
#SBATCH -p H100,A100,L40S,A40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=full_downstream_eegvae_mamba_mapped
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

datasets=(BCI2A PHYSIONETMI FACED TUAB SLEEPEDFX SEEDV)
seeds=(8 42 100)

# ── Models / versions to evaluate ─────────────────────────────────────────────
# A "target" is one label to benchmark. Two kinds:
#   • EEGVAE versions  → RUN_MODEL=EEGVAE, weights = a Phase 1/2 checkpoint.
#     The wrapper auto-detects ch / model_type / vq_n_quantizers / sequence_block.
#   • Baseline models  → RUN_MODEL=<NAME>, no weights here: they are pulled from
#     configs/model_configs.yaml (pretrained_checkpoint).
# Add/remove labels in `targets` to choose what runs. Each target maps to a
# model name (RUN_MODEL) and, for EEGVAE only, a checkpoint (RUN_WEIGHTS).
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
    # baselines: weights come from configs/model_configs.yaml, leave unset
)
# Downstream mode per target. EEGVAE trains a channel adaptor (channel_training);
# baselines have no adaptor, so linear_probing (frozen encoder + head) is the
# comparable protocol — switch a baseline to finetuning here if you prefer.
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

targets=(vq_mamba)   # e.g. add: kl_mamba vq_transformers cbramod eegmamba luna femba eegpt

OUT_ROOT=/projects/EEG-foundation-model/RECH202/EEG-VAE/checkpoints/full_downstream
LOG_ROOT=/home/infres/yrothlin-24/EEG-VAE/logs/full_downstream

for target in "${targets[@]}"; do
    model=${RUN_MODEL[$target]}
    weights=${RUN_WEIGHTS[$target]:-}        # empty for baselines
    mode=${RUN_MODE[$target]:-linear_probing}
    for dataset in "${datasets[@]}"; do
        # TUAB is evaluated on the mapped 19-ch layout; all other datasets use raw.
        if [[ "$dataset" == "TUAB" ]]; then
            channel_mode=mapped
        else
            channel_mode=mapped
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
            # EEGVAE picks its checkpoint here; baselines use the model zoo.
            if [[ -n "$weights" ]]; then
                args+=(--model_weights "$weights")
            fi

            python main_downstream.py "${args[@]}"
        done
    done
done
