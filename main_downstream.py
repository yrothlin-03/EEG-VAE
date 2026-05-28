from pathlib import Path
import random

import numpy as np
import torch
import yaml
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader

from utils import downstream_Trainer, downstream_parser
from utils.downstream_wrapper import DownstreamEEGVAE
from models import Model as FoundationModel
from eeg_preprocessing.loaders import CustomDataset, build_loaders


def get_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def update_config(base_config, parsed_args):
    parsed_dict = vars(parsed_args)

    dataset_name = parsed_dict.get("dataset_name") or base_config.get("dataset_name")
    if parsed_dict.get("dataset_name") is not None:
        base_config["dataset_name"] = parsed_dict["dataset_name"]

    dataset_keys = {
        "dataset_path",
        "n_channels",
        "n_classes",
        "patch_size",
        "train_ids",
        "val_ids",
        "test_ids",
    }
    data_keys = {
        "batch_size",
        "split_ratio",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "seed",
        "channel_mode",
        "return_ch_names",
    }
    training_keys = {
        "task",
        "model_name",
        "model_config_path",
        "mode",
        "model_weights",
        "strict_checkpoint_loading",
        "loss_dir",
        "model_checkpoint_dir",
        "tsne_enabled",
        "tsne_dir",
        "tsne_splits",
        "tsne_max_samples",
        "tsne_perplexity",
        "tsne_pca_enabled",
        "tsne_pca_components",
        "tsne_pca_whiten",
        "tsne_learning_rate",
        "tsne_max_iter",
        "tsne_random_state",
        "tsne_dpi",
        "tsne_figsize",
        "tsne_feature_source",
        "num_epochs",
        "device",
        "optimizer",
        "scheduler",
        "lr",
        "max_lr",
        "onecycle_pct_start",
        "onecycle_div_factor",
        "onecycle_final_div_factor",
        "weight_decay",
        "betas",
        "grad_clip",
        "use_amp",
        "log_step",
        "max_step_train",
        "max_step_val",
        "max_step_test",
        "loss",
        "label_smoothing",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "restore_best_checkpoint",
        "metric_monitoring",
        "plateau_factor",
        "plateau_patience",
        "output_dim",
    }
    head_mapping = {
        "head_type": "type",
        "pooling": "pooling",
        "head_n_layer": "n_layer",
        "hidden_dim": "hidden_dim",
        "head_dropout": "dropout",
        "head_norm": "norm",
        "probe_max_norm": "probe_max_norm",
        "classifier_max_norm": "classifier_max_norm",
    }
    model_keys = {
        "in_channels",
        "out_channels",
        "adapted_channels",
        "adaptor_layers",
        "z_channels",
        "embed_dim",
        "ch",
        "ch_mult",
        "num_res_blocks",
        "attn_resolutions",
        "resolution",
        "dropout",
        "resamp_with_conv",
        "tanh_out",
        "use_checkpoint",
        "sequence_block",
        "criss_cross_patch_size",
    }

    nullable_overrides = {"early_stopping_patience"}

    for key, value in parsed_dict.items():
        if key == "use_parsing" or (value is None and key not in nullable_overrides):
            continue

        if key in dataset_keys:
            if dataset_name is None:
                raise KeyError("Missing dataset_name for dataset-specific CLI override.")
            base_config.setdefault(dataset_name, {})[key] = value
        elif key in data_keys:
            base_config.setdefault("data", {})[key] = value
        elif key in training_keys:
            base_config.setdefault("training", {})[key] = value
        elif key in head_mapping:
            base_config.setdefault("head", {})[head_mapping[key]] = value
        elif key in model_keys:
            base_config.setdefault("model", {})[key] = value

    return base_config



def checkpoint_subdir_for_sequence_block(model_config: dict) -> str:
    sequence_block = str(model_config.get("sequence_block", "attention")).lower()
    if sequence_block == "attention":
        return "transformers"
    if sequence_block == "mamba":
        return "mamba"
    raise ValueError(
        f"Unknown model.sequence_block={sequence_block!r}. "
        "Expected 'attention' or 'mamba'."
    )


def add_eegvae_checkpoint_type_dir(training_config: dict, model_name: str, model_config: dict) -> dict:
    training_config = dict(training_config)
    if model_name != "EEGVAE":
        return training_config

    checkpoint_root = Path(training_config.get("model_checkpoint_dir", "checkpoints/downstream"))
    checkpoint_type = checkpoint_subdir_for_sequence_block(model_config)
    if checkpoint_root.name not in {"mamba", "transformers"}:
        checkpoint_root = checkpoint_root / checkpoint_type
    training_config["model_checkpoint_dir"] = str(checkpoint_root)
    training_config["checkpoint_type"] = checkpoint_type
    return training_config

def build_model(
    model_name: str,
    model_config: dict,
    dataset_config: dict,
    training_config: dict,
    dataset_name: str,
    data_config: dict,
    head_config: dict | None = None,
):
    if "n_classes" not in dataset_config:
        raise KeyError(
            "Missing `n_classes` in the selected downstream dataset config."
        )

    mode = str(training_config.get("mode", "linear_probing")).lower()
    checkpoint_path = None if mode in {"full_training", "full-training", "from_scratch"} else training_config.get("model_weights")

    if model_name != "EEGVAE":
        return FoundationModel(
            model_name=model_name,
            dataset_name=dataset_name,
            data_mode=data_config.get("channel_mode", "mapped"),
            out_path=None,
            checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
            model_config=model_config,
            head_config=head_config or {},
            task=training_config.get("task", "classification"),
            n_classes=dataset_config["n_classes"],
            mode=mode,
            patch_size=dataset_config.get("patch_size"),
        )

    return DownstreamEEGVAE(
        model_config=model_config,
        head_config=head_config,
        n_classes=dataset_config["n_classes"],
        task=training_config.get("task", "classification"),
        output_dim=training_config.get("output_dim", 1),
        checkpoint_path=checkpoint_path,
        mode=mode,
        strict=training_config.get("strict_checkpoint_loading", False),
    )


def _normalize_model_name(model_name: str) -> str:
    aliases = {
        "EEG-VAE": "EEGVAE",
        "EEG_VAE": "EEGVAE",
        "CBRAMODD": "CBRAMOD",
    }
    model_name = str(model_name).upper()
    return aliases.get(model_name, model_name)


def _select_model_config(base_model_config: dict, training_config: dict):
    model_name = _normalize_model_name(training_config.get("model_name", "EEGVAE"))

    if model_name == "EEGVAE":
        return model_name, dict(base_model_config), training_config.get("model_weights")

    model_config_path = Path(
        training_config.get("model_config_path", "configs/model_configs.yaml")
    )
    model_zoo_config = get_config(model_config_path)

    if model_name not in model_zoo_config:
        raise KeyError(
            f"Unknown model_name={model_name}. "
            f"Available models: EEGVAE, {', '.join(model_zoo_config.keys())}"
        )

    selected = model_zoo_config[model_name] or {}
    model_config = dict(selected.get("params") or {})
    checkpoint_path = (
        selected.get("pretrained_checkpoint")
        or selected.get("pretrained_checkpoints")
        or training_config.get("model_weights")
    )

    training_config["model_weights"] = checkpoint_path
    return model_name, model_config, checkpoint_path


def _is_initialized(param):
    return not isinstance(param, UninitializedParameter)


def count_parameters(module):
    return sum(
        param.numel()
        for param in module.parameters()
        if _is_initialized(param)
    )


def count_uninitialized_parameters(module):
    return sum(
        1
        for param in module.parameters()
        if not _is_initialized(param)
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_model_size(model):
    total_params = count_parameters(model)
    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad and _is_initialized(param)
    )
    frozen_params = total_params - trainable_params
    uninitialized_params = count_uninitialized_parameters(model)

    print("Model size before training:")
    print(f"  Initialized params  : {total_params:,}")
    print(f"  Trainable params    : {trainable_params:,}")
    print(f"  Frozen params       : {frozen_params:,}")
    if uninitialized_params:
        print(f"  Lazy params pending : {uninitialized_params}")
    if hasattr(model, "backbone"):
        print(f"  Backbone params     : {count_parameters(model.backbone):,}")
    if hasattr(model, "classifier"):
        print(f"  Head params         : {count_parameters(model.classifier):,}")


def build_dataloaders(dataset_name: str, config: dict):
    if dataset_name not in config:
        raise KeyError(f"Unknown downstream dataset: {dataset_name}")

    data_config = config["data"]

    dataset_config = config[dataset_name]
    dataset_path = Path(dataset_config["dataset_path"])

    train_loader, val_loader, test_loader = build_loaders(
        lmdb_path=dataset_path,
        split_ratio=tuple(data_config.get("split_ratio", [0.8, 0.1, 0.1])),
        batch_size=data_config.get("batch_size", 32),
        seed=data_config.get("seed", 42),
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        persistent_workers=data_config.get("persistent_workers", False),
        channel_mode=data_config.get("channel_mode", "mapped"),
        return_ch_names=data_config.get("return_ch_names", False),
        train_ids=dataset_config.get("train_ids"),
        val_ids=dataset_config.get("val_ids"),
        test_ids=dataset_config.get("test_ids"),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def _close_dataset_env(dataset):
    env = getattr(dataset, "env", None)
    if env is not None:
        env.close()
        dataset.env = None


def _infer_loader_channels(loaders: dict) -> int:
    for split_name in ("train", "val", "test"):
        loader = loaders.get(split_name)
        if loader is None or getattr(loader, "dataset", None) is None:
            continue
        if len(loader.dataset) == 0:
            continue

        dataset = loader.dataset
        try:
            sample = dataset[0]
        finally:
            # Reading one sample opens the LMDB env in the main process. Close it
            # before DataLoader workers fork, otherwise LMDB can segfault in workers.
            _close_dataset_env(dataset)

        x = sample[0] if isinstance(sample, (tuple, list)) else sample
        if x.dim() != 2:
            raise ValueError(
                f"Expected dataset samples with shape (C, T), got {tuple(x.shape)} "
                f"from {split_name} split."
            )
        return int(x.shape[0])

    raise ValueError("Cannot infer input channels: all downstream splits are empty.")


def _adapt_model_channels_for_raw(
    model_config: dict,
    data_config: dict,
    dataset_config: dict,
    loaders: dict,
) -> dict:
    model_config = dict(model_config)

    channel_mode = str(data_config.get("channel_mode", "mapped")).lower()
    if channel_mode != "raw":
        return model_config

    inferred_channels = _infer_loader_channels(loaders)
    configured_channels = dataset_config.get("n_channels")

    if configured_channels is not None and int(configured_channels) != inferred_channels:
        print(
            "[DOWNSTREAM] Raw channel mismatch: "
            f"dataset config n_channels={configured_channels}, "
            f"but loader samples have {inferred_channels} channels. "
            "Using loader channel count."
        )

    n_channels = inferred_channels
    model_config["in_channels"] = n_channels
    model_config["out_channels"] = n_channels

    print(
        "[DOWNSTREAM] Raw channel mode: "
        f"setting model.in_channels=model.out_channels={n_channels}"
    )
    return model_config


def main(config: dict):
    base_model_config = config.get("model", {})
    dataset_name = config.get("dataset_name")
    training_config = config.get("training", {})
    head_config = config.get("head", {})
    data_config = config.get("data", {})

    if dataset_name is None:
        raise KeyError("Missing `dataset_name` in downstream config.")

    seed_everything(int(training_config.get("seed", 8)))

    dataset_config = config[dataset_name]
    n_classes = dataset_config["n_classes"]
    loaders = build_dataloaders(dataset_name, config)
    model_name, selected_model_config, checkpoint_path = _select_model_config(
        base_model_config=base_model_config,
        training_config=training_config,
    )

    if model_name == "EEGVAE":
        model_config = _adapt_model_channels_for_raw(
            model_config=selected_model_config,
            data_config=data_config,
            dataset_config=dataset_config,
            loaders=loaders,
        )
    else:
        model_config = selected_model_config
        if checkpoint_path is not None:
            print(f"[DOWNSTREAM] {model_name} checkpoint: {checkpoint_path}")

    trainer_config = add_eegvae_checkpoint_type_dir(
        training_config=training_config,
        model_name=model_name,
        model_config=model_config,
    )
    trainer_config.setdefault("n_classes", n_classes)
    trainer_config.setdefault("seed", 8)
    if "class_weights" in dataset_config and "class_weights" not in trainer_config:
        trainer_config["class_weights"] = dataset_config["class_weights"]

    model = build_model(
        model_name=model_name,
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        dataset_name=dataset_name,
        data_config=data_config,
        head_config=head_config,
    )

    print_model_size(model)
    print(f"[DOWNSTREAM] checkpoint dir: {trainer_config['model_checkpoint_dir']}")

    trainer = downstream_Trainer(
        model=model,
        loaders=loaders,
        n_classes=n_classes,
        config=trainer_config,
    )

    trainer.downstream()


if __name__ == "__main__":

    config_path = "/home/infres/yrothlin-24/EEG-VAE/configs/downstream.yaml"
    config = get_config(config_path)
    parsed_args = downstream_parser.parse_args()

    if parsed_args.use_parsing:
        config = update_config(config, parsed_args)

    main(config)
