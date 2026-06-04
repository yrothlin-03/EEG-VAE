from pathlib import Path

import yaml

from eeg_vae.jepa import build_jepa_model
from eeg_preprocessing.loaders import build_loaders
from utils import jepa_Trainer, jepa_parser


def get_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def update_config(base_config, parsed_args):
    parsed_dict = vars(parsed_args)
    jepa_keys = {"block_size", "target_ratio", "ema_momentum"}

    def _set_nested(config, key, value):
        for section_key, section_value in config.items():
            if section_key == key:
                config[section_key] = value
                return True
            if isinstance(section_value, dict) and _set_nested(section_value, key, value):
                return True
        return False

    for key, value in parsed_dict.items():
        if key in ("use_parsing", "config") or value is None:
            continue
        if key == "vae_checkpoint":
            base_config.setdefault("phase1", {})["checkpoint"] = value
            continue
        if key in jepa_keys:
            base_config.setdefault("training", {}).setdefault("jepa", {})[key] = value
            continue
        _set_nested(base_config, key, value)

    return base_config


def jepa_subdir(sequence_block: str, n_quantizers: int) -> str:
    block = str(sequence_block).lower()
    if block == "attention":
        block_dir = "transformers"
    elif block == "mamba":
        block_dir = "mamba"
    else:
        raise ValueError(
            f"Unknown model.sequence_block={block!r}. Expected 'attention' or 'mamba'."
        )
    return str(Path(block_dir) / f"q{n_quantizers}")


def _append_subdir(path: str, subdir: str) -> str:
    p = Path(path)
    subdir_parts = Path(subdir).parts
    if p.parts[-len(subdir_parts):] == subdir_parts:
        return str(p)
    return str(p / subdir)


def add_type_dirs(training_config: dict, model_config: dict, n_quantizers: int) -> dict:
    training_config = dict(training_config)
    subdir = jepa_subdir(model_config.get("sequence_block", "attention"), n_quantizers)
    for key in ("model_checkpoint_dir", "loss_dir"):
        if key in training_config:
            training_config[key] = _append_subdir(training_config[key], subdir)
    training_config["checkpoint_type"] = subdir
    return training_config


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def print_model_size(model):
    ctx = count_parameters(model.context_encoder)
    pred = count_parameters(model.predictor)
    mask = model.mask_token.numel()
    total = ctx + pred + mask
    print("JEPA Phase 2 model size:")
    print(f"  Context encoder params : {ctx:,}")
    print(f"  Predictor params       : {pred:,}")
    print(f"  Mask token params      : {mask:,}")
    print(f"  Total trainable params : {total:,}")
    print(f"  Target encoder params  : {count_parameters(model.target_encoder):,}  (EMA, not optimised)")


def build_dataloaders(data_config):
    train_loader, val_loader, test_loader = build_loaders(
        lmdb_path=Path(data_config["dataset_path"]),
        split_ratio=tuple(data_config.get("split_ratio", [0.8, 0.1, 0.1])),
        batch_size=data_config.get("batch_size", 32),
        seed=data_config.get("seed", 42),
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        persistent_workers=data_config.get("persistent_workers", False),
        channel_mode=data_config.get("channel_mode", "mapped"),
        return_ch_names=data_config.get("return_ch_names", False),
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


def main(config):
    model_config = config["model"]
    predictor_config = config["predictor"]
    data_config = config["data"]
    training_config = config["training"]
    phase1_config = config["phase1"]

    vae_checkpoint_path = phase1_config["checkpoint"]
    device = training_config.get("device", "cuda")
    ema_momentum = training_config.get("jepa", {}).get("ema_momentum", 0.996)

    model = build_jepa_model(
        vae_checkpoint_path=vae_checkpoint_path,
        model_config=model_config,
        predictor_config=predictor_config,
        ema_momentum=ema_momentum,
        device=device,
    )

    print_model_size(model)

    n_quantizers = getattr(model.context_encoder.quantize, "n_quantizers", 1)
    detected_block = getattr(model, "sequence_block", model_config.get("sequence_block", "attention"))
    training_config = add_type_dirs(training_config, {"sequence_block": detected_block}, n_quantizers)

    print(f"[JEPA] Phase 1 checkpoint : {vae_checkpoint_path}")
    print(f"[JEPA] Checkpoint dir     : {training_config['model_checkpoint_dir']}")

    loaders = build_dataloaders(data_config)

    trainer = jepa_Trainer(
        model=model,
        loaders=loaders,
        training_config=training_config,
    )

    trainer.train()


if __name__ == "__main__":
    config_path = "/home/infres/yrothlin-24/EEG-VAE/configs/jepa.yaml"
    config = get_config(config_path)

    parsed_args = jepa_parser.parse_args()
    if parsed_args.use_parsing:
        config = update_config(config, parsed_args)

    main(config)
