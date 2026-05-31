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
    """Override config values from CLI args (mirrors main_pretraining.py)."""
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
