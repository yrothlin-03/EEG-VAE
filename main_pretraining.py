from pathlib import Path
import yaml

from utils import pretraining_Trainer, pretraining_parser
from eeg_preprocessing.loaders import build_loaders
from eeg_vae import EEGVAE, Discriminator


def get_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def update_config(base_config, parsed_args):
    parsed_dict = vars(parsed_args)
    key_mapping = {
        "disc_in_channels": ("discriminator", "in_channels"),
    }

    def update_nested(config, key, value):
        for config_key, config_value in config.items():
            if config_key == key:
                config[config_key] = value
                return True

            if isinstance(config_value, dict) and update_nested(config_value, key, value):
                return True

        return False

    for key, value in parsed_dict.items():
        if key == "use_parsing" or value is None:
            continue

        if key in key_mapping:
            section, config_key = key_mapping[key]
            base_config.setdefault(section, {})[config_key] = value
            continue

        if not update_nested(base_config, key, value):
            base_config[key] = value

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


def add_checkpoint_type_dir(training_config: dict, model_config: dict) -> dict:
    training_config = dict(training_config)
    checkpoint_root = Path(training_config["model_checkpoint_dir"])
    checkpoint_type = checkpoint_subdir_for_sequence_block(model_config)
    if checkpoint_root.name not in {"mamba", "transformers"}:
        checkpoint_root = checkpoint_root / checkpoint_type
    training_config["model_checkpoint_dir"] = str(checkpoint_root)
    training_config["checkpoint_type"] = checkpoint_type
    return training_config

def build_model(model_config: dict, discriminator_config: dict):
    output_channels = model_config.get(
        "out_channels",
        model_config["in_channels"],
    )

    discriminator_in_channels = discriminator_config.get("in_channels")

    if discriminator_in_channels != output_channels:
        raise ValueError(
            f"discriminator.in_channels={discriminator_in_channels} "
            f"doit être égal à model.out_channels={output_channels}, "
            "car le discriminateur reçoit les EEG reconstruits/originaux."
        )

    model = EEGVAE(**model_config)
    discriminator = Discriminator(**discriminator_config)

    return model, discriminator


def count_parameters(module):
    return sum(param.numel() for param in module.parameters())


def print_model_size(model, discriminator):
    model_params = count_parameters(model)
    discriminator_params = count_parameters(discriminator)
    total_params = model_params + discriminator_params

    print("Model size before training:")
    print(f"  EEGVAE params       : {model_params:,}")
    print(f"  Discriminator params: {discriminator_params:,}")
    print(f"  Total params        : {total_params:,}")


def build_dataloaders(data_config: dict):
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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def main(config: dict):
    model_config = config.get("model", {})
    discriminator_config = config.get("discriminator", {})
    data_config = config.get("data", {})
    training_config = add_checkpoint_type_dir(
        config.get("training", {}),
        model_config,
    )

    model, discriminator = build_model(
        model_config=model_config,
        discriminator_config=discriminator_config,
    )

    print_model_size(model, discriminator)
    print(f"[PRETRAINING] checkpoint dir: {training_config['model_checkpoint_dir']}")

    loaders = build_dataloaders(data_config)

    trainer = pretraining_Trainer(
        model=model,
        discriminator=discriminator,
        loaders=loaders,
        training_config=training_config,
    )

    trainer.pretrain()


if __name__ == "__main__":

    config_path = "/home/infres/yrothlin-24/EEG-VAE/configs/pretraining.yaml"

    config = get_config(config_path)

    parsed_args = pretraining_parser.parse_args()

    if parsed_args.use_parsing:
        config = update_config(config, parsed_args)

    main(config)
