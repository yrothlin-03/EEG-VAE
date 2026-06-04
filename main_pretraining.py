from pathlib import Path
import yaml

from utils import pretraining_Trainer, pretraining_parser
from eeg_preprocessing.loaders import get_pretraining_loader, EEGAugment
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



def model_subdir(model_config: dict) -> str:
    sequence_block = str(model_config.get("sequence_block", "attention")).lower()
    if sequence_block == "attention":
        block_dir = "transformers"
    elif sequence_block == "mamba":
        block_dir = "mamba"
    else:
        raise ValueError(
            f"Unknown model.sequence_block={sequence_block!r}. "
            "Expected 'attention' or 'mamba'."
        )

    model_type = str(model_config.get("model_type", "kl")).lower()
    if model_type == "kl":
        return str(Path("kl") / block_dir)
    if model_type == "vq":
        n_q = int(model_config.get("vq_n_quantizers", 1))
        return str(Path("vq") / block_dir / f"q{n_q}")
    raise ValueError(
        f"Unknown model.model_type={model_type!r}. Expected 'kl' or 'vq'."
    )


def _append_subdir(path: str, subdir: str) -> str:
    p = Path(path)
    subdir_parts = Path(subdir).parts
    if p.parts[-len(subdir_parts):] == subdir_parts:
        return str(p)
    return str(p / subdir)


def add_type_dirs(training_config: dict, model_config: dict) -> dict:
    training_config = dict(training_config)
    subdir = model_subdir(model_config)
    for key in ("model_checkpoint_dir", "recon_figures_dir", "loss_dir"):
        if key in training_config:
            training_config[key] = _append_subdir(training_config[key], subdir)
    training_config["checkpoint_type"] = subdir
    return training_config

KL_WEIGHT_DEFAULTS = {"kl": 1e-5, "vq": 1.0}


def resolve_kl_weight(training_config: dict, model_config: dict) -> dict:
    loss_config = training_config.get("loss")
    if loss_config is None:
        return training_config

    weight = loss_config.get("kl_weight", "auto")
    if isinstance(weight, (int, float)) and not isinstance(weight, bool):
        return training_config

    model_type = str(model_config.get("model_type", "kl")).lower()
    if model_type not in KL_WEIGHT_DEFAULTS:
        raise ValueError(
            f"Cannot auto-resolve kl_weight for unknown model_type={model_type!r}."
        )

    resolved = KL_WEIGHT_DEFAULTS[model_type]
    loss_config["kl_weight"] = resolved
    print(f"[LOSS] kl_weight=auto → {resolved:g} (model_type={model_type})")
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


def build_augmenter(augment_config: dict | None) -> EEGAugment | None:
    if not augment_config or not augment_config.get("enabled", True):
        return None

    kwargs = {}
    for key in ("time_shift_ratio", "max_channel_dropout", "noise_std", "p"):
        if key in augment_config:
            kwargs[key] = augment_config[key]
    if "amplitude_range" in augment_config:
        kwargs["amplitude_range"] = tuple(augment_config["amplitude_range"])
    if "time_mask_ratio" in augment_config:
        kwargs["time_mask_ratio"] = tuple(augment_config["time_mask_ratio"])

    return EEGAugment(**kwargs)


def build_dataloaders(data_config: dict):
    dataset_paths = data_config.get("dataset_paths", data_config.get("dataset_path"))
    if dataset_paths is None:
        raise KeyError("data config must define 'dataset_path' or 'dataset_paths'.")

    augment_config = data_config.get("augment", {})
    augment_train = bool(augment_config.get("enabled", True)) if augment_config else True
    augmenter = build_augmenter(augment_config)

    train_loader, val_loader = get_pretraining_loader(
        lmdb_paths=dataset_paths,
        split_ratio=tuple(data_config.get("split_ratio", [0.9, 0.1, 0.0])),
        batch_size=data_config.get("batch_size", 32),
        seed=data_config.get("seed", 42),
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        persistent_workers=data_config.get("persistent_workers", False),
        channel_mode=data_config.get("channel_mode", "mapped"),
        return_ch_names=data_config.get("return_ch_names", False),
        augmenter=augmenter,
        augment_train=augment_train,
        drop_last=data_config.get("drop_last", True),
    )

    print(
        f"[LOADER] augmentation {'ON' if augment_train else 'OFF'} on train split"
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": None,
    }


def main(config: dict):
    model_config = config.get("model", {})
    discriminator_config = config.get("discriminator", {})
    data_config = config.get("data", {})
    training_config = add_type_dirs(
        config.get("training", {}),
        model_config,
    )
    training_config = resolve_kl_weight(training_config, model_config)

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
