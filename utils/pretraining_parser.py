import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_parsing", action="store_true")

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split_ratio", type=float, nargs=3, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", type=str2bool, default=None)
    parser.add_argument("--persistent_workers", type=str2bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--channel_mode", type=str, default=None)
    parser.add_argument("--return_ch_names", type=str2bool, default=None)

    parser.add_argument("--recon_figures_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--discriminator_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--discriminator_weight_decay", type=float, default=None)
    parser.add_argument("--betas", type=float, nargs=2, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--use_amp", type=str2bool, default=None)
    parser.add_argument("--log_step", type=int, default=None)
    parser.add_argument("--max_step_train", type=int, default=None)
    parser.add_argument("--max_step_val", type=int, default=None)
    parser.add_argument("--recon_loss", type=str, default=None)
    parser.add_argument("--adversarial_start_epoch", type=int, default=None)
    parser.add_argument("--discriminator_update_freq", type=int, default=None)
    parser.add_argument("--generator_update_freq", type=int, default=None)

    parser.add_argument("--rec_weight", type=float, default=None)
    parser.add_argument("--kl_weight", type=float, default=None)
    parser.add_argument("--spectral_weight", type=float, default=None)
    parser.add_argument("--adversarial_weight", type=float, default=None)
    parser.add_argument("--rec_loss_type", type=str, default=None)
    parser.add_argument("--spectral_loss_type", type=str, default=None)

    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--out_channels", type=int, default=None)
    parser.add_argument("--adapted_channels", type=int, default=None)
    parser.add_argument("--adaptor_layers", type=int, default=None)
    parser.add_argument("--z_channels", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--ch", type=int, default=None)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=None)
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--attn_resolutions", type=int, nargs="*", default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--resamp_with_conv", type=str2bool, default=None)
    parser.add_argument("--tanh_out", type=str2bool, default=None)
    parser.add_argument("--use_checkpoint", type=str2bool, default=None)
    parser.add_argument("--sequence_block", type=str, choices=("attention", "mamba"), default=None)
    parser.add_argument("--criss_cross_patch_size", type=int, default=None)

    parser.add_argument("--model_type", type=str, choices=("kl", "vq"), default=None)
    parser.add_argument("--vq_num_embeddings", type=int, default=None)
    parser.add_argument("--vq_commitment_cost", type=float, default=None)
    parser.add_argument("--vq_decay", type=float, default=None)
    parser.add_argument("--vq_n_quantizers", type=int, default=None)

    parser.add_argument("--disc_in_channels", type=int, default=None)
    parser.add_argument("--ndf", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--use_actnorm", type=str2bool, default=None)

    return parser


def parse_args():
    return get_parser().parse_args()


def build_config_from_args(args):
    config = {
        "data": {},
        "training": {
            "loss": {},
        },
        "model": {},
        "discriminator": {},
    }

    data_keys = [
        "dataset_path",
        "batch_size",
        "split_ratio",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "seed",
        "channel_mode",
        "return_ch_names",
    ]

    training_keys = [
        "recon_figures_dir",
        "num_epochs",
        "device",
        "optimizer",
        "scheduler",
        "lr",
        "discriminator_lr",
        "weight_decay",
        "discriminator_weight_decay",
        "betas",
        "grad_clip",
        "use_amp",
        "log_step",
        "max_step_train",
        "max_step_val",
        "recon_loss",
        "adversarial_start_epoch",
        "discriminator_update_freq",
        "generator_update_freq",
    ]

    loss_keys = [
        "rec_weight",
        "kl_weight",
        "spectral_weight",
        "adversarial_weight",
        "rec_loss_type",
        "spectral_loss_type",
    ]

    model_keys = [
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
        "model_type",
        "vq_num_embeddings",
        "vq_commitment_cost",
        "vq_decay",
        "vq_n_quantizers",
    ]

    discriminator_mapping = {
        "disc_in_channels": "in_channels",
        "ndf": "ndf",
        "n_layers": "n_layers",
        "use_actnorm": "use_actnorm",
    }

    for key in data_keys:
        value = getattr(args, key, None)
        if value is not None:
            config["data"][key] = value

    for key in training_keys:
        value = getattr(args, key, None)
        if value is not None:
            config["training"][key] = value

    for key in loss_keys:
        value = getattr(args, key, None)
        if value is not None:
            config["training"]["loss"][key] = value

    for key in model_keys:
        value = getattr(args, key, None)
        if value is not None:
            config["model"][key] = value

    for arg_key, config_key in discriminator_mapping.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            config["discriminator"][config_key] = value

    if not config["training"]["loss"]:
        config["training"].pop("loss")

    return config


parser = get_parser()