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
    parser = argparse.ArgumentParser(description="EEG-JEPA Phase 2 pretraining")

    parser.add_argument("--use_parsing", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="Path to jepa.yaml config")

    # Data
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Training
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--use_amp", type=str2bool, default=None)
    parser.add_argument("--max_step_train", type=int, default=None)
    parser.add_argument("--max_step_val", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)

    # JEPA-specific
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--target_ratio", type=float, default=None)
    parser.add_argument("--ema_momentum", type=float, default=None)

    # Phase 1 checkpoint
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to the Phase 1 VQ-VAE checkpoint (.pt)")

    return parser


def parse_args():
    return get_parser().parse_args()
