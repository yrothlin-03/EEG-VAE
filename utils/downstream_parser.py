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


def nullable_int(value):
    if value is None:
        return None

    if isinstance(value, int):
        return value

    value = value.lower()
    if value in {"none", "null"}:
        return None

    return int(value)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_parsing", action="store_true")

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--n_channels", type=int, default=None)
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--train_ids", type=str, nargs="+", default=None)
    parser.add_argument("--val_ids", type=str, nargs="+", default=None)
    parser.add_argument("--test_ids", type=str, nargs="+", default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--split_ratio", type=float, nargs=3, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", type=str2bool, default=None)
    parser.add_argument("--persistent_workers", type=str2bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--channel_mode", type=str, default=None)
    parser.add_argument("--return_ch_names", type=str2bool, default=None)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_config_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--model_weights", type=str, default=None)
    parser.add_argument("--strict_checkpoint_loading", type=str2bool, default=None)
    parser.add_argument("--loss_dir", type=str, default=None)
    parser.add_argument("--model_checkpoint_dir", type=str, default=None)
    parser.add_argument("--tsne_enabled", type=str2bool, default=None)
    parser.add_argument("--tsne_dir", type=str, default=None)
    parser.add_argument("--tsne_splits", type=str, nargs="+", default=None)
    parser.add_argument("--tsne_max_samples", type=int, default=None)
    parser.add_argument("--tsne_perplexity", type=float, default=None)
    parser.add_argument("--tsne_pca_enabled", type=str2bool, default=None)
    parser.add_argument("--tsne_pca_components", type=int, default=None)
    parser.add_argument("--tsne_pca_whiten", type=str2bool, default=None)
    parser.add_argument("--tsne_learning_rate", default=None)
    parser.add_argument("--tsne_max_iter", type=int, default=None)
    parser.add_argument("--tsne_random_state", type=int, default=None)
    parser.add_argument("--tsne_dpi", type=int, default=None)
    parser.add_argument("--tsne_figsize", type=float, nargs=2, default=None)
    parser.add_argument("--tsne_feature_source", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--onecycle_pct_start", type=float, default=None)
    parser.add_argument("--onecycle_div_factor", type=float, default=None)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--betas", type=float, nargs=2, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--use_amp", type=str2bool, default=None)
    parser.add_argument("--log_step", type=int, default=None)
    parser.add_argument("--max_step_train", type=int, default=None)
    parser.add_argument("--max_step_val", type=int, default=None)
    parser.add_argument("--max_step_test", type=int, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument(
        "--early_stopping_patience",
        type=nullable_int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--restore_best_checkpoint", type=str2bool, default=None)
    parser.add_argument("--metric_monitoring", type=str, default=None)
    parser.add_argument("--plateau_factor", type=float, default=None)
    parser.add_argument("--plateau_patience", type=int, default=None)
    parser.add_argument("--output_dim", type=int, default=None)

    parser.add_argument("--head_type", type=str, default=None)
    parser.add_argument("--pooling", type=str, default=None)
    parser.add_argument("--head_n_layer", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--head_norm", type=str, default=None)
    parser.add_argument("--probe_max_norm", type=float, default=None)
    parser.add_argument("--classifier_max_norm", type=float, default=None)

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

    return parser


def parse_args():
    return get_parser().parse_args()


def build_config_from_args(args):
    config = {
        "data": {},
        "training": {},
        "head": {},
        "model": {},
    }

    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name is not None:
        config["dataset_name"] = dataset_name
        config.setdefault(dataset_name, {})

    dataset_keys = [
        "dataset_path",
        "n_channels",
        "n_classes",
        "patch_size",
        "train_ids",
        "val_ids",
        "test_ids",
    ]
    data_keys = [
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
    ]
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
    ]

    dataset_overrides = {}
    for key in dataset_keys:
        value = getattr(args, key, None)
        if value is not None:
            dataset_overrides[key] = value

    if dataset_overrides:
        if dataset_name is None:
            raise ValueError("--dataset_name is required when overriding dataset-specific fields")
        config.setdefault(dataset_name, {}).update(dataset_overrides)

    for key in data_keys:
        value = getattr(args, key, None)
        if value is not None:
            config["data"][key] = value

    for key in training_keys:
        value = getattr(args, key, None)
        if value is not None:
            config["training"][key] = value

    for arg_key, config_key in head_mapping.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            config["head"][config_key] = value

    for key in model_keys:
        value = getattr(args, key, None)
        if value is not None:
            config["model"][key] = value

    for key in ["data", "training", "head", "model"]:
        if not config[key]:
            config.pop(key)

    return config


parser = get_parser()
