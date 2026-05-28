import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from torch.cuda.amp import GradScaler, autocast

from .metrics import Metrics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Trainer:
    def __init__(self, model, loaders, n_classes=None, config=None):
        self.model = model
        self.loaders = loaders
        self.n_classes = n_classes
        self.config = {} if config is None else config
        self.seed = self.config.get("seed", None)
        if self.seed is not None:
            self._seed_everything(int(self.seed))

        self.task = self.config.get("task", "classification")
        self.num_epochs = self.config.get("num_epochs", 20)
        self.grad_clip = self.config.get("grad_clip", None)
        self.use_amp = self.config.get("use_amp", False)
        self.log_step = self.config.get("log_step", 50)
        self.max_step_train = self.config.get("max_step_train", None)
        self.max_step_val = self.config.get("max_step_val", None)
        self.max_step_test = self.config.get("max_step_test", None)
        self.early_stopping_patience = self.config.get("early_stopping_patience", None)
        self.early_stopping_min_delta = self.config.get("early_stopping_min_delta", 0.0)
        self.restore_best_checkpoint = self.config.get(
            "restore_best_checkpoint", True
        )
        self.metric_monitoring = self.config.get("metric_monitoring", "BACC")

        self.loss_dir = Path(self.config.get("loss_dir", "logs/downstream_losses"))
        self.checkpoint_dir = Path(
            self.config.get("model_checkpoint_dir", "checkpoints/downstream")
        )
        self.tsne_enabled = self.config.get("tsne_enabled", True)
        self.tsne_dir = Path(self.config.get("tsne_dir", "logs/downstream/tsne"))
        self.tsne_splits = self.config.get("tsne_splits", ["train", "val", "test"])
        self.tsne_max_samples = self.config.get("tsne_max_samples", 2000)
        self.tsne_perplexity = self.config.get("tsne_perplexity", 30.0)
        self.tsne_pca_enabled = self.config.get("tsne_pca_enabled", True)
        self.tsne_pca_components = self.config.get("tsne_pca_components", 50)
        self.tsne_pca_whiten = self.config.get("tsne_pca_whiten", False)
        self.tsne_learning_rate = self.config.get("tsne_learning_rate", "auto")
        if isinstance(self.tsne_learning_rate, str) and self.tsne_learning_rate != "auto":
            self.tsne_learning_rate = float(self.tsne_learning_rate)
        self.tsne_max_iter = self.config.get(
            "tsne_max_iter",
            self.config.get("tsne_n_iter", 1000),
        )
        self.tsne_random_state = self.config.get("tsne_random_state", 42)
        self.tsne_dpi = self.config.get("tsne_dpi", 180)
        self.tsne_figsize = tuple(self.config.get("tsne_figsize", [8, 6]))
        self.tsne_feature_source = self.config.get("tsne_feature_source", "backbone")

        self.device = torch.device(
            self.config.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )

        self.current_epoch = 0
        self.global_step = 0
        self.best_score = None
        self.best_epoch = None
        self.best_checkpoint_saved = False
        self.bad_epochs = 0
        self.history = []
        self._printed_model_size_after_first_inference = False

        self.model.to(self.device)

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self.use_amp)

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _build_criterion(self):
        loss_name = self.config.get("loss", "cross_entropy").lower()

        if self.task == "regression":
            if loss_name in {"mse", "l2"}:
                return nn.MSELoss()
            if loss_name in {"l1", "mae"}:
                return nn.L1Loss()
            if loss_name == "smooth_l1":
                return nn.SmoothL1Loss()
            raise ValueError(f"Unknown regression loss: {loss_name}")

        class_weights = self.config.get("class_weights", None)
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss(weight=weight)

        if loss_name == "label_smoothing":
            smoothing = self.config.get("label_smoothing", 0.1)
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=smoothing)

        if loss_name in {"bce", "bce_with_logits"}:
            return nn.BCEWithLogitsLoss()

        raise ValueError(f"Unknown classification loss: {loss_name}")

    def _build_optimizer(self):
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 0.0)
        betas = tuple(self.config.get("betas", [0.9, 0.999]))

        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found for downstream training.")

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                params, lr=lr, betas=betas, weight_decay=weight_decay
            )

        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

        if optimizer_name == "sgd":
            momentum = self.config.get("momentum", 0.9)
            return torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )

        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _build_scheduler(self):
        scheduler_name = self.config.get("scheduler", None)
        if scheduler_name is None:
            return None

        scheduler_name = scheduler_name.lower().replace("-", "_")
        if scheduler_name == "none":
            return None

        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )

        if scheduler_name in {"onecycle", "one_cycle", "onecyclelr", "one_cycle_lr"}:
            train_loader = self.loaders.get("train")
            if train_loader is None:
                raise ValueError("OneCycleLR requires a train loader.")

            steps_per_epoch = len(train_loader)
            if self.max_step_train is not None:
                steps_per_epoch = min(steps_per_epoch, int(self.max_step_train))
            steps_per_epoch = max(1, steps_per_epoch)

            max_lr = self.config.get("max_lr")
            if max_lr is None:
                max_lr = self.config.get("lr", 1e-3)

            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                epochs=self.num_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.config.get("onecycle_pct_start", 0.2),
                div_factor=self.config.get("onecycle_div_factor", 25.0),
                final_div_factor=self.config.get("onecycle_final_div_factor", 1e4),
            )

        if scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("step_size", 10),
                gamma=self.config.get("gamma", 0.1),
            )

        if scheduler_name == "plateau":
            mode = "min" if self._monitor_is_loss() else "max"
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=self.config.get("plateau_factor", 0.5),
                patience=self.config.get("plateau_patience", 3),
            )

        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def _scheduler_steps_per_batch(self):
        return isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def _count_initialized_parameters(self, module, trainable_only=False):
        return sum(
            param.numel()
            for param in module.parameters()
            if not isinstance(param, UninitializedParameter)
            and (not trainable_only or param.requires_grad)
        )

    def _count_uninitialized_parameters(self, module):
        return sum(
            1
            for param in module.parameters()
            if isinstance(param, UninitializedParameter)
        )

    def _print_model_size_after_first_inference(self):
        if self._printed_model_size_after_first_inference:
            return

        total_params = self._count_initialized_parameters(self.model)
        trainable_params = self._count_initialized_parameters(self.model, trainable_only=True)
        frozen_params = total_params - trainable_params
        uninitialized_params = self._count_uninitialized_parameters(self.model)

        print("Model size after first inference:")
        print(f"  Initialized params  : {total_params:,}")
        print(f"  Trainable params    : {trainable_params:,}")
        print(f"  Frozen params       : {frozen_params:,}")
        if uninitialized_params:
            print(f"  Lazy params pending : {uninitialized_params}")
        if hasattr(self.model, "backbone"):
            print(
                "  Backbone params     : "
                f"{self._count_initialized_parameters(self.model.backbone):,}"
            )
        if hasattr(self.model, "classifier"):
            print(
                "  Head params         : "
                f"{self._count_initialized_parameters(self.model.classifier):,}"
            )

        self._printed_model_size_after_first_inference = True

    def _plot_tsne(self):
        self._plot_tsne_stage("tsne")

    def _extract_tsne_features(self, x):
        source = str(self.tsne_feature_source).lower()

        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        elif hasattr(self.model, "extract_features"):
            features = self.model.extract_features(x)
        elif (
            source in {"backbone", "latent", "encoder"}
            and hasattr(self.model, "backbone")
            and hasattr(self.model.backbone, "encode")
        ):
            posterior = self.model.backbone.encode(x)
            features = posterior.mode()
        else:
            features = self.model(x)

        return features.reshape(features.shape[0], -1)

    @torch.no_grad()
    def _collect_tsne_data(self):
        features = []
        labels = []
        splits = []
        max_samples = self.tsne_max_samples
        if max_samples is not None:
            max_samples = int(max_samples)

        was_training = self.model.training
        self.model.eval()

        try:
            for split in self.tsne_splits:
                loader = self.loaders.get(split)
                if loader is None:
                    continue

                collected = 0
                for batch in loader:
                    if max_samples is not None and collected >= max_samples:
                        break

                    x, y = self._get_batch(batch)
                    if max_samples is not None:
                        remaining = max_samples - collected
                        x = x[:remaining]
                        y = y[:remaining]

                    x = x.to(self.device, non_blocking=True)
                    batch_features = self._extract_tsne_features(x)

                    features.append(batch_features.detach().cpu())
                    labels.append(y.detach().cpu().view(-1))
                    splits.extend([split] * batch_features.shape[0])
                    collected += batch_features.shape[0]
        finally:
            self.model.train(mode=was_training)

        if not features:
            return None, None, None

        features = torch.cat(features, dim=0).float().numpy()
        labels = torch.cat(labels, dim=0).numpy()
        return features, labels, np.asarray(splits)

    def _apply_tsne_pca(self, features, stage):
        if not self.tsne_pca_enabled or self.tsne_pca_components is None:
            return features

        n_samples, n_features = features.shape
        n_components = min(int(self.tsne_pca_components), n_samples - 1, n_features)
        if n_components < 2 or n_components >= n_features:
            return features

        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for downstream PCA before t-SNE. "
                "Install sklearn or set training.tsne_pca_enabled=false."
            ) from exc

        pca = PCA(
            n_components=n_components,
            whiten=bool(self.tsne_pca_whiten),
            random_state=self.tsne_random_state,
        )
        reduced = pca.fit_transform(features)
        explained = float(pca.explained_variance_ratio_.sum())
        print(
            f"[t-SNE] Applied PCA before {stage}: "
            f"{n_features} -> {n_components} dims "
            f"({explained:.2%} variance)."
        )
        return reduced

    def _build_tsne(self, n_samples):
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for downstream t-SNE plotting. "
                "Install sklearn or set training.tsne_enabled=false."
            ) from exc

        perplexity = min(float(self.tsne_perplexity), max(1.0, (n_samples - 1) / 3))
        kwargs = {
            "n_components": 2,
            "perplexity": perplexity,
            "learning_rate": self.tsne_learning_rate,
            "init": "pca",
            "random_state": self.tsne_random_state,
        }

        try:
            return TSNE(max_iter=int(self.tsne_max_iter), **kwargs)
        except TypeError:
            return TSNE(n_iter=int(self.tsne_max_iter), **kwargs)

    def _plot_tsne_stage(self, stage):
        if not self.tsne_enabled:
            return

        features, labels, splits = self._collect_tsne_data()
        if features is None:
            print(f"[t-SNE] Skipping {stage}: no samples found.")
            return

        n_samples = features.shape[0]
        if n_samples < 3:
            print(f"[t-SNE] Skipping {stage}: need at least 3 samples, got {n_samples}.")
            return

        features = self._apply_tsne_pca(features, stage)
        tsne = self._build_tsne(n_samples)
        embedding = tsne.fit_transform(features)

        self.tsne_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=self.tsne_figsize)

        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
        markers = {
            "train": "o",
            "val": "^",
            "test": "s",
        }

        for split in np.unique(splits):
            split_mask = splits == split
            for label_idx, label in enumerate(unique_labels):
                mask = split_mask & (labels == label)
                if not np.any(mask):
                    continue
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=14,
                    alpha=0.72,
                    marker=markers.get(str(split), "o"),
                    color=cmap(label_idx % cmap.N),
                    label=f"{split} | class {label}",
                    linewidths=0,
                )

        ax.set_title(f"t-SNE {stage.replace('_', ' ')}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
        ax.legend(loc="best", fontsize="x-small", ncol=2)
        fig.tight_layout()

        output_path = self.tsne_dir / f"{stage}.png"
        fig.savefig(output_path, dpi=int(self.tsne_dpi))
        plt.close(fig)
        print(f"[t-SNE] Saved {stage} plot to {output_path}")

    def _log_metrics(self):
        pass

    def _get_batch(self, batch):
        if isinstance(batch, dict):
            x = None
            y = None
            for key in ["x", "eeg", "signal", "data", "input", "inputs"]:
                if key in batch:
                    x = batch[key]
                    break
            for key in ["y", "label", "labels", "target", "targets", "class"]:
                if key in batch:
                    y = batch[key]
                    break
            if x is None or y is None:
                raise KeyError(f"Cannot find x/y tensors in batch keys: {list(batch.keys())}")
            return x, y

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(f"Unsupported downstream batch type: {type(batch)}")

    def _prepare_targets(self, targets):
        targets = targets.to(self.device, non_blocking=True)
        if self.task == "regression":
            return targets.float()
        return targets.long().view(-1)

    def _compute_loss(self, outputs, targets):
        loss_name = self.config.get("loss", "cross_entropy").lower()

        if self.task == "regression":
            return self.criterion(outputs.float(), targets.float())

        if loss_name in {"bce", "bce_with_logits"}:
            return self.criterion(outputs.view(-1).float(), targets.float().view(-1))

        return self.criterion(outputs, targets)

    def _new_metrics(self):
        return Metrics(task=self.task, num_classes=self.n_classes)

    def _run_epoch(self, loader, train=False, max_steps=None):
        if loader is None:
            return {}

        self.model.train(mode=train)
        metrics = self._new_metrics()
        total_loss = 0.0
        n_steps = 0
        n_samples = 0

        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            x, targets = self._get_batch(batch)
            x = x.to(self.device, non_blocking=True)
            targets = self._prepare_targets(targets)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with autocast(enabled=self.use_amp):
                    outputs = self.model(x)
                    self._print_model_size_after_first_inference()
                    loss = self._compute_loss(outputs, targets)

                if train:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler is not None and self._scheduler_steps_per_batch():
                        self.scheduler.step()
                    self.global_step += 1

            batch_size = x.shape[0]
            total_loss += float(loss.detach()) * batch_size
            n_samples += batch_size
            n_steps += 1
            metrics.update(outputs.detach(), targets.detach())

            # if train and self.log_step and step % self.log_step == 0:
            #     print(
            #         f"epoch={self.current_epoch + 1}/{self.num_epochs} "
            #         f"step={step} loss={float(loss.detach()):.6f}"
            #     )

        if n_steps == 0 or n_samples == 0:
            return {}

        logs = {"loss": total_loss / n_samples}
        logs.update(metrics.compute())
        return logs

    def train_one_epoch(self):
        return self._run_epoch(
            self.loaders.get("train"),
            train=True,
            max_steps=self.max_step_train,
        )

    @torch.no_grad()
    def validate_one_epoch(self):
        logs = self._run_epoch(
            self.loaders.get("val"),
            train=False,
            max_steps=self.max_step_val,
        )
        return {f"val_{key}": value for key, value in logs.items()}

    @torch.no_grad()
    def test(self):
        logs = self._run_epoch(
            self.loaders.get("test"),
            train=False,
            max_steps=self.max_step_test,
        )
        return {f"test_{key}": value for key, value in logs.items()}

    def _monitor_is_loss(self):
        return "loss" in self.metric_monitoring.lower()

    def _monitor_key(self):
        key = self.metric_monitoring
        if key in {"loss", "ACC", "BACC", "AUROC", "KAPPA", "F1_macro", "F1_weighted"}:
            return f"val_{key}"
        return key

    def _is_better(self, score):
        min_delta = float(self.early_stopping_min_delta)
        if self.best_score is None:
            return True
        if self._monitor_is_loss():
            return score < self.best_score - min_delta
        return score > self.best_score + min_delta

    def _save_history(self):
        if not self.history:
            return

        self.loss_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = ["epoch"] + sorted(
            {
                key
                for epoch_logs in self.history
                for key in epoch_logs
                if key != "epoch" and key != "confusion_matrix"
            }
        )

        with open(self.loss_dir / "downstream_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow(
                    {
                        key: value
                        for key, value in row.items()
                        if key in fieldnames
                    }
                )

    def _plot_history(self):
        if not self.history:
            return

        self.loss_dir.mkdir(parents=True, exist_ok=True)
        epochs = [row["epoch"] for row in self.history]
        keys = [
            key
            for key in ["loss", "val_loss", "ACC", "val_ACC", "BACC", "val_BACC"]
            if any(key in row for row in self.history)
        ]
        if not keys:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        for key in keys:
            values = [row.get(key, float("nan")) for row in self.history]
            ax.plot(epochs, values, marker="o", linewidth=1.5, label=key)

        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.loss_dir / "downstream_history.png", dpi=150)
        plt.close(fig)

    def _checkpoint_history(self):
        return [
            {
                key: value
                for key, value in row.items()
                if not key.endswith("confusion_matrix")
            }
            for row in self.history
        ]

    def _save_checkpoint(self, name):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": self.config,
            "n_classes": self.n_classes,
            "best_score": self.best_score,
            "history": self._checkpoint_history(),
        }
        torch.save(checkpoint, self.checkpoint_dir / name)

    def _load_checkpoint(self, name):
        checkpoint_path = self.checkpoint_dir / name
        if not checkpoint_path.exists():
            return False

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        return True

    def _print_epoch(self, logs):
        msg = f"epoch={self.current_epoch + 1}/{self.num_epochs}"
        for key, value in logs.items():
            if key.endswith("confusion_matrix"):
                continue
            try:
                msg += f" {key}={float(value):.6f}"
            except (TypeError, ValueError):
                pass
        print(msg)

    def downstream(self):
        monitor_key = self._monitor_key()

        self._plot_tsne_stage("before_finetuning")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            train_logs = self.train_one_epoch()
            val_logs = self.validate_one_epoch()
            logs = {**train_logs, **val_logs}

            if self.scheduler is not None and not self._scheduler_steps_per_batch():
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_value = logs.get(monitor_key, logs.get("val_loss"))
                    if monitor_value is not None:
                        self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()

            self.history.append({"epoch": epoch + 1, **logs})
            self._save_history()
            self._plot_history()
            self._print_epoch(logs)

            monitor_value = logs.get(monitor_key)
            if monitor_value is None and monitor_key != "val_loss":
                monitor_value = logs.get("val_loss")

            if monitor_value is not None and self._is_better(monitor_value):
                self.best_score = float(monitor_value)
                self.best_epoch = epoch + 1
                self.best_checkpoint_saved = True
                self.bad_epochs = 0
                self._save_checkpoint("downstream_best.pt")
            else:
                self.bad_epochs += 1

            if (
                self.early_stopping_patience is not None
                and self.bad_epochs >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best {monitor_key}={self.best_score:.6f} "
                    f"at epoch {self.best_epoch}."
                )
                break

        current_state = None
        if not self.restore_best_checkpoint:
            current_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }

        best_loaded = (
            self._load_checkpoint("downstream_best.pt")
            if self.best_checkpoint_saved
            else False
        )
        if best_loaded:
            if self.restore_best_checkpoint:
                print(
                    f"Restored best checkpoint from epoch {self.best_epoch} "
                    f"before test evaluation."
                )
            self._plot_tsne_stage("best_model")
        else:
            self._plot_tsne_stage("best_model")

        if current_state is not None and best_loaded:
            self.model.load_state_dict(current_state)

        test_logs = self.test()
        if test_logs:
            self.history[-1].update(test_logs)
            self._save_history()
            self._print_epoch(test_logs)

        self._save_checkpoint("downstream_final.pt")
