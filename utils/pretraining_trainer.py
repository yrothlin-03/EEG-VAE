import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .losses import VAEGANLoss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Trainer:
    def __init__(self, model, discriminator, loaders, training_config):
        self.model = model
        self.discriminator = discriminator
        self.loaders = loaders
        self.training_config = training_config
        self.loss_config = training_config["loss"]
        self.recon_figures_dir = Path(training_config["recon_figures_dir"])
        self.model_checkpoint_dir = Path(training_config["model_checkpoint_dir"])
        self.loss_dir = Path(training_config["loss_dir"])

        self.num_random_recon_batches = training_config.get("num_random_recon_batches", 5)

        self.device = torch.device(
            training_config.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )

        self.num_epochs = training_config.get("num_epochs", 20)
        self.grad_clip = training_config.get("grad_clip", None)
        self.use_amp = training_config.get("use_amp", False)
        self.log_step = training_config.get("log_step", 50)
        self.max_step_train = training_config.get("max_step_train", None)
        self.max_step_val = training_config.get("max_step_val", None)

        self.adversarial_start_epoch = training_config.get("adversarial_start_epoch", 0)
        self.discriminator_update_freq = training_config.get("discriminator_update_freq", 1)
        self.generator_update_freq = training_config.get("generator_update_freq", 1)

        self.current_epoch = 0
        self.global_step = 0
        self.ae_optimizer_steps_this_epoch = 0
        self.disc_optimizer_steps_this_epoch = 0
        self.loss_history = []

        self.model.to(self.device)
        self.discriminator.to(self.device)

        self.criterion = self._build_criterion()
        self.optimizer_ae, self.optimizer_disc = self._build_optimizer()
        self.scheduler_ae, self.scheduler_disc = self._build_scheduler()

        self.scaler = GradScaler(enabled=self.use_amp)

    def _build_criterion(self):
        return VAEGANLoss(**self.loss_config)

    def _build_optimizer(self):
        optimizer_name = self.training_config.get("optimizer", "adamw").lower()

        lr = self.training_config.get("lr", 1e-3)
        discriminator_lr = self.training_config.get("discriminator_lr", lr)

        weight_decay = self.training_config.get("weight_decay", 0.0)
        discriminator_weight_decay = self.training_config.get(
            "discriminator_weight_decay",
            weight_decay,
        )

        betas = tuple(self.training_config.get("betas", [0.5, 0.9]))

        if optimizer_name == "adamw":
            optimizer_cls = torch.optim.AdamW
        elif optimizer_name == "adam":
            optimizer_cls = torch.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        optimizer_ae = optimizer_cls(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        optimizer_disc = optimizer_cls(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=betas,
            weight_decay=discriminator_weight_decay,
        )

        return optimizer_ae, optimizer_disc

    def _build_scheduler(self):
        scheduler_name = self.training_config.get("scheduler", None)

        if scheduler_name is None:
            return None, None

        scheduler_name = scheduler_name.lower()

        if scheduler_name == "cosine":
            scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_ae,
                T_max=self.num_epochs,
            )
            scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_disc,
                T_max=self.num_epochs,
            )
            return scheduler_ae, scheduler_disc

        if scheduler_name == "none":
            return None, None

        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def _get_batch(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, dict):
            for key in ["x", "eeg", "signal", "data"]:
                if key in batch:
                    return batch[key]
            raise KeyError(f"Cannot find EEG tensor in batch keys: {list(batch.keys())}")

        if isinstance(batch, (list, tuple)):
            return batch[0]

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _check_reconstruction_shapes(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes do not match. "
                f"pred.shape={tuple(pred.shape)}, target.shape={tuple(target.shape)}. "
                "Your AutoencoderKL should output the original EEG channel count."
            )

    def _forward_autoencoder(self, x, sample_posterior=True):
        x_adapted = self.model.channel_adaptor(x)
        pred, posterior = self.model.autoencoder(
            x_adapted,
            sample_posterior=sample_posterior,
            return_posterior=True,
        )

        target = x
        self._check_reconstruction_shapes(pred, target)

        return pred, target, posterior

    def _is_non_zero_batch(self, target, eps=1e-8):
        return target.detach().float().abs().amax() > eps
    
    def _plot_recons_figures(self, target, pred, epoch, batch_idx, max_channels=4):
        self.recon_figures_dir.mkdir(parents=True, exist_ok=True)

        target = target.detach().float().cpu()
        pred = pred.detach().float().cpu()

        # self._check_reconstruction_shapes(pred, target)

        target = target[0]
        pred = pred[0]

        n_channels = min(max_channels, target.shape[0], pred.shape[0])

        fig, axes = plt.subplots(
            n_channels,
            1,
            figsize=(14, 3 * n_channels),
            squeeze=False,
        )

        for ch in range(n_channels):
            ax = axes[ch, 0]

            target_ch = target[ch].numpy()
            pred_ch = pred[ch].numpy()

            ax.plot(target_ch, label="original EEG", linewidth=1.0)
            ax.plot(pred_ch, label="reconstruction EEG", linewidth=1.0)

            ax.set_title(
                f"Epoch {epoch + 1} | Batch {batch_idx} | EEG Channel {ch} | "
                f"target std={target_ch.std():.3g} | "
                f"pred std={pred_ch.std():.3g}"
            )

            ax.legend(loc="upper right")

        plt.tight_layout()

        save_path = (
            self.recon_figures_dir
            / f"epoch_{epoch + 1:04d}_batch_{batch_idx:04d}.png"
        )
        plt.savefig(save_path)
        plt.close(fig)

    def _save_loss_history(self):
        if not self.loss_history:
            return

        self.loss_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = ["epoch"] + sorted(
            {
                key
                for epoch_logs in self.loss_history
                for key in epoch_logs
                if key != "epoch"
            }
        )

        with open(self.loss_dir / "loss_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.loss_history)

    def _save_final_checkpoint(self):
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_ae_state_dict": self.optimizer_ae.state_dict(),
            "optimizer_disc_state_dict": self.optimizer_disc.state_dict(),
            "scheduler_ae_state_dict": (
                self.scheduler_ae.state_dict()
                if self.scheduler_ae is not None
                else None
            ),
            "scheduler_disc_state_dict": (
                self.scheduler_disc.state_dict()
                if self.scheduler_disc is not None
                else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_config": self.training_config,
            "loss_history": self.loss_history,
        }

        save_path = self.model_checkpoint_dir / "pretraining_final.pt"
        torch.save(checkpoint, save_path)
        print(f"Saved final pretraining checkpoint to {save_path}")

    def _is_finite_number(self, value):
        return value == value and value not in (float("inf"), float("-inf"))

    def _plot_metric_lines(self, ax, epochs, series, ylabel, title):
        finite_abs_values = []

        for _, values in series:
            finite_abs_values.extend(
                abs(value)
                for value in values
                if self._is_finite_number(value)
            )

        non_zero_values = [value for value in finite_abs_values if value > 0]
        linthresh = min(non_zero_values) if non_zero_values else 1e-8

        for label, values in series:
            ax.plot(epochs, values, marker="o", linewidth=1.6, markersize=4, label=label)

        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best", fontsize="small", ncol=2)

    def _plot_losses_over_epochs(self):
        if not self.loss_history:
            return

        self.loss_dir.mkdir(parents=True, exist_ok=True)

        epochs = [epoch_logs["epoch"] for epoch_logs in self.loss_history]
        total_loss_keys = [
            "loss",
            "val_loss",
            "generator_loss",
            "disc_loss",
            "val_generator_loss",
            "val_disc_loss",
        ]
        total_series = []

        for key in total_loss_keys:
            if not any(key in epoch_logs for epoch_logs in self.loss_history):
                continue

            total_series.append(
                (
                    key,
                    [
                        float(epoch_logs[key]) if key in epoch_logs else float("nan")
                        for epoch_logs in self.loss_history
                    ],
                )
            )

        weighted_loss_specs = [
            ("rec_loss", "rec_weight", "weighted_rec_loss"),
            ("kl_loss", "kl_weight", "weighted_kl_loss"),
            ("spectral_loss", "spectral_weight", "weighted_spectral_loss"),
            ("adv_loss", "adversarial_weight", "weighted_adv_loss"),
            ("val_rec_loss", "rec_weight", "weighted_val_rec_loss"),
            ("val_kl_loss", "kl_weight", "weighted_val_kl_loss"),
            ("val_spectral_loss", "spectral_weight", "weighted_val_spectral_loss"),
            ("val_adv_loss", "adversarial_weight", "weighted_val_adv_loss"),
        ]
        weighted_series = []

        for key, weight_key, label in weighted_loss_specs:
            if not any(key in epoch_logs for epoch_logs in self.loss_history):
                continue

            weight = float(self.loss_config.get(weight_key, 1.0))
            weighted_series.append(
                (
                    f"{label} ({weight_key}={weight:g})",
                    [
                        float(epoch_logs[key]) * weight
                        if key in epoch_logs
                        else float("nan")
                        for epoch_logs in self.loss_history
                    ],
                )
            )

        if not total_series and not weighted_series:
            return

        n_rows = int(bool(total_series)) + int(bool(weighted_series))
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(12, 5 * n_rows),
            squeeze=False,
            sharex=True,
        )
        row = 0

        if total_series:
            self._plot_metric_lines(
                axes[row, 0],
                epochs,
                total_series,
                ylabel="Loss value (symlog scale)",
                title="Total losses across epochs",
            )
            row += 1

        if weighted_series:
            self._plot_metric_lines(
                axes[row, 0],
                epochs,
                weighted_series,
                ylabel="Weighted contribution (symlog scale)",
                title="Loss components after applying training weights",
            )

        axes[-1, 0].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(self.loss_dir / "losses_over_epochs.png", dpi=150)
        plt.close(fig)

    def train_one_epoch(self):
        self.model.train()
        self.discriminator.train()

        train_loader = self.loaders["train"]

        total_logs = {}
        n_steps = 0
        self.ae_optimizer_steps_this_epoch = 0
        self.disc_optimizer_steps_this_epoch = 0

        use_adversarial = self.current_epoch >= self.adversarial_start_epoch

        for step, batch in enumerate(train_loader):
            if self.max_step_train is not None and step >= self.max_step_train:
                break

            x = self._get_batch(batch)

            x = x.to(self.device, non_blocking=True)

            update_generator = step % self.generator_update_freq == 0
            update_discriminator = (
                use_adversarial
                and step % self.discriminator_update_freq == 0
            )

            if update_generator:
                self.optimizer_ae.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    pred, target, posterior = self._forward_autoencoder(x)

                    if use_adversarial:
                        logits_fake = self.discriminator(pred)
                        loss_ae, logs_ae = self.criterion.generator_loss(
                            pred=pred,
                            target=target,
                            posterior=posterior,
                            logits_fake=logits_fake,
                        )
                    else:
                        loss_ae, logs_ae = self.criterion.pretraining_loss(
                            pred=pred,
                            target=target,
                            posterior=posterior,
                        )

                self.scaler.scale(loss_ae).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer_ae)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer_ae)
                self.scaler.update()
                self.ae_optimizer_steps_this_epoch += 1

            else:
                logs_ae = {}

            if update_discriminator:
                self.optimizer_disc.zero_grad(set_to_none=True)

                with torch.no_grad():
                    pred, target, _ = self._forward_autoencoder(x)

                with autocast(enabled=self.use_amp):
                    logits_real = self.discriminator(target.detach())
                    logits_fake = self.discriminator(pred.detach())

                    loss_disc, logs_disc = self.criterion.discriminator_loss(
                        logits_real=logits_real,
                        logits_fake=logits_fake,
                    )

                self.scaler.scale(loss_disc).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer_disc)
                    nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.grad_clip,
                    )

                self.scaler.step(self.optimizer_disc)
                self.scaler.update()
                self.disc_optimizer_steps_this_epoch += 1

            else:
                logs_disc = {}

            logs = {**logs_ae, **logs_disc}

            for key, value in logs.items():
                total_logs[key] = total_logs.get(key, 0.0) + float(value)

            n_steps += 1
            self.global_step += 1

            # if self.log_step and step % self.log_step == 0:
            #     msg = f"epoch={self.current_epoch + 1}/{self.num_epochs} step={step}"
            #     for key, value in logs.items():
            #         msg += f" {key}={float(value):.6f}"
            #     print(msg)

        if n_steps == 0:
            return {}

        return {key: value / n_steps for key, value in total_logs.items()}

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        self.discriminator.eval()

        val_loader = self.loaders.get("val", None)

        if val_loader is None:
            return {}

        total_logs = {}
        n_steps = 0

        max_val_steps = (
            self.max_step_val
            if self.max_step_val is not None
            else len(val_loader)
        )

        num_to_plot = min(self.num_random_recon_batches, max_val_steps)

        random_plot_steps = set(
            torch.randperm(max_val_steps)[:num_to_plot].tolist()
        )

        # print(
        #     f"Epoch {self.current_epoch + 1}: plotting validation batches "
        #     f"{sorted(random_plot_steps)}"
        # )

        for step, batch in enumerate(val_loader):
            if self.max_step_val is not None and step >= self.max_step_val:
                break

            x = self._get_batch(batch)

            x = x.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                pred, target, posterior = self._forward_autoencoder(x)

                _, logs = self.criterion.pretraining_loss(
                    pred=pred,
                    target=target,
                    posterior=posterior,
                )

                if self.current_epoch >= self.adversarial_start_epoch:
                    logits_fake = self.discriminator(pred)
                    logs["adv_loss"] = (-logits_fake.mean()).detach()

            if step in random_plot_steps and self._is_non_zero_batch(target):
                pred_plot, target_plot, _ = self._forward_autoencoder(
                    x,
                    sample_posterior=False,
                )
                self._plot_recons_figures(
                    target=target_plot,
                    pred=pred_plot,
                    epoch=self.current_epoch,
                    batch_idx=step,
                )

            for key, value in logs.items():
                total_logs[key] = total_logs.get(key, 0.0) + float(value)

            n_steps += 1

        if n_steps == 0:
            return {}

        return {f"val_{key}": value / n_steps for key, value in total_logs.items()}

    def pretrain(self):
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            train_logs = self.train_one_epoch()
            val_logs = self.validate_one_epoch()

            if self.scheduler_ae is not None and self.ae_optimizer_steps_this_epoch > 0:
                self.scheduler_ae.step()

            if self.scheduler_disc is not None and self.disc_optimizer_steps_this_epoch > 0:
                self.scheduler_disc.step()

            logs = {**train_logs, **val_logs}
            self.loss_history.append({"epoch": epoch + 1, **logs})
            self._save_loss_history()
            self._plot_losses_over_epochs()

            msg = f"epoch={epoch + 1}/{self.num_epochs}"
            for key, value in logs.items():
                msg += f" {key}={value:.6f}"
            print(msg)

        self._save_final_checkpoint()
