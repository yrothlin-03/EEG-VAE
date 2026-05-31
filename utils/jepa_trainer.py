import csv
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from .losses import JEPA_Loss
from eeg_vae.jepa import create_block_mask, create_channel_mask

warnings.filterwarnings("ignore", category=FutureWarning)


class JEPATrainer:
    """
    Phase 2 training loop for EEG-JEPA.

    Differences from pretraining_trainer.Trainer:
    - No discriminator / no adversarial objective.
    - EMA update of the target encoder after every gradient step.
    - Target encoder always kept in eval mode (no BatchNorm stat drift).
    - Masking created fresh each step (same positions for all batch items).
    - NaN-safe loss accumulation (same pattern as pretraining_trainer).
    """

    def __init__(self, model, loaders, training_config):
        self.model = model
        self.loaders = loaders
        self.training_config = training_config
        self.jepa_config = training_config.get("jepa", {})
        self.loss_config = training_config.get("loss", {})

        self.loss_dir = Path(training_config["loss_dir"])
        self.model_checkpoint_dir = Path(training_config["model_checkpoint_dir"])

        self.device = torch.device(
            training_config.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )

        self.num_epochs = training_config.get("num_epochs", 20)
        self.grad_clip = training_config.get("grad_clip", 1.0)
        self.use_amp = training_config.get("use_amp", False)
        self.max_step_train = training_config.get("max_step_train", None)
        self.max_step_val = training_config.get("max_step_val", None)

        self.block_size = self.jepa_config.get("block_size", 25)
        self.target_ratio = self.jepa_config.get("target_ratio", 0.25)
        self.channel_mask_ratio = self.jepa_config.get("channel_mask_ratio", 0.0)

        # EMA momentum schedule — cosine warmup from 1.0 → ema_momentum_base
        # over the full training. Starting at 1.0 freezes the target encoder
        # at initialisation, which prevents the "moving target" instability
        # where train and val loss both grow over epochs.
        self.ema_momentum_base = self.jepa_config.get("ema_momentum", 0.996)
        self.ema_momentum_warmup = bool(self.jepa_config.get("ema_momentum_warmup", True))
        self.model.ema_momentum = 1.0 if self.ema_momentum_warmup else self.ema_momentum_base

        self.current_epoch = 0
        self.global_step = 0
        self.loss_history = []
        self.best_val_loss = float("inf")
        self._T_tokens = None   # inferred lazily from the first batch

        self.model.to(self.device)

        self.criterion = JEPA_Loss(loss_type=self.loss_config.get("loss_type", "mse"))
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self.use_amp)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self):
        name = self.training_config.get("optimizer", "adamw").lower()
        lr = self.training_config.get("lr", 1e-4)
        weight_decay = self.training_config.get("weight_decay", 0.05)
        betas = tuple(self.training_config.get("betas", [0.9, 0.95]))

        params = [
            {"params": self.model.context_encoder.parameters()},
            {"params": self.model.predictor.parameters()},
            {"params": [self.model.mask_token]},
        ]

        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, betas=betas)
        raise ValueError(f"Unknown optimizer: {name!r}")

    def _build_scheduler(self):
        name = self.training_config.get("scheduler", "cosine")
        if name is None or str(name).lower() == "none":
            return None
        if str(name).lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs
            )
        raise ValueError(f"Unknown scheduler: {name!r}")

    # ------------------------------------------------------------------
    # Batch extraction (mirrors pretraining_trainer)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # T_tokens inference (lazy, cached for the run)
    # ------------------------------------------------------------------

    def _current_ema_momentum(self) -> float:
        """Cosine schedule: m(t) = 1 - (1 - m_base) * (1 - cos(π·progress)) / 2.

        At progress=0   → m = 1.0 (target frozen at init)
        At progress=1   → m = m_base
        Smoothly interpolates between, giving the encoder time to find a
        stable representation before the target starts following it.
        """
        if not self.ema_momentum_warmup:
            return self.ema_momentum_base
        progress = self.current_epoch / max(1, self.num_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        return 1.0 - (1.0 - self.ema_momentum_base) * (1 - math.cos(math.pi * progress)) / 2

    def _infer_T_tokens(self, x):
        if self._T_tokens is None:
            with torch.no_grad():
                z = self.model.context_encoder(x[:1].to(self.device))
                self._T_tokens = z.shape[-1]
        return self._T_tokens

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def train_one_epoch(self):
        self.model.train()
        # Target encoder always stays in eval: no BatchNorm stat updates,
        # no dropout — it is a momentum copy, not a trained module.
        self.model.target_encoder.eval()

        train_loader = self.loaders["train"]
        total_logs = {}
        count_logs = {}
        n_steps = 0

        for step, batch in enumerate(train_loader):
            if self.max_step_train is not None and step >= self.max_step_train:
                break

            x = self._get_batch(batch).to(self.device, non_blocking=True)

            T_tokens = self._infer_T_tokens(x)
            _, target_pos = create_block_mask(
                T_tokens, self.block_size, self.target_ratio, self.device
            )
            channel_mask = (
                create_channel_mask(x.shape[1], self.channel_mask_ratio, x.shape[0], self.device)
                if self.channel_mask_ratio > 0 else None
            )

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                z_pred, z_target = self.model(x, target_pos, channel_mask=channel_mask)
                loss, logs = self.criterion(z_pred, z_target)

            if torch.isfinite(loss):
                self.scaler.scale(loss).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.model.context_encoder.parameters())
                        + list(self.model.predictor.parameters())
                        + [self.model.mask_token],
                        self.grad_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # EMA update with the warmup-scheduled momentum
                self.model.update_target_encoder(momentum=self._current_ema_momentum())

            for key, value in logs.items():
                v = float(value)
                if v == v and v not in (float("inf"), float("-inf")):
                    total_logs[key] = total_logs.get(key, 0.0) + v
                    count_logs[key] = count_logs.get(key, 0) + 1

            n_steps += 1
            self.global_step += 1

        if n_steps == 0:
            return {}
        return {key: total_logs[key] / count_logs[key] for key in count_logs}

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()

        val_loader = self.loaders.get("val", None)
        if val_loader is None:
            return {}

        total_logs = {}
        count_logs = {}
        n_steps = 0

        for step, batch in enumerate(val_loader):
            if self.max_step_val is not None and step >= self.max_step_val:
                break

            x = self._get_batch(batch).to(self.device, non_blocking=True)

            T_tokens = self._infer_T_tokens(x)
            _, target_pos = create_block_mask(
                T_tokens, self.block_size, self.target_ratio, self.device
            )
            channel_mask = (
                create_channel_mask(x.shape[1], self.channel_mask_ratio, x.shape[0], self.device)
                if self.channel_mask_ratio > 0 else None
            )

            with autocast(enabled=self.use_amp):
                z_pred, z_target = self.model(x, target_pos, channel_mask=channel_mask)
                _, logs = self.criterion(z_pred, z_target)

            for key, value in logs.items():
                v = float(value)
                if v == v and v not in (float("inf"), float("-inf")):
                    total_logs[key] = total_logs.get(key, 0.0) + v
                    count_logs[key] = count_logs.get(key, 0) + 1

            n_steps += 1

        if n_steps == 0:
            return {}
        return {f"val_{k}": total_logs[k] / count_logs[k] for k in count_logs}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_loss_history(self):
        if not self.loss_history:
            return
        self.loss_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = ["epoch"] + sorted(
            {k for row in self.loss_history for k in row if k != "epoch"}
        )
        with open(self.loss_dir / "jepa_loss_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.loss_history)

    def _build_checkpoint(self) -> dict:
        return {
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "training_config": self.training_config,
            "loss_history": self.loss_history,
        }

    def _save_checkpoint(self):
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.model_checkpoint_dir / "jepa_final.pt"
        torch.save(self._build_checkpoint(), save_path)
        print(f"Saved JEPA checkpoint to {save_path}")

    def _maybe_save_best(self, val_logs: dict):
        """Save jepa_best.pt whenever val_loss improves."""
        val_loss = val_logs.get("val_loss")
        if val_loss is None or val_loss != val_loss or val_loss in (float("inf"), float("-inf")):
            return

        if val_loss < self.best_val_loss:
            previous = self.best_val_loss
            self.best_val_loss = float(val_loss)
            self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.model_checkpoint_dir / "jepa_best.pt"
            torch.save(self._build_checkpoint(), save_path)
            print(
                f"[CHECKPOINT] val_loss {previous:.6f} → {self.best_val_loss:.6f} "
                f"— saved best to {save_path}"
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            train_logs = self.train_one_epoch()
            val_logs = self.validate_one_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            logs = {**train_logs, **val_logs, "ema_m": self._current_ema_momentum()}
            self.loss_history.append({"epoch": epoch + 1, **logs})
            self._save_loss_history()
            self._maybe_save_best(val_logs)

            msg = f"epoch={epoch + 1}/{self.num_epochs}"
            for key, value in logs.items():
                msg += f" {key}={value:.6f}"
            print(msg)

        self._save_checkpoint()
