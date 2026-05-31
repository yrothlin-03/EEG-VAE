from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .backbone_wrapper import BackboneWrapper
from .cbramod_probe_head import CBRAMODProbeHead
from .eegmamba_probe_head import EEGMambaProbeHead
from .eegpt_probe_head import Conv1dWithConstraint, EEGPTLinearProbeHead
from .femba_probe_head import FEMBAProbeHead
from .head_wrapper import HeadWrapper
from .headV3 import HeadV3
from .luna_probe_head import LunaProbeHead
from .reve_probe_head import REVEProbeHead

class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        dataset_name: Optional[str],
        data_mode: str,
        out_path: Optional[Path],
        checkpoint_path: Optional[Path],
        model_config: dict,
        head_config: dict,
        task: str,
        n_classes: int,
        mode: str = "finetuning",
        logger: Optional[object] = None,
        patch_size: Optional[int] = None,
    ):
        super().__init__()

        self.logger = logger
        self.model_name = model_name
        self.out_path = out_path
        self.patch_size = patch_size
        self.task = task
        self.n_classes = n_classes
        self.mode = mode
        self.input_adapter = None
        self.input_target_length = None

        if self.model_name == "EEGPT" and model_config.get("use_eegpt_bcic2a_probe_adapter", False):
            self.input_adapter = Conv1dWithConstraint(
                int(model_config.get("input_adapter_in_channels", 22)),
                int(model_config.get("input_adapter_out_channels", 19)),
                kernel_size=1,
                max_norm=float(model_config.get("input_adapter_max_norm", 1.0)),
            )
            self.input_target_length = int(model_config.get("input_target_length", 1024))

        self.backbone = BackboneWrapper(
            model_name=model_name,
            model_config=model_config,
            patch_size=patch_size,
            dataset_name=dataset_name,
            mode=data_mode,
        )

        if mode in ("linear_probing", "finetuning"):
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided for linear_probing/finetuning.")
            if self.model_name != "REVE":
                self.load_checkpoints(checkpoint_path, backbone_only=True)

            if self.logger is not None:
                self.logger.info(f"Backbone initialized from checkpoint: {checkpoint_path}")

            if mode == "linear_probing":
                self.freeze_backbone()
                if self.logger is not None:
                    self.logger.info("Backbone frozen for linear probing.")

        # Auto-select probe head from model_name so the downstream config does
        # not need to carry a `head:` section. Each backbone has a dedicated
        # head file tuned for its output feature shape.
        hc = dict(head_config) if head_config else {}
        self.head = self._build_head(task, n_classes, model_config, hc)

    #     self.head = HeadV3(
    #     task=task,
    #     n_classes=n_classes,
    #     pooling=hc.get("pooling", "attn"),
    #     hidden_dim=hc.get("hidden_dim", 256),
    #     dropout=hc.get("dropout", 0.5),
    #     n_layer=hc.get("n_layer", 2),
    #     output_dim=hc.get("output_dim", 1),
    #     norm=hc.get("norm", "layernorm"),
    # )

        self._log_every = 1000
        self._cpt = 0


    def _build_head(self, task: str, n_classes: int, model_config: dict, hc: dict):
        """Pick the right probe head for the backbone."""
        name = self.model_name.upper()

        if name == "LUNA":
            return LunaProbeHead(
                task=task,
                n_classes=n_classes,
                embed_dim=model_config.get("embed_dim", hc.get("embed_dim", 64)),
                num_queries=model_config.get("num_queries", hc.get("num_queries", 4)),
                num_heads=model_config.get("num_heads", hc.get("num_heads", 2)),
                dropout=hc.get("dropout", 0.15),
                output_dim=hc.get("output_dim", 1),
            )

        if name == "EEGPT":
            return EEGPTLinearProbeHead(
                task=task,
                n_classes=n_classes,
                hidden_dim=hc.get("hidden_dim", 16),
                dropout=hc.get("dropout", 0.5),
                output_dim=hc.get("output_dim", 1),
                probe_max_norm=hc.get("probe_max_norm", 1.0),
                classifier_max_norm=hc.get("classifier_max_norm", 0.25),
            )

        if name == "CBRAMOD":
            return CBRAMODProbeHead(task=task, n_classes=n_classes,
                                     output_dim=hc.get("output_dim", 1))

        if name == "EEGMAMBA":
            return EEGMambaProbeHead(task=task, n_classes=n_classes,
                                      output_dim=hc.get("output_dim", 1))

        if name == "FEMBA":
            return FEMBAProbeHead(task=task, n_classes=n_classes,
                                   output_dim=hc.get("output_dim", 1))

        if name == "REVE":
            return REVEProbeHead(task=task, n_classes=n_classes,
                                  output_dim=hc.get("output_dim", 1))

        # Fallback for any other backbone
        return HeadWrapper(
            task=task,
            n_classes=n_classes,
            hidden_dim=hc.get("hidden_dim", 256),
            dropout=hc.get("dropout", 0.5),
            n_layer=hc.get("n_layer", 1),
            norm=hc.get("norm", "layernorm"),
        )

    def print_model_info(self, checkpoint_path: Optional[Path] = None):
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_backbone = not any(p.requires_grad for p in self.backbone.parameters())

        print("----" * 20)
        print("MODEL INFORMATIONS \n")
        print(f"Model name       : {self.model_name}")
        print(f"Task             : {self.task}")
        print(f"Mode             : {self.mode}")
        print(f"Num classes      : {self.n_classes}")
        print(f"Patch size       : {self.patch_size}")
        print(f"Checkpoint       : {checkpoint_path}")
        print(f"Backbone params  : {backbone_params:,}")
        print(f"Head params      : {head_params:,}")
        print(f"Total params     : {total_params:,}")
        print(f"Trainable params : {trainable_params:,}")
        print(f"Frozen backbone  : {frozen_backbone}")
        print("----" * 20)

        if self.logger is not None:
            self.logger.info("MODEL INFORMATIONS")
            self.logger.info(f"Model name       : {self.model_name}")
            self.logger.info(f"Task             : {self.task}")
            self.logger.info(f"Mode             : {self.mode}")
            self.logger.info(f"Num classes      : {self.n_classes}")
            self.logger.info(f"Patch size       : {self.patch_size}")
            self.logger.info(f"Checkpoint       : {checkpoint_path}")
            self.logger.info(f"Backbone params  : {backbone_params:,}")
            self.logger.info(f"Head params      : {head_params:,}")
            self.logger.info(f"Total params     : {total_params:,}")
            self.logger.info(f"Trainable params : {trainable_params:,}")
            self.logger.info(f"Frozen backbone  : {frozen_backbone}")


    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()

    def load_checkpoints(self, checkpoint_path: Path, backbone_only: bool = False, strict: bool = False):
        if backbone_only:
            return self.backbone.load_checkpoints(checkpoint_path, strict=strict)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        print(f"[MODEL LOAD] Missing: {len(missing)} \n {missing}\n Unexpected: {len(unexpected)} \n {unexpected}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def save_checkpoints(
        self,
        checkpoint_path: Optional[Path] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: Optional[int] = None,
        extra: Optional[dict] = None,
    ):
        if checkpoint_path is None:
            if self.out_path is None:
                return
            checkpoint_path = Path(self.out_path)

        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.suffix == "":
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_path / "checkpoint.pt"
        else:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "model_name": self.model_name,
            "task": self.task,
            "n_classes": self.n_classes,
            "mode": self.mode,
            "patch_size": self.patch_size,
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if extra is not None:
            state["extra"] = extra

        torch.save(state, checkpoint_path)

        if self.logger is not None:
            self.logger.info(f"Checkpoint saved to: {checkpoint_path}")

    def _save_ckpt(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: Optional[int] = None,
    ):
        return self.save_checkpoints(
            checkpoint_path=self.out_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )

    def forward_features(self, x: torch.Tensor, seed: Optional[int] = None, ch_names=None) -> torch.Tensor:
        if self.input_target_length is not None and x.shape[-1] != self.input_target_length:
            x = F.interpolate(
                x,
                size=self.input_target_length,
                mode="linear",
                align_corners=False,
            )
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        return self.backbone(x, return_features=True, seed=seed, ch_names=ch_names)

    def forward(self, x: torch.Tensor, return_features: bool = False, seed: Optional[int] = None, ch_names=None):
        feats = self.forward_features(x, seed=seed, ch_names=ch_names)

        self._cpt += 1
        if self.logger is not None and self._cpt % self._log_every == 0:
            self.logger.info(
                f"Features shape: {tuple(feats.shape)} | mean: {feats.mean().item():.4f} | std: {feats.std().item():.4f}"
            )

        if return_features:
            return feats

        return self.head(feats)