import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from pathlib import Path
from typing import Optional, Tuple

from eeg_vae import EEGVAE
from models.eegvae_probe_head import EEGVAEProbeHead


# ── Checkpoint introspection helpers ─────────────────────────────────────────

def _detect_eegvae_config(state_dict: dict) -> dict:
    """
    Inspect a checkpoint state_dict and return model_config overrides inferred
    from the weight shapes.  Works for Phase 1 (KL / VQ) and Phase 2 (JEPA).

    Returns a dict with keys:
        is_jepa         (bool)
        model_type      ('kl' | 'vq')
        ch              (int | None)
        vq_n_quantizers (int)   — 0 = single-VQ, >0 = RVQ
        sequence_block  ('mamba' | 'attention')
    """
    is_jepa = 'mask_token' in state_dict or any(
        k.startswith('predictor.') for k in state_dict
    )

    # Key prefixes differ between Phase 1 (autoencoder.*) and Phase 2 (context_encoder.*)
    ae = 'context_encoder.' if is_jepa else 'autoencoder.'

    # ch: inferred from encoder conv_in shape [ch, adapted_channels, 3]
    w = state_dict.get(f'{ae}encoder.conv_in.weight')
    ch = int(w.shape[0]) if w is not None else None

    # sequence_block: mamba if Mamba-specific keys are present
    has_mamba = any(f'{ae}encoder.mid.attn_1.mamba.' in k for k in state_dict)
    sequence_block = 'mamba' if has_mamba else 'attention'

    # model_type: vq if quantize buffers are present
    has_vq = (
        f'{ae}quantize.embedding' in state_dict
        or f'{ae}quantize.quantizers.0.embedding' in state_dict
    )
    model_type = 'vq' if has_vq else 'kl'

    # n_quantizers: count RVQ stages (0 = standard single-VQ)
    n_q = 0
    if has_vq:
        while f'{ae}quantize.quantizers.{n_q}.embedding' in state_dict:
            n_q += 1

    return dict(
        is_jepa=is_jepa,
        model_type=model_type,
        ch=ch,
        vq_n_quantizers=n_q,
        sequence_block=sequence_block,
    )


def _remap_jepa_to_eegvae(state_dict: dict) -> dict:
    """
    Remap JEPA context_encoder keys to EEGVAE keys so the backbone can be
    loaded directly.

        context_encoder.channel_adaptor.* → channel_adaptor.*
        context_encoder.encoder.*         → autoencoder.encoder.*
        context_encoder.quant_conv.*      → autoencoder.quant_conv.*
        context_encoder.quantize.*        → autoencoder.quantize.*

    target_encoder.*, predictor.*, mask_token are discarded (not needed for
    downstream inference).
    """
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith('context_encoder.channel_adaptor.'):
            remapped[k[len('context_encoder.'):]] = v
        elif k.startswith('context_encoder.encoder.'):
            remapped['autoencoder.' + k[len('context_encoder.'):]] = v
        elif k.startswith('context_encoder.quant_conv.'):
            remapped['autoencoder.' + k[len('context_encoder.'):]] = v
        elif k.startswith('context_encoder.post_quant_conv.'):
            remapped['autoencoder.' + k[len('context_encoder.'):]] = v
        elif k.startswith('context_encoder.quantize.'):
            remapped['autoencoder.' + k[len('context_encoder.'):]] = v
    return remapped


class Classifier(nn.Module):
    def __init__(
        self,
        task: str = "classification",
        n_classes: int = 2,
        output_dim: int = 1,
        pooling: str = "attn",
        n_layer: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.task = task
        self.pooling = pooling.lower()
        self.n_layer = int(n_layer)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.norm = norm.lower()

        out_dim = int(n_classes) if task == "classification" else int(output_dim)
        if self.pooling == "attn":
            self.attn = nn.Sequential(nn.LazyLinear(1), nn.Softmax(dim=1))
        elif self.pooling in {"flatten", "mean", "max", "mean_max"}:
            self.attn = None
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.mlp = self._build(out_dim)

    def _build(self, out_dim: int):
        layers = []

        if self.n_layer <= 1:
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LazyLinear(out_dim))
            return nn.Sequential(*layers)

        layers.append(nn.Dropout(self.dropout))
        layers.append(nn.LazyLinear(self.hidden_dim))
        self._add_norm_activation_dropout(layers)

        for _ in range(self.n_layer - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self._add_norm_activation_dropout(layers)

        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _add_norm_activation_dropout(self, layers):
        if self.norm == "batchnorm":
            layers.append(nn.BatchNorm1d(self.hidden_dim))
        elif self.norm == "layernorm":
            layers.append(nn.LayerNorm(self.hidden_dim))
        elif self.norm == "none":
            pass
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        layers.append(nn.GELU())
        layers.append(nn.Dropout(self.dropout))

    def _as_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if x.dim() == 2:
            return x.unsqueeze(1), x.shape[-1]

        if x.dim() == 3:
            if x.shape[1] <= 512 and x.shape[2] > 512:
                return x.transpose(1, 2).contiguous(), x.shape[1]
            return x.contiguous(), x.shape[2]

        if x.dim() == 4:
            b, c, l, d = x.shape
            return x.reshape(b, c * l, d).contiguous(), d

        raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooling == "flatten":
            b, n, d = tokens.shape
            return tokens.reshape(b, n * d)

        if self.pooling == "mean":
            return tokens.mean(dim=1)

        if self.pooling == "max":
            return tokens.amax(dim=1)

        if self.pooling == "mean_max":
            return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)

        if self.pooling == "attn":
            weights = self.attn(tokens)
            return (tokens * weights).sum(dim=1)

        raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        tokens, _ = self._as_tokens(feats)
        x = self._pool_tokens(tokens)
        x = self.mlp(x)

        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x


class DownstreamEEGVAE(nn.Module):
    SUPPORTED_MODES = {
        "linear_probing",
        "finetuning",
        "channel_training",
        "full_training",
    }

    def __init__(
        self,
        model_config: dict,
        head_config: Optional[dict] = None,
        n_classes: int = 2,
        task: str = "classification",
        output_dim: int = 1,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: Optional[bool] = None,
        mode: str = "linear_probing",
        strict: bool = False,
    ):
        super().__init__()
        head_config = {} if head_config is None else dict(head_config)
        self.mode = self._normalize_mode(mode)

        # Auto-detect architecture from checkpoint so the yaml model section
        # never needs to be manually kept in sync with the checkpoint.
        self._is_jepa = False
        if checkpoint_path is not None and self.mode != "full_training":
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd   = ckpt.get("model_state_dict", ckpt)
            info = _detect_eegvae_config(sd)

            self._is_jepa = info["is_jepa"]
            model_config  = dict(model_config)
            if info["ch"] is not None:
                model_config["ch"] = info["ch"]
            model_config["model_type"]     = info["model_type"]
            model_config["sequence_block"] = info["sequence_block"]
            if info["vq_n_quantizers"] > 0:
                model_config["vq_n_quantizers"] = info["vq_n_quantizers"]

            src = "JEPA Phase 2" if info["is_jepa"] else f"Phase 1 ({info['model_type'].upper()})"
            print(
                f"[DOWNSTREAM] Detected {src} checkpoint  "
                f"ch={info['ch']}  sequence_block={info['sequence_block']}  "
                f"model_type={info['model_type']}  vq_n_quantizers={info['vq_n_quantizers']}"
            )

        self.backbone = EEGVAE(**model_config)

        # The head is always EEGVAEProbeHead — it is the only one tuned for
        # EEGVAE's (B, embed_dim, T_tokens) latent. The yaml head section is
        # ignored except for optional probe hyperparameters.
        head_config.pop("type", None)
        head_config.pop("probe_max_norm", None)
        head_config.pop("classifier_max_norm", None)
        self.classifier = EEGVAEProbeHead(
            task=task,
            n_classes=n_classes,
            output_dim=output_dim,
            **head_config,
        )

        if checkpoint_path is not None and self.mode != "full_training":
            self._load_encoder(checkpoint_path, strict=strict, is_jepa=self._is_jepa)
        elif checkpoint_path is not None and self.mode == "full_training":
            print("[DOWNSTREAM] mode=full_training: skipping checkpoint loading")

        self.freeze_encoder = self.mode == "linear_probing" if freeze_encoder is None else freeze_encoder
        self._configure_trainable_parameters()

    def _normalize_mode(self, mode: str) -> str:
        mode = str(mode).lower()
        aliases = {
            "linear_probe": "linear_probing",
            "linear-probing": "linear_probing",
            "fine_tuning": "finetuning",
            "fine-tuning": "finetuning",
            "channel-training": "channel_training",
            "channel_tuning": "channel_training",
            "full-training": "full_training",
            "from_scratch": "full_training",
        }
        mode = aliases.get(mode, mode)
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unknown downstream mode: {mode}. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        return mode

    def _configure_trainable_parameters(self):
        for param in self.parameters():
            param.requires_grad = True

        if self.mode == "linear_probing":
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif self.mode == "channel_training":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.channel_adaptor.parameters():
                param.requires_grad = True

        elif self.mode in {"finetuning", "full_training"}:
            pass

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        )
        total = sum(
            p.numel()
            for p in self.parameters()
            if not isinstance(p, UninitializedParameter)
        )
        print(f"[DOWNSTREAM] mode={self.mode} | trainable params: {trainable:,}/{total:,}")

    def _load_encoder(self, checkpoint_path: str, strict: bool = False, is_jepa: bool = False):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        if is_jepa:
            state_dict = _remap_jepa_to_eegvae(state_dict)
            print(f"[DOWNSTREAM] Remapped JEPA encoder keys → EEGVAE namespace ({len(state_dict)} keys)")

        if strict:
            return self.backbone.load_state_dict(state_dict, strict=True)

        current_state = self.backbone.state_dict()
        compatible_state = {}
        skipped = []

        for key, value in state_dict.items():
            if key not in current_state:
                skipped.append((key, tuple(value.shape), None))
                continue

            if current_state[key].shape != value.shape:
                skipped.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue

            compatible_state[key] = value

        if skipped:
            print(f"[CHECKPOINT] Skipping {len(skipped)} incompatible keys:")
            for key, old_shape, new_shape in skipped[:20]:
                print(f"  {key}: checkpoint={old_shape} model={new_shape}")
            if len(skipped) > 20:
                print(f"  ... {len(skipped) - 20} more")

        return self.backbone.load_state_dict(compatible_state, strict=False)

    def forward(self, x):
        if self.mode == "linear_probing":
            with torch.no_grad():
                posterior = self.backbone.encode(x)
                feats = posterior.mode()
        else:
            posterior = self.backbone.encode(x)
            feats = posterior.mode()

        return self.classifier(feats)
