from typing import Optional

import torch
import torch.nn as nn


class EEGVAEFlattenProbeHead(nn.Module):
    """Minimal *test* probe for EEGVAE latent tensors: flatten then project.

    Unlike :class:`EEGVAEProbeHead` (conv projection + temporal conv + pooling
    + MLP), this head keeps no spatio-temporal processing. It simply flattens
    the frozen latent map ``[B, C_latent, T_latent]`` into a single vector and
    applies a linear projection to the output dimension. This is the canonical
    "flatten + linear" probe, useful as a lower-capacity reference point.

    With ``n_layer == 1`` (default) the head is a pure linear classifier
    (logistic regression) on the flattened latent. With ``n_layer >= 2`` a
    single hidden projection of width ``hidden_dim`` is inserted before the
    output layer.

    Notes
    -----
    * The input size after flattening is ``C_latent * T_latent`` and therefore
      depends on the downstream window length. The first linear layer is a
      ``LazyLinear`` so it is materialized at the first forward pass.
    * Because the flattened dimension scales with the window length, this head
      can have many more parameters than the pooling-based probe on long
      windows (e.g. 30 s Sleep-EDF / TUAB) and is more prone to overfitting on
      small datasets. It is intended for ablation/diagnostic use.
    """

    def __init__(
        self,
        task: str = "classification",
        n_classes: int = 2,
        output_dim: int = 1,
        n_layer: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__()
        self.task = task
        self.n_layer = int(n_layer)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        out_dim = int(n_classes) if task == "classification" else int(output_dim)
        self.mlp = self._build_mlp(out_dim)

    def _build_mlp(self, out_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Dropout(self.dropout)]

        if self.n_layer <= 1:
            # Pure linear probe: flatten -> Linear(out_dim)
            layers.append(nn.LazyLinear(out_dim))
            return nn.Sequential(*layers)

        # flatten -> Linear(hidden) -> GELU -> Dropout -> ... -> Linear(out_dim)
        layers.append(nn.LazyLinear(self.hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(self.dropout))

        for _ in range(self.n_layer - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(self.hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _to_latent_map(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the latent to ``[B, C, T]`` before flattening."""
        if x.ndim == 2:
            return x.unsqueeze(-1)
        if x.ndim == 3:
            return x.contiguous()
        if x.ndim == 4:
            b, c, h, w = x.shape
            return x.reshape(b, c, h * w).contiguous()
        raise ValueError(f"Unsupported EEGVAE feature shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_latent_map(x)   # [B, C_latent, T_latent]
        x = x.flatten(1)             # [B, C_latent * T_latent]
        x = self.mlp(x)              # [B, out_dim]

        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x
