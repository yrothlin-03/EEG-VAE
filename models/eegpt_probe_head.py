from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter


class LinearWithConstraint(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        max_norm: Optional[float] = None,
    ):
        self.max_norm = max_norm
        super().__init__(int(in_features), int(out_features), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data,
                p=2,
                dim=0,
                maxnorm=float(self.max_norm),
            )
        return super().forward(x)


class LazyLinearWithConstraint(nn.Module):
    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        max_norm: Optional[float] = None,
    ):
        super().__init__()
        self.out_features = int(out_features)
        self.bias = bool(bias)
        self.max_norm = max_norm
        self.linear = nn.LazyLinear(self.out_features, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.linear.weight, UninitializedParameter):
            self.linear.initialize_parameters(x)

        if self.max_norm is not None:
            self.linear.weight.data = torch.renorm(
                self.linear.weight.data,
                p=2,
                dim=0,
                maxnorm=float(self.max_norm),
            )
        return F.linear(x, self.linear.weight, self.linear.bias)


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm: float = 1.0, do_weight_norm: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)
        self.do_weight_norm = bool(do_weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class EEGPTLinearProbeHead(nn.Module):
    """Linear-probe head mirroring the BCIC2A EEGPT script.

    Expected EEGPT features are [B, N, E, D]. The reference script flattens
    the last two dimensions, projects each token to 16 dims, then flattens
    all tokens before the final classifier. This implementation keeps that
    structure but uses lazy constrained linears so it can attach to the
    existing downstream interface and to feature tensors with nearby shapes
    from other backbones.
    """

    def __init__(
        self,
        task: str,
        n_classes: int,
        hidden_dim: int = 16,
        dropout: float = 0.5,
        n_layer: int = 2,
        norm: str = "none",
        pooling: str = "eegpt_flatten",
        output_dim: int = 1,
        probe_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
        **_,
    ):
        super().__init__()
        self.task = task
        self.n_classes = int(n_classes)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        out_dim = self.n_classes if task == "classification" else self.output_dim
        self.pooling = str(pooling).lower()
        self.probe = LazyLinearWithConstraint(
            self.hidden_dim,
            max_norm=probe_max_norm,
        )
        self.classifier = LazyLinearWithConstraint(
            out_dim,
            max_norm=classifier_max_norm,
        )

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.flatten(2)

        if x.ndim == 3:
            return x

        if x.ndim == 2:
            return x.unsqueeze(1)

        raise ValueError(f"Unsupported feature shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tokens(x)
        x = self.probe(self.dropout(x))
        x = x.flatten(1)
        x = self.classifier(x)

        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x
