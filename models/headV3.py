import torch
import torch.nn as nn
from typing import Optional, Tuple


class HeadV3(nn.Module):
    def __init__(
        self,
        task: str = "classification",
        n_classes: int = 2,
        output_dim: int = 1,
        pooling: str = "attn",
        n_layer: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.task = task
        self.pooling = pooling.lower()
        self.n_layer = int(n_layer)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.norm = norm.lower()

        self.out_dim = int(n_classes) if task == "classification" else int(output_dim)

        self.mlp = None
        self._built_for_flat_dim: Optional[int] = None

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

    def _build(self, flat_dim: int, device: torch.device):
        layers = []

        if self.n_layer <= 1:
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(flat_dim, self.out_dim))
        else:
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(flat_dim, self.hidden_dim))

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

            for _ in range(self.n_layer - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

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

            layers.append(nn.Linear(self.hidden_dim, self.out_dim))

        self.mlp = nn.Sequential(*layers).to(device)
        self._built_for_flat_dim = int(flat_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        tokens, d = self._as_tokens(feats)
        b, n, _ = tokens.shape
        flat_dim = n * d

        if self._built_for_flat_dim is None or self._built_for_flat_dim != flat_dim:
            self._build(flat_dim, feats.device)

        x = tokens.reshape(b, flat_dim)
        x = self.mlp(x)

        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x