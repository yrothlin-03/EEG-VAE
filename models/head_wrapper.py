from typing import Literal
import torch
import torch.nn as nn


class HeadWrapper(nn.Module):
    def __init__(
        self,
        task: str,
        n_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        n_layer: int = 2,
        norm: Literal["layernorm", "batchnorm", "none"] = "layernorm",
    ):
        super().__init__()
        self.task = task
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_layer = n_layer
        self.norm = norm
        self.head = self._make_layers()
        self._built_dim = None

    def _make_layers(self):
        layers = []

        if self.n_layer <= 1:
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LazyLinear(self.n_classes))
        else:
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LazyLinear(self.hidden_dim))

            if self.norm == "layernorm":
                layers.append(nn.LayerNorm(self.hidden_dim))
            elif self.norm == "batchnorm":
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            elif self.norm != "none":
                raise ValueError(f"Invalid norm: {self.norm}")

            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))

            for _ in range(self.n_layer - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

                if self.norm == "layernorm":
                    layers.append(nn.LayerNorm(self.hidden_dim))
                elif self.norm == "batchnorm":
                    layers.append(nn.BatchNorm1d(self.hidden_dim))
                elif self.norm != "none":
                    raise ValueError(f"Invalid norm: {self.norm}")

                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout))

            layers.append(nn.Linear(self.hidden_dim, self.n_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 4:
            B = x.size(0)
            x = x.reshape(B, -1)

        elif x.ndim == 3:
            B = x.size(0)
            x = x.reshape(B, -1)

        elif x.ndim == 2:
            pass

        else:
            raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

        if self._built_dim is not None and self._built_dim != x.shape[-1]:
            raise ValueError(
                f"HeadWrapper was initialized for {self._built_dim} input features, "
                f"but received {x.shape[-1]}."
            )

        out = self.head(x)
        self._built_dim = x.shape[-1]
        return out
