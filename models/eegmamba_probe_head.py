import torch
import torch.nn as nn

from .eegpt_probe_head import LazyLinearWithConstraint


class EEGMambaProbeHead(nn.Module):
    """Linear-probe head for EEGMAMBA.

    EEGMAMBA returns patch-token features with shape similar to CBRAMOD:
    [B, C, L, D] for channel × temporal patch tokens. The probe flattens
    channels and time, projects each token to a small hidden width, then
    flattens all tokens before the final classifier.
    """

    def __init__(
        self,
        task: str,
        n_classes: int,
        hidden_dim: int = 32,
        dropout: float = 0.5,
        output_dim: int = 1,
        probe_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
        **_,
    ):
        super().__init__()
        self.task = task
        self.n_classes = int(n_classes)
        self.output_dim = int(output_dim)
        self.dropout = nn.Dropout(float(dropout))

        out_dim = self.n_classes if task == "classification" else self.output_dim
        self.probe = LazyLinearWithConstraint(int(hidden_dim), max_norm=probe_max_norm)
        self.classifier = LazyLinearWithConstraint(out_dim, max_norm=classifier_max_norm)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            b, c, l, d = x.shape
            return x.reshape(b, c * l, d)
        if x.ndim == 3:
            return x
        if x.ndim == 2:
            return x.unsqueeze(1)
        raise ValueError(f"Unsupported EEGMAMBA feature shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tokens(x)
        x = self.probe(self.dropout(x))
        x = x.flatten(1)
        x = self.classifier(x)
        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x
