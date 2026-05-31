import torch
import torch.nn as nn

from .eegpt_probe_head import LazyLinearWithConstraint


class CBRAMODProbeHead(nn.Module):
    """Linear-probe head for CBRAMOD.

    CBRAMOD outputs token features shaped [B, C, L, D] where C is the number
    of EEG channels, L the number of temporal patches, and D the embedding
    dimension. We flatten the channel/temporal dims, project each token to a
    small hidden width, then flatten all tokens before the final classifier.

    Inspired by EEGPTLinearProbeHead with hyperparameters tuned for CBRAMOD's
    larger embedding (200) and typical token count.
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
        if x.ndim == 4:                       # (B, C, L, D)
            b, c, l, d = x.shape
            return x.reshape(b, c * l, d)
        if x.ndim == 3:                       # (B, N, D)
            return x
        if x.ndim == 2:                       # (B, D)
            return x.unsqueeze(1)
        raise ValueError(f"Unsupported CBRAMOD feature shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_tokens(x)
        x = self.probe(self.dropout(x))       # (B, N, hidden_dim)
        x = x.flatten(1)                      # (B, N * hidden_dim)
        x = self.classifier(x)
        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x
