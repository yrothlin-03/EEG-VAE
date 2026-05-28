import torch
import torch.nn as nn


class LunaProbeHead(nn.Module):
    """LUNA downstream head inspired by BioFoundation's classifier.

    LUNA returns token features shaped [B, num_patches, num_queries * embed_dim].
    BioFoundation classifies those features with a learned aggregate query that
    attends over patch tokens, followed by an MLP classifier.
    """

    def __init__(
        self,
        task: str,
        n_classes: int,
        embed_dim: int = 64,
        num_queries: int = 4,
        num_heads: int = 2,
        dropout: float = 0.15,
        output_dim: int = 1,
        **_,
    ):
        super().__init__()
        self.task = task
        self.n_classes = int(n_classes)
        self.output_dim = int(output_dim)
        self.feature_dim = int(embed_dim) * int(num_queries)
        out_dim = self.n_classes if task == "classification" else self.output_dim

        self.learned_agg = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        self.decoder_attn = nn.MultiheadAttention(
            self.feature_dim,
            int(num_heads),
            batch_first=True,
            dropout=float(dropout),
        )
        self.decoder_ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim * 4, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.flatten(1, 2)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(f"Unsupported LUNA feature shape: {tuple(x.shape)}")

        if x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"LunaProbeHead expected feature dim {self.feature_dim}, "
                f"but received {x.shape[-1]}. Check LUNA embed_dim/num_queries."
            )

        query = self.learned_agg.repeat(x.shape[0], 1, 1)
        x = self.decoder_attn(query=query, key=x, value=x)[0]
        x = self.decoder_ffn(x[:, 0, :])

        if self.task == "regression" and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x
