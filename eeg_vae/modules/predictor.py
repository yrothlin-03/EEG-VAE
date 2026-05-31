import torch
import torch.nn as nn


__all__ = ["EEGPredictor"]


class _TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class EEGPredictor(nn.Module):
    """
    Narrow transformer predictor for EEG-JEPA (Phase 2).

    Receives the full latent sequence (B, D, T_tokens) where target positions have
    been replaced by a learnable mask token, then predicts the representations at
    those positions.

    The "narrow predictor" pattern (I-JEPA): D is projected up to predictor_embed_dim
    internally, keeping the predictor smaller than the main encoder while still
    having capacity for temporal reasoning.

    Args:
        embed_dim:            Dimension of z_q tokens (must match VQ embed_dim).
        predictor_embed_dim:  Internal width of the predictor transformer.
        num_heads:            Attention heads (must divide predictor_embed_dim).
        num_layers:           Number of transformer blocks.
        max_seq_len:          Maximum T_tokens supported by the positional embedding.
        dropout:              Dropout in attention and MLP.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        predictor_embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        self.input_proj = nn.Linear(embed_dim, predictor_embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, predictor_embed_dim)

        self.blocks = nn.ModuleList([
            _TransformerBlock(predictor_embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(predictor_embed_dim)
        self.output_proj = nn.Linear(predictor_embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z, target_pos):
        """
        Args:
            z:          (B, D, T_tokens) — full sequence with mask tokens at target positions.
            target_pos: (n_targets,)     — LongTensor of token indices to predict.

        Returns:
            (B, D, n_targets) — predicted representations at target positions.
        """
        B, D, T = z.shape

        h = z.permute(0, 2, 1)                              # (B, T, D)
        h = self.input_proj(h)                              # (B, T, predictor_dim)
        h = h + self.pos_embed(torch.arange(T, device=z.device)).unsqueeze(0)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        h = h[:, target_pos, :]                             # (B, n_targets, predictor_dim)
        pred = self.output_proj(h)                          # (B, n_targets, D)

        return pred.permute(0, 2, 1)                        # (B, D, n_targets)
