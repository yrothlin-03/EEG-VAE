import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["VectorQuantizerEMA", "ResidualVectorQuantizer"]


class VectorQuantizerEMA(nn.Module):
    """
    EMA-updated vector quantization bottleneck (VQVAE-2 style).

    Operates on (B, D, T) tensors — each time step is quantized independently
    by finding its nearest entry in the codebook (num_embeddings, D).

    EMA updates only happen during training; the codebook is frozen at eval time.
    Gradients flow to the encoder via the straight-through estimator.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Codebook vectors — stored as buffers so they are part of state_dict
        # but not updated by the optimizer (updated by EMA instead).
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.ones(num_embeddings))
        self.register_buffer("embedding_avg", embedding.clone())

    def freeze_codebook(self):
        """Disable EMA codebook updates (e.g. during a second training phase)."""
        self._codebook_frozen = True

    def forward(self, z):
        """
        Args:
            z: (B, D, T) — encoder output after quant_conv projection.

        Returns:
            z_q:             (B, D, T) — quantized tensor (straight-through in backward).
            commitment_loss: scalar    — β * MSE(z_encoder, sg[z_codebook]).
            indices:         (B, T)   — codebook index per time step.
            perplexity:      scalar   — effective codebook utilisation.
        """
        B, D, T = z.shape

        # (B, D, T) → (B*T, D) for flat nearest-neighbour search
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)

        # Squared L2 distances to every codebook entry: (N, K)
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2.0 * z_flat @ self.embedding.T
        )

        indices = dist.argmin(dim=1)          # (N,) = (B*T,)
        z_q_flat = self.embedding[indices]    # (N, D)

        if self.training and not getattr(self, "_codebook_frozen", False):
            # Cast to fp32 for EMA: buffers are fp32 and z_flat may be fp16 under AMP.
            # Using fp16 here would silently corrupt the codebook if any value overflows.
            z_flat_fp32 = z_flat.detach().float()

            indices_onehot = F.one_hot(indices, self.num_embeddings).float()  # (N, K)
            cluster_size_new = indices_onehot.sum(dim=0)                       # (K,)
            embedding_sum_new = indices_onehot.T @ z_flat_fp32                 # (K, D)

            # Guard: skip the EMA update entirely if the batch contains NaN/Inf.
            # This can happen when AMP overflows before the GradScaler catches it —
            # the EMA runs during forward, so it is not protected by the scaler.
            if torch.isfinite(embedding_sum_new).all():
                self.cluster_size.mul_(self.decay).add_(cluster_size_new, alpha=1.0 - self.decay)
                self.embedding_avg.mul_(self.decay).add_(embedding_sum_new, alpha=1.0 - self.decay)

                # Laplace smoothing to keep every entry alive even if temporarily unused
                n = self.cluster_size.sum()
                smoothed = (
                    (self.cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps)
                    * n
                )
                self.embedding.copy_(self.embedding_avg / smoothed.unsqueeze(1))

        # Commitment loss: always in fp32 — z_flat may be fp16 under AMP, and squaring
        # fp16 values in MSE can overflow (fp16 max ≈ 65504).
        commitment_loss = self.commitment_cost * F.mse_loss(
            z_flat.float(), z_q_flat.detach().float()
        )

        # Straight-through estimator: keep the same dtype as the encoder output so
        # the decoder receives a tensor in the expected precision.
        z_q_flat = z_flat + (z_q_flat.to(z_flat.dtype) - z_flat).detach()

        # (N, D) → (B, T, D) → (B, D, T)
        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1).contiguous()
        indices = indices.view(B, T)

        # Perplexity ≈ effective number of codebook entries used per batch
        avg_probs = F.one_hot(indices, self.num_embeddings).float().view(-1, self.num_embeddings).mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, commitment_loss, indices, perplexity


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ): stacks n_quantizers VQ stages in series.

    Each stage quantizes the residual left by all previous stages:
        residual_0 = z_encoder
        z_q_i, residual_i = VQ_i(residual_{i-1})
        z_q_total = Σ z_q_i

    This gives an effective codebook of num_embeddings^n_quantizers combinations
    while keeping each individual lookup small (same num_embeddings).
    Used in EnCodec for audio; directly applicable to EEG latent codes.

    The STE gradient flows through every stage independently, so the encoder
    receives gradients from all n_quantizers commitment losses simultaneously.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        n_quantizers: int = 8,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay, eps)
            for _ in range(n_quantizers)
        ])

    def freeze_codebook(self):
        for q in self.quantizers:
            q.freeze_codebook()

    def forward(self, z):
        """
        Args:
            z: (B, D, T)

        Returns:
            z_q:             (B, D, T)           — sum of all stage quantizations.
            commitment_loss: scalar              — mean commitment loss across stages.
            indices:         (B, n_quantizers, T) — per-stage codebook indices.
            perplexity:      scalar              — mean perplexity across stages.
        """
        residual = z
        z_q_total = torch.zeros_like(z)
        total_commitment = torch.tensor(0.0, device=z.device, dtype=torch.float32)
        all_indices = []
        all_perplexities = []

        for quantizer in self.quantizers:
            z_q_i, commitment_i, indices_i, perplexity_i = quantizer(residual)
            # Next stage quantizes the residual; detach z_q_i so its STE
            # does not interfere with the next stage's input gradient.
            residual = residual - z_q_i.detach()
            z_q_total = z_q_total + z_q_i
            total_commitment = total_commitment + commitment_i
            all_indices.append(indices_i)       # (B, T)
            all_perplexities.append(perplexity_i)

        indices = torch.stack(all_indices, dim=1)              # (B, n_quantizers, T)
        perplexity = torch.stack(all_perplexities).mean()
        commitment_loss = total_commitment / self.n_quantizers  # normalise by stages

        return z_q_total, commitment_loss, indices, perplexity
