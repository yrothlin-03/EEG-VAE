import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .vector_quantizer import VectorQuantizerEMA, ResidualVectorQuantizer


__all__ = ["AutoencoderVQ"]


class VQPosterior:
    """
    Duck-typed to match DiagonalGaussianDistribution so the training pipeline
    (PretrainingLoss, VAEGANLoss, Trainer) works without any modification.

    .kl()      → returns the commitment loss (scalar tensor).
                 In PretrainingLoss: kl_weight * mean(posterior.kl()) acts as the
                 commitment loss scale. Set kl_weight: 1.0 in the config for VQ
                 (commitment_cost inside the VQ layer already scales the raw MSE).
    .sample() / .mode() → both return the quantized latent z_q.
    .loss_name → used by PretrainingLoss to name the log key ("commitment_loss").
    """

    loss_name = "commitment_loss"

    def __init__(self, z_quantized, commitment_loss, indices, perplexity):
        self.z_quantized = z_quantized
        self.commitment_loss = commitment_loss   # scalar tensor, gradient-connected
        self.indices = indices                   # (B, T) int64 — discrete codes
        self.perplexity = perplexity             # scalar — effective codebook usage

    # ---- latent access -------------------------------------------------------

    def sample(self):
        return self.z_quantized

    def mode(self):
        return self.z_quantized

    # ---- KL-compatible interface ---------------------------------------------

    def kl(self, _other=None, _dims=None):
        # torch.mean(scalar) = scalar, so kl_loss(posterior) works as-is.
        return self.commitment_loss


class AutoencoderVQ(nn.Module):
    """
    Autoencoder with a vector-quantized bottleneck.
    Drop-in replacement for AutoencoderKL — same forward / encode / decode /
    reconstruct / get_last_layer interface.

    The only behavioural difference visible to the trainer is that encode()
    returns a VQPosterior instead of a DiagonalGaussianDistribution; both
    expose .sample(), .mode(), and .kl().
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        z_channels=4,
        embed_dim=None,
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        resolution=None,
        dropout=0.0,
        resamp_with_conv=True,
        tanh_out=False,
        use_checkpoint=False,
        sequence_block="attention",
        criss_cross_patch_size=200,
        # VQ-specific
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
        n_quantizers=1,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        embed_dim = z_channels if embed_dim is None else embed_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.embed_dim = embed_dim

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=z_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            double_z=False,   # VQ does not split into mean/logvar
            use_checkpoint=use_checkpoint,
            sequence_block=sequence_block,
            criss_cross_patch_size=criss_cross_patch_size,
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            z_channels=embed_dim,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            tanh_out=tanh_out,
            use_checkpoint=use_checkpoint,
            sequence_block=sequence_block,
        )

        # z_channels → embed_dim  (no factor-2 doubling needed for VQ)
        self.quant_conv = nn.Conv1d(z_channels, embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        if n_quantizers > 1:
            self.quantize = ResidualVectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embed_dim,
                n_quantizers=n_quantizers,
                commitment_cost=commitment_cost,
                decay=decay,
            )
        else:
            self.quantize = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embed_dim,
                commitment_cost=commitment_cost,
                decay=decay,
            )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        z_q, commitment_loss, indices, perplexity = self.quantize(h)
        return VQPosterior(z_q, commitment_loss, indices, perplexity)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True, return_posterior=False):
        posterior = self.encode(x)
        z = posterior.sample()   # always z_q — no stochasticity in VQ
        dec = self.decode(z)

        if return_posterior:
            return dec, posterior
        return dec

    def reconstruct(self, x):
        return self.forward(x, sample_posterior=False)

    def get_last_layer(self):
        return self.decoder.conv_out.weight
