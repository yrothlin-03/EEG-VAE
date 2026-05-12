import torch.nn as nn

from .decoder import Decoder
from .distributions import DiagonalGaussianDistribution
from .encoder import Encoder


__all__ = ["AutoencoderKL"]


class AutoencoderKL(nn.Module):
    """
    Variational autoencoder with a diagonal Gaussian latent distribution.
    Inputs and outputs are 1D signals with shape (B, C, T).
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
            double_z=True,
            use_checkpoint=use_checkpoint,
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
        )

        self.quant_conv = nn.Conv1d(2 * z_channels, 2 * embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

    def encode(self, x):
        moments = self.encoder(x)
        moments = self.quant_conv(moments)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True, return_posterior=False):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)

        if return_posterior:
            return dec, posterior
        return dec

    def reconstruct(self, x):
        return self.forward(x, sample_posterior=False)

    def get_last_layer(self):
        return self.decoder.conv_out.weight



