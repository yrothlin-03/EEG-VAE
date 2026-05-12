import torch
import torch.nn as nn

from .modules.autoencodeur_kl import AutoencoderKL
from .modules.channel_adaptor import ChannelAdaptor

__all__ = ["EEGVAE"]


class EEGVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        adapted_channels=None,
        adaptor_layers=2,
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

        adapted_channels = (
            in_channels if adapted_channels is None else adapted_channels
        )

        self.channel_adaptor = ChannelAdaptor(
            in_channels=in_channels,
            out_channels=adapted_channels,
            n_layers=adaptor_layers,
        )

        self.autoencoder = AutoencoderKL(
            in_channels=adapted_channels,
            out_channels=adapted_channels,
            z_channels=z_channels,
            embed_dim=embed_dim,
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

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def reconstruct(self, x):
        return self.autoencoder.reconstruct(x)

    def forward(self, x, sample_posterior=True, return_posterior=True):
        x = self.channel_adaptor(x)

        return self.autoencoder(
            x,
            sample_posterior=sample_posterior,
            return_posterior=return_posterior,
        )

    def get_last_layer(self):
        return self.autoencoder.get_last_layer()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EEGVAE(
        in_channels=16,
        adapted_channels=32,
        adaptor_layers=2,
        z_channels=4,
        ch=32,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(16,),
        resolution=64,
        dropout=0.0,
        resamp_with_conv=True,
        tanh_out=False,
        use_checkpoint=False,
    ).to(device)

    x = torch.randn(8, 16, 64).to(device)

    recon, posterior = model(x)

    print("Input shape :", x.shape)
    print("Recon shape :", recon.shape)