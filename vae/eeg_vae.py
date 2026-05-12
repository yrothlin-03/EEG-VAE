import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.autoencodeur_kl import AutoencoderKL
from .modules.discriminator import (
    NLayerDiscriminator1D,
    hinge_d_loss,
    hinge_g_loss,
    vanilla_d_loss,
    weights_init,
)


__all__ = ["EEGVAE"]


class EEGVAE(nn.Module):
    """
    VAE-GAN wrapper for EEG signals shaped as (B, C, T).

    The autoencoder handles reconstruction and KL regularization. The
    discriminator can be used during training to add an adversarial signal.
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
        use_discriminator=True,
        disc_channels=64,
        disc_layers=3,
        disc_use_actnorm=False,
        disc_loss="hinge",
        kl_weight=1.0e-6,
        rec_weight=1.0,
        adv_weight=0.0,
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.autoencoder = AutoencoderKL(
            in_channels=in_channels,
            out_channels=out_channels,
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

        self.use_discriminator = use_discriminator
        self.kl_weight = kl_weight
        self.rec_weight = rec_weight
        self.adv_weight = adv_weight

        if disc_loss not in {"hinge", "vanilla"}:
            raise ValueError("disc_loss must be either 'hinge' or 'vanilla'.")
        self.disc_loss = disc_loss

        self.discriminator = None
        if use_discriminator:
            self.discriminator = NLayerDiscriminator1D(
                input_nc=out_channels,
                ndf=disc_channels,
                n_layers=disc_layers,
                use_actnorm=disc_use_actnorm,
            )
            self.discriminator.apply(weights_init)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def reconstruct(self, x):
        return self.autoencoder.reconstruct(x)

    def forward(self, x, sample_posterior=True, return_posterior=True):
        return self.autoencoder(
            x,
            sample_posterior=sample_posterior,
            return_posterior=return_posterior,
        )

    def reconstruction_loss(self, x, recon, reduction="mean"):
        return F.l1_loss(recon, x, reduction=reduction)

    def kl_loss(self, posterior):
        return torch.mean(posterior.kl())

    def generator_loss(self, x, sample_posterior=True):
        recon, posterior = self.forward(
            x,
            sample_posterior=sample_posterior,
            return_posterior=True,
        )

        rec_loss = self.reconstruction_loss(x, recon)
        kl_loss = self.kl_loss(posterior)
        loss = self.rec_weight * rec_loss + self.kl_weight * kl_loss

        logs = {
            "loss/generator_total": loss.detach(),
            "loss/reconstruction": rec_loss.detach(),
            "loss/kl": kl_loss.detach(),
        }

        if self.discriminator is not None and self.adv_weight > 0.0:
            logits_fake = self.discriminator(recon)
            adv_loss = hinge_g_loss(logits_fake)
            loss = loss + self.adv_weight * adv_loss
            logs["loss/adversarial"] = adv_loss.detach()

        return loss, logs, recon, posterior

    def discriminator_loss(self, x, recon=None):
        if self.discriminator is None:
            raise RuntimeError("EEGVAE was created with use_discriminator=False.")

        if recon is None:
            with torch.no_grad():
                recon = self.reconstruct(x)

        logits_real = self.discriminator(x.detach())
        logits_fake = self.discriminator(recon.detach())

        if self.disc_loss == "hinge":
            loss = hinge_d_loss(logits_real, logits_fake)
        else:
            loss = vanilla_d_loss(logits_real, logits_fake)

        logs = {
            "loss/discriminator": loss.detach(),
            "logits/real": logits_real.mean().detach(),
            "logits/fake": logits_fake.mean().detach(),
        }

        return loss, logs

    def get_last_layer(self):
        return self.autoencoder.get_last_layer()
