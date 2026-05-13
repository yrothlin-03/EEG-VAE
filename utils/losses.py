import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(pred, target, loss_type="l1"):
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unknown reconstruction loss: {loss_type}")


def kl_loss(posterior):
    return posterior.kl().mean()


def spectral_loss(pred, target, loss_type="l1", eps=1e-8):
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    pred_mag = torch.sqrt(pred_fft.real.pow(2) + pred_fft.imag.pow(2) + eps)
    target_mag = torch.sqrt(target_fft.real.pow(2) + target_fft.imag.pow(2) + eps)

    return reconstruction_loss(pred_mag, target_mag, loss_type=loss_type)


def generator_hinge_loss(logits_fake):
    return -logits_fake.mean()


def discriminator_hinge_loss(logits_real, logits_fake):
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


class PretrainingLoss(nn.Module):
    def __init__(
        self,
        rec_weight=1.0,
        kl_weight=1e-6,
        spectral_weight=0.1,
        rec_loss_type="l1",
        spectral_loss_type="l1",
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.rec_loss_type = rec_loss_type
        self.spectral_loss_type = spectral_loss_type

    def forward(self, pred, target, posterior):
        rec = reconstruction_loss(pred, target, self.rec_loss_type)
        kl = kl_loss(posterior)
        spectral = spectral_loss(pred, target, self.spectral_loss_type)

        total = (
            self.rec_weight * rec
            + self.kl_weight * kl
            + self.spectral_weight * spectral
        )

        logs = {
            "loss": total.detach(),
            "rec_loss": rec.detach(),
            "kl_loss": kl.detach(),
            "spectral_loss": spectral.detach(),
        }

        return total, logs


class VAEGANLoss(nn.Module):
    def __init__(
        self,
        rec_weight=1.0,
        kl_weight=1e-6,
        spectral_weight=0.1,
        adversarial_weight=0.1,
        rec_loss_type="l1",
        spectral_loss_type="l1",
    ):
        super().__init__()
        self.pretraining_loss = PretrainingLoss(
            rec_weight=rec_weight,
            kl_weight=kl_weight,
            spectral_weight=spectral_weight,
            rec_loss_type=rec_loss_type,
            spectral_loss_type=spectral_loss_type,
        )
        self.adversarial_weight = adversarial_weight

    def generator_loss(self, pred, target, posterior, logits_fake):
        base_loss, logs = self.pretraining_loss(pred, target, posterior)
        adv = generator_hinge_loss(logits_fake)
        total = base_loss + self.adversarial_weight * adv

        logs = {
            **logs,
            "loss": total.detach(),
            "adv_loss": adv.detach(),
        }

        return total, logs

    def discriminator_loss(self, logits_real, logits_fake):
        loss = discriminator_hinge_loss(logits_real, logits_fake)
        logs = {"disc_loss": loss.detach()}
        return loss, logs


if __name__ == "__main__":
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from eeg_vae.modules.distributions import DiagonalGaussianDistribution

    pred = torch.randn(2, 8, 32, requires_grad=True)
    target = torch.randn(2, 8, 32)
    posterior_params = torch.randn(2, 8, 16)
    posterior = DiagonalGaussianDistribution(posterior_params)

    pretraining_loss = PretrainingLoss()
    loss, logs = pretraining_loss(pred, target, posterior)
    loss.backward()

    print("PretrainingLoss")
    for name, value in logs.items():
        print(f"  {name}: {value.item():.6f}")
    print(f"  pred grad: {pred.grad.abs().sum().item():.6f}")

    pred = torch.randn(2, 8, 32, requires_grad=True)
    logits_fake = torch.randn(2, 1, 6, requires_grad=True)
    logits_real = torch.randn(2, 1, 6, requires_grad=True)

    vae_gan_loss = VAEGANLoss()
    gen_loss, gen_logs = vae_gan_loss.generator_loss(
        pred,
        target,
        posterior,
        logits_fake,
    )
    disc_loss, disc_logs = vae_gan_loss.discriminator_loss(
        logits_real,
        logits_fake.detach(),
    )

    gen_loss.backward()
    disc_loss.backward()

    print("\nVAEGANLoss generator")
    for name, value in gen_logs.items():
        print(f"  {name}: {value.item():.6f}")
    print(f"  pred grad: {pred.grad.abs().sum().item():.6f}")
    print(f"  fake logits grad: {logits_fake.grad.abs().sum().item():.6f}")

    print("\nVAEGANLoss discriminator")
    for name, value in disc_logs.items():
        print(f"  {name}: {value.item():.6f}")
    print(f"  real logits grad: {logits_real.grad.abs().sum().item():.6f}")
