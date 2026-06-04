import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(pred, target, loss_type="l1"):
    if loss_type == "l1":
        return F.l1_loss(pred, target)

    if loss_type == "l2" or loss_type == "mse":
        return F.mse_loss(pred, target)

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target)

    raise ValueError(f"Unknown reconstruction loss type: {loss_type}")


def kl_loss(posterior):
    kl = posterior.kl()
    return torch.mean(kl)


def spectral_loss(pred, target, loss_type="l1", eps=1e-8):
    pred_fft = torch.fft.rfft(pred.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)

    pred_mag = torch.sqrt(pred_fft.real.pow(2) + pred_fft.imag.pow(2) + eps)
    target_mag = torch.sqrt(target_fft.real.pow(2) + target_fft.imag.pow(2) + eps)

    return reconstruction_loss(pred_mag, target_mag, loss_type=loss_type)


def multiscale_spectral_loss(pred, target, loss_type="l1", eps=1e-8,
                              fft_sizes=(64, 128, 256, 512)):
    pred_f, target_f = pred.float(), target.float()
    T = pred_f.shape[-1]
    losses = []

    for n in fft_sizes:
        if T < n:
            continue
        hop = n // 2
        p_frames = pred_f.unfold(-1, n, hop)
        t_frames = target_f.unfold(-1, n, hop)
        p_mag = torch.fft.rfft(p_frames, dim=-1).abs() + eps
        t_mag = torch.fft.rfft(t_frames, dim=-1).abs() + eps
        losses.append(reconstruction_loss(p_mag, t_mag, loss_type))

    if not losses:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(losses).mean()


def generator_hinge_loss(logits_fake):
    return -logits_fake.clamp(min=-50.0, max=50.0).mean()


def discriminator_hinge_loss(logits_real, logits_fake):
    logits_real = logits_real.clamp(min=-50.0, max=50.0)
    logits_fake = logits_fake.clamp(min=-50.0, max=50.0)
    loss_real = F.relu(1.0 - logits_real).mean()
    loss_fake = F.relu(1.0 + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


class PretrainingLoss(nn.Module):
    def __init__(
        self,
        rec_weight=1.0,
        kl_weight=1e-6,
        spectral_weight=0.1,
        rec_loss_type="smooth_l1",
        spectral_loss_type="l1",
        spectral_fft_sizes=None,
    ):
        super().__init__()
        self.rec_weight = rec_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.rec_loss_type = rec_loss_type
        self.spectral_loss_type = spectral_loss_type
        self.spectral_fft_sizes = tuple(spectral_fft_sizes) if spectral_fft_sizes else None

    def forward(self, pred, target, posterior):
        rec = reconstruction_loss(pred, target, self.rec_loss_type)
        kl = kl_loss(posterior)
        if self.spectral_fft_sizes:
            spectral = multiscale_spectral_loss(
                pred, target, self.spectral_loss_type,
                fft_sizes=self.spectral_fft_sizes,
            )
        else:
            spectral = spectral_loss(pred, target, self.spectral_loss_type)

        total = (
            self.rec_weight * rec
            + self.kl_weight * kl
            + self.spectral_weight * spectral
        )

        reg_key = getattr(posterior, "loss_name", "kl_loss")

        logs = {
            "loss": total.detach(),
            "rec_loss": rec.detach(),
            reg_key: kl.detach(),
            "spectral_loss": spectral.detach(),
        }

        return total, logs


class VAEGANLoss(nn.Module):
    def __init__(
        self,
        rec_weight=1.0,
        kl_weight=1e-6,
        spectral_weight=0.1,
        adversarial_weight=0.01,
        rec_loss_type="smooth_l1",
        spectral_loss_type="l1",
        spectral_fft_sizes=None,
    ):
        super().__init__()

        self.pretraining_loss = PretrainingLoss(
            rec_weight=rec_weight,
            kl_weight=kl_weight,
            spectral_weight=spectral_weight,
            rec_loss_type=rec_loss_type,
            spectral_loss_type=spectral_loss_type,
            spectral_fft_sizes=spectral_fft_sizes,
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
            "generator_loss": total.detach(),
        }

        return total, logs

    def discriminator_loss(self, logits_real, logits_fake):
        loss = discriminator_hinge_loss(logits_real, logits_fake)

        logs = {
            "disc_loss": loss.detach(),
            "logits_real": logits_real.mean().detach(),
            "logits_fake": logits_fake.mean().detach(),
        }

        return loss, logs








class JEPA_Loss(nn.Module):

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        if loss_type not in {"mse", "smooth_l1"}:
            raise ValueError(f"Unknown JEPA loss type: {loss_type!r}. Expected 'mse' or 'smooth_l1'.")
        self.loss_type = loss_type

    def forward(self, z_pred, z_target):
        if self.loss_type == "mse":
            loss = F.mse_loss(z_pred, z_target)
        else:
            loss = F.smooth_l1_loss(z_pred, z_target)

        logs = {
            "loss": loss.detach(),
            "jepa_loss": loss.detach(),
        }
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
