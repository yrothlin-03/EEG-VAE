from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) in sys.path:
    sys.path.remove(str(THIS_DIR))

if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

sys.path.insert(0, str(ROOT))

from eeg_vae.discriminator import Discriminator
from eeg_vae.eeg_vae import EEGVAE
from eeg_vae.modules.autoencodeur_kl import AutoencoderKL
from eeg_vae.modules.channel_adaptor import ChannelAdaptor
from eeg_vae.modules.decoder import Decoder
from eeg_vae.modules.encoder import Encoder


def assert_finite(tensor):
    assert torch.isfinite(tensor).all()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shape(tensor):
    return tuple(tensor.shape)


def print_model(name, model):
    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    print(f"\n{name}")
    print(f"  parameters: {total:,}")
    print(f"  trainable:  {trainable:,}")


def print_shapes(**tensors):
    for name, tensor in tensors.items():
        print(f"  {name}: {shape(tensor)}")


def backward_check(output, model):
    loss = output.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_channel_adaptor():
    model = ChannelAdaptor(in_channels=8, out_channels=12, n_layers=2)
    x = torch.randn(2, 8, 32)
    y = model(x)

    print_model("ChannelAdaptor", model)
    print_shapes(input=x, output=y)

    assert y.shape == (2, 12, 32)
    assert_finite(y)
    backward_check(y, model)


def test_encoder_decoder():
    encoder = Encoder(
        in_channels=8,
        z_channels=4,
        ch=16,
        ch_mult=(1, 2),
        num_res_blocks=1,
        resolution=32,
    )
    decoder = Decoder(
        out_channels=8,
        z_channels=4,
        ch=16,
        ch_mult=(1, 2),
        num_res_blocks=1,
        resolution=32,
    )

    x = torch.randn(2, 8, 32)
    z = encoder(x)
    y = decoder(z[:, :4])

    print_model("Encoder", encoder)
    print_model("Decoder", decoder)
    print_shapes(input=x, encoded=z, latent=z[:, :4], decoded=y)

    assert z.shape == (2, 8, 16)
    assert y.shape == x.shape
    assert_finite(z)
    assert_finite(y)
    backward_check(y, decoder)


def test_autoencoder_kl():
    model = AutoencoderKL(
        in_channels=8,
        z_channels=4,
        ch=16,
        ch_mult=(1, 2),
        num_res_blocks=1,
        resolution=32,
    )
    x = torch.randn(2, 8, 32)
    y, posterior = model(x, sample_posterior=False, return_posterior=True)

    print_model("AutoencoderKL", model)
    print_shapes(
        input=x,
        output=y,
        posterior_mean=posterior.mean,
        posterior_logvar=posterior.logvar,
    )

    assert y.shape == x.shape
    assert posterior.mean.shape == (2, 4, 16)
    assert posterior.logvar.shape == (2, 4, 16)
    assert posterior.kl().shape == (2,)
    assert_finite(y)
    assert_finite(posterior.mean)
    backward_check(y, model)


def test_eeg_vae():
    model = EEGVAE(
        in_channels=8,
        adapted_channels=8,
        adaptor_layers=2,
        z_channels=4,
        ch=16,
        ch_mult=(1, 2),
        num_res_blocks=1,
        resolution=32,
    )
    x = torch.randn(2, 8, 32)
    y, posterior = model(x, sample_posterior=False, return_posterior=True)
    reconstruction = model.reconstruct(model.channel_adaptor(x))

    print_model("EEGVAE", model)
    print_shapes(
        input=x,
        output=y,
        reconstruction=reconstruction,
        posterior_mean=posterior.mean,
    )

    assert y.shape == x.shape
    assert reconstruction.shape == x.shape
    assert posterior.mean.shape == (2, 4, 16)
    assert_finite(y)
    backward_check(y, model)


def test_discriminator():
    model = Discriminator(in_channels=8, ndf=16, n_layers=2)
    x = torch.randn(2, 8, 32)
    y = model(x)

    print_model("Discriminator", model)
    print_shapes(input=x, output=y)

    assert y.shape[0] == 2
    assert y.shape[1] == 1
    assert y.ndim == 3
    assert_finite(y)
    backward_check(y, model)


def main():
    tests = [
        test_channel_adaptor,
        test_encoder_decoder,
        test_autoencoder_kl,
        test_eeg_vae,
        test_discriminator,
    ]

    for test in tests:
        test()
        print(f"{test.__name__}: ok")


if __name__ == "__main__":
    main()  
