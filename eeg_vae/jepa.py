import copy

import torch
import torch.nn as nn

from .eeg_vae import EEGVAE
from .modules.predictor import EEGPredictor


__all__ = ["EEGJEPA", "JEPAEncoder", "create_block_mask", "create_channel_mask", "build_jepa_model"]


class JEPAEncoder(nn.Module):

    def __init__(self, eeg_vae: EEGVAE):
        super().__init__()

        if not hasattr(eeg_vae.autoencoder, "quantize"):
            raise ValueError(
                "JEPA Phase 2 requires a VQ-VAE (model_type='vq'). "
                "The provided model does not have a quantize layer."
            )

        self.channel_adaptor = eeg_vae.channel_adaptor
        self.encoder = eeg_vae.autoencoder.encoder
        self.quant_conv = eeg_vae.autoencoder.quant_conv
        self.quantize = eeg_vae.autoencoder.quantize

        self.quantize.freeze_codebook()

    def forward(self, x):
        h = self.channel_adaptor(x)
        h = self.encoder(h)
        h = self.quant_conv(h)
        z_q, _, _, _ = self.quantize(h)
        return z_q


def create_block_mask(
    T_tokens: int,
    block_size: int,
    target_ratio: float,
    device,
):
    n_full_blocks = T_tokens // block_size
    n_target_blocks = max(1, round(n_full_blocks * target_ratio))

    perm = torch.randperm(n_full_blocks, device=device)
    target_block_ids = perm[:n_target_blocks]
    context_block_ids = perm[n_target_blocks:]

    def _blocks_to_positions(block_ids):
        if len(block_ids) == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        positions = []
        for b in block_ids.tolist():
            positions.extend(range(b * block_size, (b + 1) * block_size))
        return torch.tensor(sorted(positions), dtype=torch.long, device=device)

    target_pos = _blocks_to_positions(target_block_ids)
    context_pos = _blocks_to_positions(context_block_ids)

    remainder_start = n_full_blocks * block_size
    if remainder_start < T_tokens:
        remainder = torch.arange(remainder_start, T_tokens, dtype=torch.long, device=device)
        context_pos = torch.cat([context_pos, remainder])

    return context_pos, target_pos


def create_channel_mask(
    n_channels: int,
    channel_mask_ratio: float,
    batch_size: int,
    device,
):
    n_mask = max(1, round(n_channels * channel_mask_ratio))
    mask = torch.ones(batch_size, n_channels, 1, device=device)
    for b in range(batch_size):
        masked_channels = torch.randperm(n_channels, device=device)[:n_mask]
        mask[b, masked_channels, :] = 0.0
    return mask


class EEGJEPA(nn.Module):

    def __init__(
        self,
        context_encoder: JEPAEncoder,
        predictor: EEGPredictor,
        ema_momentum: float = 0.996,
    ):
        super().__init__()

        self.context_encoder = context_encoder
        self.target_encoder = copy.deepcopy(context_encoder)
        self.predictor = predictor
        self.ema_momentum = ema_momentum

        embed_dim = predictor.embed_dim
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    def _apply_mask(self, z, target_pos):
        B, D, _ = z.shape
        z_masked = z.clone()
        mask = self.mask_token.to(z.dtype).view(1, D, 1).expand(B, D, len(target_pos))
        z_masked[:, :, target_pos] = mask
        return z_masked

    def forward(self, x, target_pos, channel_mask=None):
        x_context = x * channel_mask if channel_mask is not None else x

        z_context = self.context_encoder(x_context)
        z_masked = self._apply_mask(z_context, target_pos)
        z_pred = self.predictor(z_masked, target_pos)

        with torch.no_grad():
            z_target = self.target_encoder(x)[:, :, target_pos]

        return z_pred, z_target

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = None):
        m = self.ema_momentum if momentum is None else float(momentum)
        for p_c, p_t in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)


def build_jepa_model(
    vae_checkpoint_path: str,
    model_config: dict,
    predictor_config: dict,
    ema_momentum: float = 0.996,
    device="cpu",
) -> EEGJEPA:
    ckpt  = torch.load(vae_checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    model_config = dict(model_config)
    w = state.get("autoencoder.decoder.conv_out.weight")
    if w is not None:
        model_config["ch"] = int(w.shape[1])
    n_q = 0
    while f"autoencoder.quantize.quantizers.{n_q}.embedding" in state:
        n_q += 1
    if n_q > 0:
        model_config["vq_n_quantizers"] = n_q

    has_mamba = any("autoencoder.encoder.mid.attn_1.mamba." in k for k in state)
    model_config["sequence_block"] = "mamba" if has_mamba else "attention"

    print(
        f"[JEPA] Detected VQ-VAE: ch={model_config.get('ch')} "
        f"vq_n_quantizers={n_q} sequence_block={model_config['sequence_block']}"
    )

    eeg_vae = EEGVAE(**model_config)
    eeg_vae.load_state_dict(state)
    eeg_vae.to(device)

    context_encoder = JEPAEncoder(eeg_vae)
    predictor = EEGPredictor(**predictor_config)

    model = EEGJEPA(
        context_encoder=context_encoder,
        predictor=predictor,
        ema_momentum=ema_momentum,
    )
    model.sequence_block = model_config["sequence_block"]
    return model
