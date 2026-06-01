import copy

import torch
import torch.nn as nn

from .eeg_vae import EEGVAE
from .modules.predictor import EEGPredictor


__all__ = ["EEGJEPA", "JEPAEncoder", "create_block_mask", "create_channel_mask", "build_jepa_model"]


class JEPAEncoder(nn.Module):
    """
    Encoder-only wrapper around EEGVAE for use in JEPA Phase 2.

    Exposes the chain: channel_adaptor → encoder → quant_conv → quantize → z_q.
    The VQ codebook is frozen (EMA updates disabled); all other weights are
    updated normally by gradient during Phase 2.

    Requires model_type='vq' (raises if the loaded model has no quantize layer).
    """

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
        """
        Args:
            x: (B, in_channels, T)
        Returns:
            z_q: (B, embed_dim, T_tokens)   where T_tokens = T // compression_factor
        """
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
    """
    Temporal block masking for EEG-JEPA.

    Divides [0, n_full_blocks * block_size) into non-overlapping blocks of
    ``block_size`` tokens, then randomly selects ``target_ratio`` fraction as
    target blocks (to be predicted).  Remainder tokens
    (T_tokens % block_size) are always assigned to context.

    Returns the same mask for every item in a batch (broadcasted).

    Args:
        T_tokens:     Total number of latent tokens in the sequence.
        block_size:   Number of consecutive tokens per block.
        target_ratio: Fraction of full blocks selected as targets (0 < ratio < 1).
        device:       Torch device.

    Returns:
        context_pos: (n_context,) LongTensor of visible token indices.
        target_pos:  (n_target,)  LongTensor of target token indices.
    """
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

    # Append remainder tokens to context
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
    """
    Per-sample channel masking: for each item in the batch, independently zeros out
    a random subset of EEG channels before the context encoder.

    Simulates missing/noisy electrodes, making the predictor learn to infer
    spatial structure from incomplete observations.

    Args:
        n_channels:         Total number of EEG channels (e.g. 19).
        channel_mask_ratio: Fraction of channels to zero out (0 < ratio < 1).
        batch_size:         Number of items in the batch.
        device:             Torch device.

    Returns:
        mask: (B, C, 1) float tensor — 0 for masked channels, 1 for kept channels.
              Broadcasts over the time dimension when multiplied with x: (B, C, T).
    """
    n_mask = max(1, round(n_channels * channel_mask_ratio))
    mask = torch.ones(batch_size, n_channels, 1, device=device)
    for b in range(batch_size):
        masked_channels = torch.randperm(n_channels, device=device)[:n_mask]
        mask[b, masked_channels, :] = 0.0
    return mask


class EEGJEPA(nn.Module):
    """
    Phase 2 JEPA model built on top of a pretrained VQ-VAE.

    Components
    ----------
    context_encoder : JEPAEncoder
        The gradient-updated encoder.  VQ codebook is frozen; everything else
        (channel adaptor, conv weights, etc.) is trained.

    target_encoder : JEPAEncoder  (deep copy, EMA-updated)
        A momentum copy of the context encoder.  Never receives gradients
        directly — updated only via ``update_target_encoder()``.

    predictor : EEGPredictor
        Narrow transformer that predicts target z_q tokens from a masked
        version of the context z_q sequence.

    mask_token : nn.Parameter  (embed_dim,)
        Learnable token substituted at target positions before prediction.

    Training objective
    ------------------
    MSE( predictor(z_context_masked)[target_pos],
         target_encoder(x)[target_pos].detach() )
    """

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
        """Replace z[:, :, target_pos] with the learnable mask token."""
        B, D, _ = z.shape
        z_masked = z.clone()
        mask = self.mask_token.to(z.dtype).view(1, D, 1).expand(B, D, len(target_pos))
        z_masked[:, :, target_pos] = mask
        return z_masked

    def forward(self, x, target_pos, channel_mask=None):
        """
        Args:
            x:            (B, in_channels, T)
            target_pos:   (n_target,) LongTensor — temporal token indices to predict.
            channel_mask: (B, in_channels, 1) float tensor or None.
                          0 = channel zeroed out in context; 1 = channel kept.
                          Created by create_channel_mask(). Target encoder always
                          receives the full unmasked signal.

        Returns:
            z_pred:   (B, D, n_target) — predicted representations (gradient-tracked).
            z_target: (B, D, n_target) — ground-truth from EMA encoder (detached).
        """
        # Apply channel mask to context input only
        x_context = x * channel_mask if channel_mask is not None else x

        # Context path: gradient flows through context_encoder and predictor
        z_context = self.context_encoder(x_context)
        z_masked = self._apply_mask(z_context, target_pos)
        z_pred = self.predictor(z_masked, target_pos)

        # Target path: full input, no gradient
        with torch.no_grad():
            z_target = self.target_encoder(x)[:, :, target_pos]

        return z_pred, z_target

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = None):
        """EMA update: θ_target ← m·θ_target + (1−m)·θ_context.

        If ``momentum`` is given it overrides ``self.ema_momentum`` for this
        step — used by JEPATrainer to apply a cosine warmup schedule.
        """
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
    """
    Build an EEGJEPA model by loading a Phase 1 VQ-VAE checkpoint.

    Args:
        vae_checkpoint_path: path to the .pt file saved by the Phase 1 trainer.
        model_config:        the same model section used in Phase 1 (to rebuild EEGVAE).
        predictor_config:    dict of EEGPredictor hyperparameters.
        ema_momentum:        EMA momentum for the target encoder.
        device:              target device.

    Returns:
        Initialised EEGJEPA ready for Phase 2 training.
    """
    ckpt  = torch.load(vae_checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Auto-detect ch, n_quantizers and sequence_block from checkpoint weights so
    # that jepa.yaml never needs to be kept manually in sync with the VQ-VAE
    # config. Switching the Phase 1 model = just changing phase1.checkpoint.
    model_config = dict(model_config)
    w = state.get("autoencoder.decoder.conv_out.weight")
    if w is not None:
        model_config["ch"] = int(w.shape[1])
    n_q = 0
    while f"autoencoder.quantize.quantizers.{n_q}.embedding" in state:
        n_q += 1
    if n_q > 0:
        model_config["vq_n_quantizers"] = n_q

    # sequence_block: mamba if Mamba-specific keys are present, else attention.
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
    # Expose the resolved sequence_block so the caller can route checkpoints
    # into the correct variant directory without re-reading the checkpoint.
    model.sequence_block = model_config["sequence_block"]
    return model
