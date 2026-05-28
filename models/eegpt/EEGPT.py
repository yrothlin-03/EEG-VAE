import torch
import torch.nn as nn
from functools import partial
from .EEGPT_mcae import EEGTransformer


class EEGPT(nn.Module):
    def __init__(
        self,
        channels,
        checkpoint_path=None,
        img_size=(19, 1024),
        patch_size=64,
        embed_num=4,
        embed_dim=512,
        depth=8,
        num_heads=8,
        freeze=True,
    ):
        super().__init__()

        self.encoder = EEGTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_num=embed_num,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.chan_ids = self.encoder.prepare_chan_ids(channels)

        if checkpoint_path is not None:
            self.load_eegpt_checkpoint(checkpoint_path)

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def load_eegpt_checkpoint(self, checkpoint_path):
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        if any(k.startswith("target_encoder.") for k in state):
            encoder_state = {
                k.replace("target_encoder.", "", 1): v
                for k, v in state.items()
                if k.startswith("target_encoder.")
            }
        elif any(k.startswith("encoder.") for k in state):
            encoder_state = {
                k.replace("encoder.", "", 1): v
                for k, v in state.items()
                if k.startswith("encoder.")
            }
        else:
            encoder_state = state

        missing, unexpected = self.encoder.load_state_dict(
            encoder_state,
            strict=False,
        )

        print(f"[EEGPT LOAD] Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def forward(self, x, mask_x=None, mask_t=None):
        """
        x: Tensor [B, C, T]
        sortie: Tensor [B, N, embed_num, embed_dim]
        """

        chan_ids = self.chan_ids.to(x.device)

        if not self.encoder.training:
            with torch.no_grad():
                z = self.encoder(
                    x,
                    chan_ids=chan_ids,
                    mask_x=mask_x,
                    mask_t=mask_t,
                )
        else:
            z = self.encoder(
                x,
                chan_ids=chan_ids,
                mask_x=mask_x,
                mask_t=mask_t,
            )

        return z