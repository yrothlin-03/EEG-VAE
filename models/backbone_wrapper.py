from pathlib import Path
from typing import Optional

import numpy as np
import mne
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torcheeg.datasets.constants import SEED_CHANNEL_LIST

from models import CBRAMOD, LUNA, FEMBA, EEGMAMBA, REVE, EEGPT

SEED_PRETRAINING_CHANNEL_LIST = SEED_CHANNEL_LIST
SEED_PRETRAINING_CHANNEL_LIST = [
    ch for ch in SEED_PRETRAINING_CHANNEL_LIST
    if ch not in ["PO5", "PO6", "CB1", "CB2"]
]

TUEG_CHANNEL_LIST = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "A1-T3", "T4-A2",
]


def get_channel_locations(channel_names):
    if "-" in channel_names[0]:
        names = list(set([part for ch in channel_names for part in ch.split("-")]))
    else:
        names = channel_names

    info = mne.create_info(ch_names=names, sfreq=256, ch_types=["eeg"] * len(names))
    info = info.set_montage(
        mne.channels.make_standard_montage("standard_1005"),
        match_case=False,
        match_alias={"cb1": "POO7", "cb2": "POO8"},
    )

    locs = []
    for name in channel_names:
        if name in TUEG_CHANNEL_LIST:
            electrode1, electrode2 = name.split("-")
            loc1 = info.get_montage().get_positions()["ch_pos"][electrode1]
            loc2 = info.get_montage().get_positions()["ch_pos"][electrode2]
            locs.append((loc1 + loc2) / 2)
        else:
            locs.append(info.get_montage().get_positions()["ch_pos"][name])
    return locs


SEED_LOC = get_channel_locations(SEED_PRETRAINING_CHANNEL_LIST)

TARGET_CHS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "FZ", "CZ", "PZ",
]

EEGPT_BCIC2A_PROBE_CHS = [
    "FP1", "FP2",
    "F7", "F3", "FZ", "F4", "F8",
    "T7", "C3", "CZ", "C4", "T8",
    "P7", "P3", "PZ", "P4", "P8",
    "O1", "O2",
]

CHANNELS_IDS = list(range(len(TARGET_CHS)))

CHANNELS = {
    "BCI2A": ["FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4", "P1", "PZ", "P2", "POZ"],
    "FACED": ["FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "CZ", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6", "PZ", "P3", "P4", "P7", "P8", "PO3", "PO4", "OZ", "O1", "O2"],
    "PHYSIONETMI": ["FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ"],
    "SLEEPEDF": ["FPZ-CZ", "PZ-OZ"],
    "SEEDV": ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2"],
}


def get_positions_3d(chs, montage_name: str = "standard_1020") -> np.ndarray:
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]
    pos_ci = {k.upper(): np.asarray(v, dtype=np.float32) for k, v in pos.items()}

    out = []
    missing = []

    for ch in chs:
        ch_u = ch.upper()

        if ch_u in pos_ci:
            out.append(pos_ci[ch_u])
            continue

        if "-" in ch_u:
            parts = ch_u.split("-")
            if all(p in pos_ci for p in parts):
                out.append(np.mean([pos_ci[p] for p in parts], axis=0))
                continue

        missing.append(ch)

    if missing:
        raise ValueError(f"Channels not found in montage {montage_name}: {missing}")

    return np.stack(out, axis=0).astype(np.float32)


CHANNEL_POS = {
    name: get_positions_3d(chs)
    for name, chs in CHANNELS.items()
}

CHANNELS_LOCATIONS_3D = get_positions_3d(TARGET_CHS)


class BackboneWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        model_config: dict,
        dataset_name: Optional[str] = None,
        mode: Optional[str] = None,
        patch_size: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_config = model_config
        self.dataset_name = dataset_name
        self.mode = mode or "raw"
        self.patch_size = patch_size
        self.cls_query_token_from_ckpt = None
        self.backbone = self._build_backbone()

    def _build_backbone(self):
        if self.model_name == "REVE":
            encoder, cls_query_token = REVE.from_pretrained(
                model_id="brain-bzh/reve-base",
                cache_dir=None,
            )
            self.cls_query_token_from_ckpt = cls_query_token
            return encoder

        if self.model_name == "CBRAMOD":
            return CBRAMOD(**self.model_config)

        if self.model_name == "LUNA":
            return LUNA(**self.model_config)

        if self.model_name == "FEMBA":
            return FEMBA(**self.model_config)

        if self.model_name == "EEGMAMBA":
            return EEGMAMBA(**self.model_config)


        if self.model_name == "EEGPT":
            eegpt_config = dict(self.model_config)
            use_bcic2a_probe_adapter = bool(
                eegpt_config.pop("use_eegpt_bcic2a_probe_adapter", False)
            )
            eegpt_config.pop("input_adapter_in_channels", None)
            eegpt_config.pop("input_adapter_out_channels", None)
            eegpt_config.pop("input_adapter_max_norm", None)
            eegpt_config.pop("input_target_length", None)

            channels = (
                EEGPT_BCIC2A_PROBE_CHS
                if use_bcic2a_probe_adapter
                else self._get_dataset_channels()
            )
            eegpt_config["channels"] = channels

            if "img_size" not in eegpt_config:
                eegpt_config["img_size"] = (len(channels), 1024)
            else:
                eegpt_config["img_size"] = (len(channels), eegpt_config["img_size"][1])

            return EEGPT(**eegpt_config)

        raise ValueError(f"Model {self.model_name} is not registered.")

    def _get_dataset_channels(self):
        if self.mode not in ("raw", "mapped"):
            raise ValueError(f"Unknown mode: {self.mode}. Available: ['raw', 'mapped']")

        if self.mode == "mapped":
            return TARGET_CHS

        if self.dataset_name is None:
            return TARGET_CHS

        key = self.dataset_name.upper()

        if key not in CHANNELS:
            raise ValueError(
                f"Unknown dataset_name: {self.dataset_name}. "
                f"Available: {list(CHANNELS.keys())}"
            )

        return CHANNELS[key]

    def _get_channel_pos(self, x: torch.Tensor):
        channels = self._get_dataset_channels()

        if self.mode == "mapped":
            pos = CHANNELS_LOCATIONS_3D
        else:
            key = self.dataset_name.upper() if self.dataset_name is not None else None
            if key is not None and key in CHANNEL_POS:
                pos = CHANNEL_POS[key]
            else:
                pos = get_positions_3d(channels)

        if pos.shape[0] != x.shape[1]:
            raise ValueError(
                f"Channel mismatch for {self.dataset_name}: "
                f"x has {x.shape[1]} channels but positions have {pos.shape[0]}"
            )

        return torch.as_tensor(
            pos,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).repeat(x.size(0), 1, 1)

    def load_checkpoints(self, checkpoint_path: Path, strict: bool = False):
        if self.model_name == "EEGPT":
            return self.backbone.load_eegpt_checkpoint(checkpoint_path)

        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.suffix == ".safetensors":
            state_dict = load_file(checkpoint_path.as_posix(), device="cpu")
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

        if any(k.startswith("backbone.") for k in state_dict.keys()):
            state_dict = {
                k.replace("backbone.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }

        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=strict)
        print(f"[BACKBONE LOAD] Missing: {len(missing)} | Unexpected: {len(unexpected)}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def make_mask(self, x: torch.Tensor, mask_ratio: float = 0.5):
        if x.ndim != 4:
            return None
        b, c, l, _ = x.shape
        return (torch.rand(b, c, l, device=x.device) < mask_ratio).long()

    def _patchify(self, x: torch.Tensor):
        if self.patch_size is None:
            raise ValueError(f"patch_size must be provided for {self.model_name}.")
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,T) for {self.model_name}, got {tuple(x.shape)}")

        t = x.size(-1)
        pad_len = (self.patch_size - (t % self.patch_size)) % self.patch_size

        if pad_len:
            x = F.pad(x, (0, pad_len), mode="constant", value=0)

        l = x.size(-1) // self.patch_size
        return x.reshape(x.size(0), x.size(1), l, self.patch_size)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = True,
        mask=None,
        seed: Optional[int] = None,
        ch_names=None,
    ):
        if self.model_name == "REVE":
            pos = self._get_channel_pos(x)
            return self.backbone(x, pos, return_output=False)

        if self.model_name in ("CBRAMOD", "EEGMAMBA"):
            x = self._patchify(x)

            if mask is None:
                return self.backbone(x, return_features=return_features)

            return self.backbone(x, mask=mask, return_features=return_features)

        if self.model_name == "LUNA":
            channel_locations = self._get_channel_pos(x)

            channel_names = torch.arange(
                x.size(1),
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0).repeat(x.size(0), 1)

            return self.backbone(
                x,
                mask=mask,
                channel_locations=channel_locations,
                channel_names=channel_names,
                return_features=return_features,
            )

        if self.model_name == "EEGPT":
            return self.backbone(x)

        return self.backbone(x, return_features=return_features)