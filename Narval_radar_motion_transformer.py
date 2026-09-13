"""
Narval_radar_motion_vit_b16_s16_s8_s16plus.py
==============================================
FOUR-MODEL pipeline: ViT-B/16 (ImageNet-21K), ViT-S/16 (DINO), ViT-S/8 (DINO),
and ViT-S16+ (DINOv3), fine-tuned on the CI4R / OpenRadarInitiative
cross-frequency micro-Doppler HAR dataset (11 classes x
{10ghz, 24ghz, 77ghz, combined}).

MODELS + WEIGHTS
----------------
    vit_b16_in21k      : ViT-B/16, ImageNet-21K pretraining (MIIL weights of
                         arXiv:2104.10972). timm: vit_base_patch16_224_miil.in21k
                         (fallbacks: .orig_in21k / .augreg_in21k, warned).
    vit_s16_dino       : ViT-S/16, DINO self-supervised ImageNet weights
                         (HF: facebook/dino-vits16 ; timm: vit_small_patch16_224.dino)
    vit_s8_dino        : ViT-S/8,  DINO self-supervised ImageNet weights
                         (HF: facebook/dino-vits8  ; timm: vit_small_patch8_224.dino)
                         NOTE: patch 8 -> 784 tokens at 224x224 => ~4x the compute
                         of S/16. Budget walltime accordingly.
    vit_s16plus_dinov3 : ViT-S+/16, DINOv3 weights (LVD-1689M)
                         (HF: facebook/dinov3-vits16plus-pretrain-lvd1689m ;
                          timm: vit_small_plus_patch16_dinov3.lvd1689m — NOT gated)

PER-MODEL, ARTICLE-FAITHFUL FINE-TUNING (see MODEL_RECIPE)
---------------------------------------------------------
The two source papers use different recipes, so the optimizer, schedule, LR,
weight decay, label smoothing, and Cutout are selected PER MODEL — the
protocols never bleed into each other:

  vit_b16_in21k  ("ImageNet-21K Pretraining for the Masses", arXiv:2104.10972):
     Adam, weight decay 1e-4, label smoothing 0.1, Cutout aug; the paper's
     fixed schedule is 1-cycle with max LR 3e-4.
  vit_s16 / vit_s8 / vit_s16plus  ("When ViTs Outperform ResNets...",
     arXiv:2106.01548, Table 9 / Appendix D):
     SGD momentum 0.9, NO weight decay, plain CE, basic aug, grad-clip 1.0;
     the paper's fixed schedule is cosine decay + warmup, base LR 3e-3.
     SAM (the paper's core trick) is available via --sam (--sam-rho, Table 11
     suggests 0.1) but OFF by default — the paper omits SAM at fine-tuning.

Honest, documented deviations: batch 32 (papers use larger) for single-GPU
memory; the {ce,edl} x {physics} x {denoising} axes are OUR study grid layered
on each paper's fine-tuning engine; the ViT-S weights are DINO/DINOv3 (the SAM
paper's own ViT-S checkpoints are JAX-only and "S16+" exists only in DINOv3),
chosen per user request; on spectrograms a horizontal flip = time reversal and
a vertical flip = Doppler flip, and light SpecAugment masking + noise stands in
for "basic" preprocessing.

TRAINING SCHEDULE (default): "train until overfitting" — each model's base
optimizer (Adam for B/16, SGD-momentum for the S models) with
ReduceLROnPlateau + early stopping (patience 25) under a 200-epoch budget.
This deviates ONLY from each paper's fixed schedule horizon (a schedule
detail); architectures and pretraining are unchanged. Use --paper-schedule
for each model's exact fixed-length paper schedule (1-cycle / cosine), no
early stopping.

ABLATION GRID (as requested)
----------------------------
    Loss          : {ce, edl}
    Physics-prior : {off, on}          (Doppler-symmetry consistency)
    Pretrain(aux) : {none, denoising}  ("contrastive" removed)
      => 2 x 2 x 2 = 8 configs, each under all three smart downscaling modes
      => 24 trainings per (model, band); 3 models x 4 bands = 12 task IDs.

SMART DOWNSCALING — all three integrated
----------------------------------------
    mixed        : alpha*max_pool + (1-alpha)*avg_pool          (1 channel)
    multichannel : Ch0 area-avg / Ch1 mixed / Ch2 max-pool      (3 ch, default)
    autoencoder  : multichannel at a 1x intermediate (224x224), then a small
                   learned conv stem compresses it to 224x224, trained jointly.

INFERENCE: TWO columns reported side by side for every config:
    softmax        : one deterministic pass (CE: softmax(logits/T); EDL: alpha/S)
    tta_mcdropout  : _enable_mc_dropout() keeps dropout active and
                     _tta_mc_probs() averages 4 TTA views x MC_DROPOUT_PASSES
                     stochastic passes. Models are fine-tuned with
                     drop_rate=MODEL_DROPOUT (0.1) because stock ViTs ship with
                     p=0.0, under which MC-dropout would be a no-op.

USAGE
-----
  # once, on a Narval LOGIN node (internet; export HF_TOKEN for DINOv3):
  python Narval_radar_motion_VIT_S16_S8_S16plus.py --download-weights

  # SLURM array (3 models x 4 bands = 12 tasks):
  sbatch --array=0-11 --time=36:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=32G \
      --account=def-shervinv \
      --wrap="python Narval_radar_motion_VIT_S16_S8_S16plus.py \
              --task-id \\$SLURM_ARRAY_TASK_ID"

  # after all tasks finish:
  python Narval_radar_motion_VIT_S16_S8_S16plus.py --aggregate
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ====================================================================
# 1. CONFIG
# ====================================================================
DATA_ROOT = Path(r"Data")
FREQ_BANDS = ["10ghz", "24ghz", "77ghz"]
NARVAL_BANDS = ["10ghz", "24ghz", "77ghz", "combined"]

CLASSES = {
    0:  ("05", "WLKT",   "Walking towards radar"),
    1:  ("06", "WALKA",  "Walking away from radar"),
    2:  ("07", "PICK",   "Picking up an object"),
    3:  ("08", "BEND",   "Bending"),
    4:  ("09", "SIT",    "Sitting on chair"),
    5:  ("10", "KNEEL",  "Kneeling"),
    6:  ("11", "CRWL",   "Crawling towards radar"),
    7:  ("16", "WTOES",  "Walking on both toes"),
    8:  ("17", "LIMP",   "Limping with right leg stiff"),
    9:  ("18", "SHTEPS", "Walking with short steps"),
    10: ("19", "SCSSR",  "Scissors gait"),
}
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = [CLASSES[i][1] for i in range(NUM_CLASSES)]

# ---- the FOUR models, each with its ARTICLE-FAITHFUL fine-tuning recipe ----
# vit_b16_in21k  -> "ImageNet-21K Pretraining for the Masses" (arXiv:2104.10972):
#                   Adam, 1-cycle LR (max 3e-4), weight decay 1e-4, Cutout aug,
#                   label smoothing 0.1.
# vit_s16/s8/s16plus -> "When ViTs Outperform ResNets..." (arXiv:2106.01548):
#                   SGD momentum 0.9, cosine decay, NO weight decay, basic aug,
#                   plain CE. (SAM optional via --sam; the paper omits SAM at FT.)
# Each model's optimizer, schedule, LR, weight decay, Cutout, and label
# smoothing are selected per MODEL_RECIPE below — so the two papers' protocols
# don't bleed into each other.
MODEL_RECIPE = {
    "vit_b16_in21k": {
        "optimizer": "adam", "lr": 3e-4, "weight_decay": 1e-4,
        "paper_sched": "onecycle", "cutout": 0.25, "label_smoothing": 0.1,
        "note": "ImageNet-21K (Adam + 1-cycle)"},
    "vit_s16_dino": {
        "optimizer": "sgd", "lr": 3e-3, "weight_decay": 0.0,
        "paper_sched": "cosine", "cutout": 0.0, "label_smoothing": 0.0,
        "note": "SAM-paper (SGD-momentum + cosine)"},
    "vit_s8_dino": {
        "optimizer": "sgd", "lr": 3e-3, "weight_decay": 0.0,
        "paper_sched": "cosine", "cutout": 0.0, "label_smoothing": 0.0,
        "note": "SAM-paper (SGD-momentum + cosine)"},
    "vit_s16plus_dinov3": {
        "optimizer": "sgd", "lr": 3e-3, "weight_decay": 0.0,
        "paper_sched": "cosine", "cutout": 0.0, "label_smoothing": 0.0,
        "note": "SAM-paper (SGD-momentum + cosine)"},
}
MODEL_LR = {k: v["lr"] for k, v in MODEL_RECIPE.items()}
ALL_MODELS = list(MODEL_RECIPE.keys())   # b16 first -> task-ids 0-3=b16, 4-7=s16, 8-11=s8, 12-15=s16+

MAX_FILES_PER_ACT = None

# ---- signal processing ----
STFT_WINDOW = 256
STFT_OVERLAP = 200
STFT_NFFT = 512
SPECTROGRAM_SIZE = (224, 224)
MTI_FILTER = True
DB_FLOOR = -40
SAMPLING_RATES = {"10ghz": 1000, "24ghz": 1000, "77ghz": 1000}
DATA_FORMAT = "auto"
DB_DYNAMIC_RANGE = 50
DOPPLER_CROP_FRAC = 0.6

BAND_REGION_MODE = "manual"
BAND_REGION_LAYOUT = "mask"
BAND_DOPPLER_REGIONS = {}
AUTO_REGION_DB = 8.0
AUTO_REGION_MIN_FRAC = 0.04

# ---- smart downscaling ----
DOWNSCALE_MODE = "multichannel"
MIX_ALPHA = 0.5
AE_UPSCALE = 1
AE_INPUT_SIZE = (int(224 * AE_UPSCALE), int(224 * AE_UPSCALE))   
DOWNSCALE_MODES_TO_COMPARE = ["mixed", "multichannel", "autoencoder"]
COMPARE_DOWNSCALING = True
_DS_TAG = {"mixed": "mix", "multichannel": "multi", "autoencoder": "ae"}

MAT_VAR_CANDIDATES = ("sx1", "data", "rawData", "raw_data", "iq", "x", "signal",
                      "spectrogram", "spectro", "micro_doppler", "md", "S", "img")

# ---- paper fine-tuning recipe (arXiv:2106.01548, Table 9 / Appendix D) ----
# DEFAULT MODE: "train until overfitting" — SGD(momentum 0.9) + plateau LR +
# early stopping under EPOCHS_BUDGET. --paper-schedule = the article's exact
# fixed-length cosine decay (with linear warmup), no early stopping.
FT_EPOCHS = 40           # fixed-length cosine schedule (used by --paper-schedule)
FT_MAX_LR = 3e-3         # base LR (paper grid {1e-3, 3e-3, 1e-2, 3e-2})
FT_MOMENTUM = 0.9        # SGD momentum (paper fine-tuning optimizer)
FT_WEIGHT_DECAY = 0.0    # paper: "no weight decay" during fine-tuning
PAPER_SCHEDULE = False
EPOCHS_BUDGET = 200
EARLY_STOPPING_PATIENCE = 25
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-7
WARMUP_EPOCHS = 3        # linear warmup (paper uses warmup steps)
BATCH_SIZE = 32          # paper: 512 — reduced for single-GPU memory (documented)
LABEL_SMOOTHING = 0.0    # paper fine-tuning uses plain CE (no label smoothing)
GRAD_CLIP = 1.0          # paper: grad clipping at global norm 1
USE_MIXUP = False

# ---- SAM (the article's core technique; OPTIONAL — paper does NOT use SAM
#      during fine-tuning, only for from-scratch ImageNet training) ----
USE_SAM = False
SAM_RHO = 0.1            # paper Table 11: rho=0.1 for ViT-S/16 (supervised)

# ViTs ship with dropout p=0.0 — MC-dropout would be a NO-OP. Fine-tune with a
# small nonzero dropout inside the transformer blocks:
MODEL_DROPOUT = 0.1
MC_DROPOUT_PASSES = 5

# SAFETY: if pretrained weights cannot be loaded (empty cache on an offline
# compute node), ABORT instead of silently training random-init models for
# days. Override only for a deliberate from-scratch experiment (--allow-random-init).
ALLOW_RANDOM_INIT = False

# ---- augmentation (spectrogram analogue of "basic" preprocessing) ----
AUG_TIME_MASK = 0.15
AUG_FREQ_MASK = 0.15
AUG_NOISE_STD = 0.02
AUG_CUTOUT_FRAC = 0.0    # paper uses only basic aug — Cutout OFF here

# ---- splits / misc ----
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42
import platform as _platform
NUM_WORKERS = 0 if _platform.system() == "Windows" else 2
try:
    _ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if _ncpu > 0:
        NUM_WORKERS = max(0, _ncpu - 1)
except Exception:
    pass

NARVAL_DATA_ROOT = Path(r"Data")
NARVAL_OUT_DIR = Path(r"RADAR_Motion_Results_VIT_S16_B16_S8_S16plus")

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


# ====================================================================
# 2. PRETRAINED-WEIGHT LOADING (timm first, HF transformers fallback)
# ====================================================================
_TIMM_CANDIDATES = {
    "vit_b16_in21k": [
        ("vit_base_patch16_224_miil.in21k",
         "MIIL ImageNet-21K ViT-B/16 (arXiv:2104.10972 paper weights)"),
        ("vit_base_patch16_224.orig_in21k",
         "original ImageNet-21K ViT-B/16 (substitute, warned)"),
        ("vit_base_patch16_224.augreg_in21k",
         "AugReg ImageNet-21K ViT-B/16 (substitute, warned)"),
    ],
    "vit_s16_dino": [
        ("vit_small_patch16_224.dino", "DINO ViT-S/16 (facebook/dino-vits16)"),
        ("vit_small_patch16_224.augreg_in21k",
         "AugReg IN-21K ViT-S/16 (NOT DINO — substitute, warned)"),
    ],
    "vit_s8_dino": [
        ("vit_small_patch8_224.dino", "DINO ViT-S/8 (facebook/dino-vits8)"),
    ],
    "vit_s16plus_dinov3": [
        ("vit_small_plus_patch16_dinov3.lvd1689m",  "DINOv3 ViT-S+/16 (timm)"),
        ("vit_small_plus_patch16_dinov3",           "DINOv3 ViT-S+/16 (timm)"),
    ],
}
_HF_FALLBACK = {
    "vit_b16_in21k":      "timm/vit_base_patch16_224.augreg_in21k",
    "vit_s16_dino":       "facebook/dino-vits16",
    "vit_s8_dino":        "facebook/dino-vits8",
    "vit_s16plus_dinov3": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
}


def _build_hf_wrapper(hf_id, num_classes, drop):
    """Fallback loader through HF transformers: backbone AutoModel + a fresh
    Dropout->Linear head on the CLS token. Requires internet or a pre-populated
    HF cache; DINOv3 additionally requires an accepted license + HF_TOKEN."""
    import torch.nn as nn
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(hf_id)
    hidden = getattr(backbone.config, "hidden_size", 384)

    class HFViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.drop = nn.Dropout(drop)
            self.head = nn.Linear(hidden, num_classes)
        def forward(self, x):
            out = self.backbone(pixel_values=x)
            cls = out.last_hidden_state[:, 0]
            return self.head(self.drop(cls))
    return HFViT()


def _build_backbone(model_name, num_classes=NUM_CLASSES, pretrained=True,
                    drop_rate=None):
    """Load the requested ViT-S variant with its pretrained weights.
    Chain: timm candidates -> HF transformers -> (S16+ only) DINO S/16
    substitute -> random init. Every fallback is loudly warned."""
    p = MODEL_DROPOUT if drop_rate is None else drop_rate
    # 1) timm
    try:
        import timm
        for tag, desc in _TIMM_CANDIDATES[model_name]:
            try:
                m = timm.create_model(tag, pretrained=pretrained,
                                      num_classes=num_classes, drop_rate=p)
                note = ("" if "substitute" not in desc
                        else "  [WARNING: substitute pretraining!]")
                print(f"[{model_name}] loaded timm {tag}  ({desc}){note}",
                      flush=True)
                return m
            except Exception as e:
                print(f"[{model_name}] timm {tag} unavailable: "
                      f"{type(e).__name__}: {e}", flush=True)
    except ImportError:
        print(f"[{model_name}] timm not installed — trying HF transformers.",
              flush=True)
    # 2) HF transformers
    try:
        m = _build_hf_wrapper(_HF_FALLBACK[model_name], num_classes, p)
        print(f"[{model_name}] loaded HF {_HF_FALLBACK[model_name]}", flush=True)
        return m
    except Exception as e:
        print(f"[{model_name}] HF {_HF_FALLBACK[model_name]} failed: "
              f"{type(e).__name__}: {e}", flush=True)
        if model_name == "vit_s16plus_dinov3":
            print("\n[WARNING] DINOv3 S16+ could not be loaded. Its timm weights "
                  "(vit_small_plus_patch16_dinov3.lvd1689m) are NOT gated, so this\n"
                  "          is almost certainly an empty cache — run "
                  "--download-weights on a login node.\n", flush=True)
    # 3) LAST RESORT. torchvision has NO ViT-S architecture (only B/16, B/32,
    # L/16, L/32, H/14), so timm is required even to build these models.
    try:
        import timm
    except ImportError as e:
        raise RuntimeError(
            "'timm' is not installed, and torchvision has no ViT-S architecture "
            "to fall back on. Install it in the venv: pip install timm") from e
    if not ALLOW_RANDOM_INIT:
        raise RuntimeError(
            f"\n{'='*72}\n"
            f"ABORTING: could not load PRETRAINED weights for '{model_name}'.\n"
            f"Every load path above failed — on an offline compute node this "
            f"almost always means the weight cache is EMPTY because the one-time\n"
            f"download was never run on a LOGIN node. Random-init training would\n"
            f"waste days and produce ~chance accuracy, so the job is stopping now.\n\n"
            f"FIX (run once on a Narval LOGIN node, which has internet):\n"
            f"    module load python cuda\n"
            f"    source ~/projects/def-shervinv/atikmahabub/"
            f"spectrum_sensing_env/bin/activate\n"
            f"    pip install timm transformers huggingface_hub\n"
            f"    export TORCH_HOME=~/projects/def-shervinv/atikmahabub/torch_weights\n"
            f"    export HF_HOME=$TORCH_HOME/hf\n"
            f"    python Narval_radar_motion_vit-s16.py --download-weights "
            f"--torch-home $TORCH_HOME\n"
            f"Then resubmit the job. (DINO/DINOv3 via timm are NOT gated — no "
            f"HF_TOKEN needed.)\n"
            f"To intentionally train from scratch instead, pass "
            f"--allow-random-init.\n{'='*72}\n")
    print(f"\n[WARNING] {model_name}: PRETRAINED weights unreachable — using "
          f"RANDOM init because --allow-random-init was set. Results will be "
          f"much worse.\n", flush=True)
    tag = _TIMM_CANDIDATES[model_name][0][0]
    return timm.create_model(tag, pretrained=False, num_classes=num_classes,
                             drop_rate=p)


def _download_all_weights(torch_home=None):
    """Run ONCE on a Narval LOGIN NODE (internet). For DINOv3, first accept the
    license at huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m and
    `export HF_TOKEN=...`."""
    if torch_home:
        os.makedirs(torch_home, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
        os.environ.setdefault("HF_HOME", str(Path(torch_home) / "hf"))
    ok = 0
    for name in ALL_MODELS:
        try:
            m = _build_backbone(name, pretrained=True)
            del m
            print(f"  [OK]   {name}", flush=True); ok += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}", flush=True)
    print(f"\nCached {ok}/{len(ALL_MODELS)} models. Submit the array job now.",
          flush=True)


class ViT_Radar:
    """Factory wrapper: interpolates non-224 inputs to 224x224 (fixed patch
    grids) and returns final logits."""
    def __new__(cls, model_name, dropout=None):
        import torch.nn as nn
        import torch.nn.functional as F
        base = _build_backbone(model_name, drop_rate=dropout)

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = base
            def forward(self, x):
                if x.shape[-1] != 224 or x.shape[-2] != 224:
                    x = F.interpolate(x, size=(224, 224), mode="bilinear",
                                      align_corners=False)
                return self.backbone(x)
        return Wrap()


# ---- autoencoder stem: learned compression 224x224 -> 224x224 ----
def _make_ae_stem(base_module, out_size):
    import torch.nn as nn
    class _AEStem(nn.Module):
        def __init__(self, base, out_hw):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
                nn.Conv2d(16, 3, 3, padding=1),
                nn.AdaptiveAvgPool2d(tuple(out_hw)),
                nn.Sigmoid())
            self.base = base
        def forward(self, x):
            return self.base(self.enc(x))
    return _AEStem(base_module, out_size)


def _get_model_cls(model_name):
    if DOWNSCALE_MODE == "autoencoder":
        def factory(_name=model_name):
            return _make_ae_stem(ViT_Radar(_name), SPECTROGRAM_SIZE)
        return factory
    def factory(_name=model_name):
        return ViT_Radar(_name)
    return factory


# ====================================================================
# 3. DATA LOADING
# ====================================================================
def _normalize(s: str) -> str:
    return re.sub(r"[\s_]+", "_", s.strip().lower())

def _folder_to_class(folder_name: str) -> int:
    m = re.search(r"(?:10|24|77)ghz[_\- ]?(\d{2})", _normalize(folder_name))
    if not m:
        return -1
    aid = m.group(1)
    for cid, (suffix, _, _) in CLASSES.items():
        if suffix == aid:
            return cid
    return -1

def _folder_to_band(folder_name: str) -> str:
    n = _normalize(folder_name)
    for b in FREQ_BANDS:
        if n.startswith(b):
            return b
    return ""

def _load_mat_array(path: Path) -> np.ndarray:
    import scipy.io as sio
    try:
        mat = sio.loadmat(str(path), squeeze_me=True)
        candidates = {k: v for k, v in mat.items()
                      if not k.startswith("__") and isinstance(v, np.ndarray)}
    except NotImplementedError:
        import h5py
        candidates = {}
        with h5py.File(str(path), "r") as f:
            for k in f.keys():
                try:
                    candidates[k] = np.array(f[k])
                except Exception:
                    pass
    if not candidates:
        raise ValueError(f"No arrays found in {path.name}")
    for name in MAT_VAR_CANDIDATES:
        if name in candidates and candidates[name].size > 100:
            return np.asarray(candidates[name]).squeeze()
    k = max(candidates, key=lambda kk: candidates[kk].size)
    return np.asarray(candidates[k]).squeeze()

def _load_dat_array(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.int16)
    if raw.size and raw.size % 2 == 0:
        iq = raw.astype(np.float32)
        return (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size and raw.size % 2 == 0:
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    raise ValueError(f"Unable to parse {path.name}")

def _mti(iq: np.ndarray) -> np.ndarray:
    if not MTI_FILTER:
        return iq
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 0.02, btype="high")
    return filtfilt(b, a, iq.real) + 1j * filtfilt(b, a, iq.imag)

def _apply_band_regions(mag: np.ndarray, band: str) -> np.ndarray:
    h = mag.shape[0]
    if BAND_REGION_MODE == "auto":
        return _auto_band_crop(mag)
    if BAND_REGION_MODE == "manual":
        regions = BAND_DOPPLER_REGIONS.get(band)
        if regions:
            clipped = [(max(0, min(h, int(lo))), max(0, min(h, int(hi))))
                       for lo, hi in regions]
            clipped = [(lo, hi) for lo, hi in clipped if hi > lo]
            if clipped:
                if BAND_REGION_LAYOUT == "mask":
                    out = np.full_like(mag, float(mag.min()))
                    for lo, hi in clipped:
                        out[lo:hi] = mag[lo:hi]
                    return out
                return np.concatenate([mag[lo:hi] for lo, hi in clipped], axis=0)
    if 0 < DOPPLER_CROP_FRAC < 1.0:
        keep = max(8, int(h * DOPPLER_CROP_FRAC)); lo = (h - keep) // 2
        return mag[lo:lo + keep]
    return mag

def _auto_band_crop(mag: np.ndarray) -> np.ndarray:
    row_e = 20 * np.log10(mag.mean(axis=1) + 1e-9)
    keep = row_e >= (row_e.max() - AUTO_REGION_DB)
    if keep.sum() < max(8, int(AUTO_REGION_MIN_FRAC * len(row_e))):
        h = len(row_e); k = max(8, int(h * DOPPLER_CROP_FRAC)); lo = (h - k) // 2
        return mag[lo:lo + k]
    return mag[keep]

def _area_avg(S: np.ndarray, target) -> np.ndarray:
    import torch, torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32))[None, None]
    return F.adaptive_avg_pool2d(t, tuple(target))[0, 0].numpy()

def _max_pool(S: np.ndarray, target) -> np.ndarray:
    import torch, torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32))[None, None]
    return F.adaptive_max_pool2d(t, tuple(target))[0, 0].numpy()

def _mixed(S: np.ndarray, target, alpha: float) -> np.ndarray:
    return alpha * _max_pool(S, target) + (1.0 - alpha) * _area_avg(S, target)

def _downscale(norm: np.ndarray, target) -> np.ndarray:
    if DOWNSCALE_MODE == "mixed":
        return _mixed(norm, target, MIX_ALPHA).astype(np.float32)
    ch0 = _area_avg(norm, target)
    ch1 = _mixed(norm, target, MIX_ALPHA)
    ch2 = _max_pool(norm, target)
    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)

def _spectrogram_image(arr: np.ndarray, band: str) -> np.ndarray:
    mag = np.abs(np.asarray(arr, dtype=np.complex128))
    if mag.ndim == 2:
        mag = _apply_band_regions(mag, band)
    db = 20 * np.log10(mag + 1e-9)
    db = np.clip(db, db.max() - DB_DYNAMIC_RANGE, db.max())
    norm = ((db - db.min()) / (db.max() - db.min() + 1e-9)).astype(np.float32)
    target = AE_INPUT_SIZE if DOWNSCALE_MODE == "autoencoder" else SPECTROGRAM_SIZE
    return _downscale(norm, target)

def _iq_to_spectrogram(iq: np.ndarray, fs: int, band: str) -> np.ndarray:
    from scipy.signal import stft
    iq = _mti(iq.astype(np.complex64).ravel())
    _, _, Z = stft(iq, fs=fs, nperseg=STFT_WINDOW,
                   noverlap=STFT_OVERLAP, nfft=STFT_NFFT, return_onesided=False)
    Z = np.fft.fftshift(Z, axes=0)
    S = np.abs(Z) ** 2
    S = 10 * np.log10(np.asarray(S, dtype=np.float64) + 1e-12)
    S = np.clip(S, S.max() + DB_FLOOR, S.max())
    norm = ((S - S.min()) / (S.max() - S.min() + 1e-9)).astype(np.float32)
    target = AE_INPUT_SIZE if DOWNSCALE_MODE == "autoencoder" else SPECTROGRAM_SIZE
    return _downscale(norm, target)

def array_to_spectrogram(arr: np.ndarray, fs: int, band: str) -> np.ndarray:
    arr = np.asarray(arr).squeeze()
    fmt = DATA_FORMAT
    if fmt == "auto":
        if arr.ndim == 2 and arr.shape[-1] != 2 and (np.iscomplexobj(arr)
                                                     or min(arr.shape) >= 8):
            fmt = "spectrogram"
        else:
            fmt = "iq"
    if fmt == "spectrogram":
        return _spectrogram_image(arr, band)
    if arr.ndim == 2 and arr.shape[-1] == 2 and not np.iscomplexobj(arr):
        arr = arr[..., 0] + 1j * arr[..., 1]
    if arr.ndim > 1:
        arr = arr.sum(axis=0)
    return _iq_to_spectrogram(arr, fs, band)


_LOAD_STATS = {"ok": 0, "failed": 0, "errors": {}}

def load_spectrogram(path: Path, band: str) -> np.ndarray:
    try:
        arr = (_load_mat_array(path) if path.suffix.lower() == ".mat"
               else _load_dat_array(path))
        S = array_to_spectrogram(arr, fs=SAMPLING_RATES[band], band=band)
        _LOAD_STATS["ok"] += 1
        return S
    except Exception as e:
        _LOAD_STATS["failed"] += 1
        key = f"{type(e).__name__}: {e}"
        _LOAD_STATS["errors"][key] = _LOAD_STATS["errors"].get(key, 0) + 1
        tgt = AE_INPUT_SIZE if DOWNSCALE_MODE == "autoencoder" else SPECTROGRAM_SIZE
        if DOWNSCALE_MODE == "mixed":
            return np.zeros(tgt, dtype=np.float32)
        return np.zeros((3, *tgt), dtype=np.float32)

def report_load_stats():
    ok, failed = _LOAD_STATS["ok"], _LOAD_STATS["failed"]
    total = ok + failed
    if total == 0:
        return
    print(f"\n[data] loaded {ok}/{total} files OK, {failed} FAILED "
          f"({100.0 * failed / total:.1f}%).")
    if failed:
        print("[data] !! Failed files were replaced with BLANK images. Top errors:")
        for err, n in sorted(_LOAD_STATS["errors"].items(),
                             key=lambda kv: -kv[1])[:5]:
            print(f"        {n:5d}x  {err}")

def index_files(root: Path, bands: List[str]) -> List[Tuple[Path, int, str]]:
    buckets = defaultdict(list)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        band = _folder_to_band(folder.name)
        cid = _folder_to_class(folder.name)
        if band not in bands or cid < 0:
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in (".mat", ".dat"):
                buckets[(cid, band)].append((f, cid, band))
    entries = []
    for (cid, band), files in sorted(buckets.items()):
        if MAX_FILES_PER_ACT is not None:
            files = files[:MAX_FILES_PER_ACT]
        entries.extend(files)
    return entries


class RadarSpectrogramDataset:
    """Picklable dataset with in-memory caching. Augmentation = light
    SpecAugment-style time/freq masking + noise (the spectrogram analogue of
    the paper's basic Inception-style preprocessing). Cutout stays OFF by
    default (AUG_CUTOUT_FRAC = 0) to match the article's 'basic' setting."""
    def __init__(self, entries, augment=False, cache=True):
        self.entries = entries
        self.augment = augment
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.entries)

    def _augment(self, S):
        H, W = (S.shape[-2], S.shape[-1])
        tm = int(np.random.uniform(0, AUG_TIME_MASK) * W)
        if tm:
            t0 = np.random.randint(0, W - tm)
            S[..., :, t0:t0 + tm] = 0
        fm = int(np.random.uniform(0, AUG_FREQ_MASK) * H)
        if fm:
            f0 = np.random.randint(0, H - fm)
            S[..., f0:f0 + fm, :] = 0
        if AUG_CUTOUT_FRAC > 0:
            side = int(AUG_CUTOUT_FRAC * min(H, W))
            if side > 1:
                cy = np.random.randint(0, H - side)
                cx = np.random.randint(0, W - side)
                S[..., cy:cy + side, cx:cx + side] = 0
        S = S + np.random.normal(0, AUG_NOISE_STD, S.shape).astype(np.float32)
        return np.clip(S, 0, 1)

    def __getitem__(self, idx):
        import torch
        path, cid, band = self.entries[idx]
        if self.cache and idx in self._cache:
            S = self._cache[idx].copy()
        else:
            S = load_spectrogram(path, band)
            if self.cache:
                self._cache[idx] = S.copy()
        if self.augment:
            S = self._augment(S)
        if S.ndim == 3:
            x = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32))
        else:
            x = torch.from_numpy(np.stack([S, S, S], axis=0).astype(np.float32))
        return x, cid


# ====================================================================
# 4. LOSSES (EDL / physics / denoising consistency)
# ====================================================================
def _kl_dirichlet_to_uniform(alpha, K):
    import torch
    S = alpha.sum(dim=1, keepdim=True)
    K_t = torch.tensor(float(K), device=alpha.device)
    first = (torch.lgamma(S).squeeze(1) - torch.lgamma(K_t)
             - torch.lgamma(alpha).sum(dim=1))
    second = ((alpha - 1) * (torch.digamma(alpha)
                             - torch.digamma(S))).sum(dim=1)
    return first + second

def edl_loss(logits, y_oh, num_classes, epoch=0, max_anneal_epochs=10, lam=0.1):
    import torch.nn.functional as F
    alpha = F.softplus(logits) + 1
    S = alpha.sum(dim=1, keepdim=True)
    err = (y_oh - alpha / S) ** 2
    var = alpha * (S - alpha) / (S ** 2 * (S + 1))
    data_term = (err + var).sum(dim=1).mean()
    alpha_tilde = y_oh + (1 - y_oh) * alpha
    kl = _kl_dirichlet_to_uniform(alpha_tilde, num_classes).mean()
    annealing = min(1.0, max(epoch, 0) / max(max_anneal_epochs, 1))
    return data_term + annealing * lam * kl

def edl_predict_probs(logits):
    import torch.nn.functional as F
    alpha = F.softplus(logits) + 1
    return alpha / alpha.sum(dim=1, keepdim=True)

def physics_doppler_symmetry_loss(model, x):
    import torch, torch.nn.functional as F
    with torch.no_grad():
        target = F.softmax(model(x), dim=1)
    lg_flip = model(torch.flip(x, dims=[-2]))     # flip Doppler axis
    return F.kl_div(F.log_softmax(lg_flip, dim=1), target, reduction="batchmean")

def consistency_denoising_loss(model, x, noise_std=0.10):
    import torch, torch.nn.functional as F
    x_noisy = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
    with torch.no_grad():
        target = F.softmax(model(x), dim=1)
    return F.kl_div(F.log_softmax(model(x_noisy), dim=1), target,
                    reduction="batchmean")


# ====================================================================
# 5. SAM OPTIMIZER (Foret et al. 2021 — the article's core technique;
#    OPTIONAL here because the article does not apply SAM at fine-tuning)
# ====================================================================
def _make_sam(params, base_cls, rho, **base_kwargs):
    import torch

    class SAM(torch.optim.Optimizer):
        """Two-step sharpness-aware update: (1) climb to w+e_hat along the
        gradient (e_hat = rho * g / ||g||), (2) descend using the gradient
        evaluated at the perturbed point."""
        def __init__(self, params, base_cls, rho, **kwargs):
            defaults = dict(rho=rho, **kwargs)
            super().__init__(params, defaults)
            self.base_optimizer = base_cls(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)

        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    e_w = p.grad * scale.to(p)
                    p.add_(e_w)
                    self.state[p]["e_w"] = e_w
            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None or "e_w" not in self.state[p]:
                        continue
                    p.sub_(self.state[p]["e_w"])
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad()

        def _grad_norm(self):
            shared = self.param_groups[0]["params"][0].device
            return torch.norm(torch.stack([
                p.grad.norm(p=2).to(shared)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None]), p=2)

        def step(self, closure=None):
            raise RuntimeError("SAM requires first_step/second_step usage.")

    return SAM(params, base_cls, rho, **base_kwargs)


# ====================================================================
# 6. TRAINING — the article's fine-tuning engine (SGD-momentum, cosine
#    or plateau+early-stop) with DUAL softmax / TTA+MC-dropout inference
# ====================================================================
def train_one_model(model_cls, model_name, train_ds, val_ds, test_ds,
                    epochs=None, max_lr=None,
                    batch_size=BATCH_SIZE, weight_decay=None,
                    loss_type="ce", use_physics=False, pretrain_mode="none",
                    use_temperature=True, verbose=False,
                    paper_schedule=None, mc_passes=None, use_sam=None):
    """Fine-tune one model with its ARTICLE-FAITHFUL recipe (see MODEL_RECIPE):
      * vit_b16_in21k      -> Adam, weight decay 1e-4, label smoothing 0.1;
                              paper schedule = 1-cycle (max_lr).
      * vit_s16/s8/s16plus -> SGD momentum 0.9, NO weight decay, plain CE;
                              paper schedule = cosine decay + warmup.
    Schedule modes:
      * default            : ReduceLROnPlateau + early stopping under
                             EPOCHS_BUDGET — "train until overfitting".
      * paper_schedule=True: the model's fixed-length paper schedule
                             (1-cycle for B/16, cosine for the S models),
                             no early stopping.
      * use_sam=True       : SAM two-step updates wrapping the recipe's base
                             optimizer (~2x cost/step).
    Best-val-loss weights restored before test. Reports softmax AND
    TTA+MC-dropout columns."""
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    if paper_schedule is None:
        paper_schedule = PAPER_SCHEDULE
    if mc_passes is None:
        mc_passes = MC_DROPOUT_PASSES
    if use_sam is None:
        use_sam = USE_SAM

    recipe = MODEL_RECIPE.get(model_name, {
        "optimizer": "sgd", "lr": FT_MAX_LR, "weight_decay": FT_WEIGHT_DECAY,
        "paper_sched": "cosine", "cutout": 0.0, "label_smoothing": 0.0})
    opt_name = recipe["optimizer"]
    if max_lr is None:
        max_lr = recipe["lr"]
    if weight_decay is None:
        weight_decay = recipe["weight_decay"]
    label_smoothing = recipe["label_smoothing"]
    if epochs is None:
        epochs = FT_EPOCHS if paper_schedule else EPOCHS_BUDGET

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls().to(device)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # ---- optimizer: per-recipe base (Adam for B/16, SGD-momentum for S) ----
    def _base_kwargs():
        kw = {"lr": max_lr, "weight_decay": weight_decay}
        if opt_name == "sgd":
            kw["momentum"] = FT_MOMENTUM
        return kw
    base_cls = torch.optim.Adam if opt_name == "adam" else torch.optim.SGD
    if use_sam:
        optim = _make_sam(model.parameters(), base_cls, rho=SAM_RHO,
                          **_base_kwargs())
        base_optim = optim.base_optimizer
    else:
        optim = base_cls(model.parameters(), **_base_kwargs())
        base_optim = optim

    # ---- schedule ----
    #   onecycle : per-BATCH (B/16 paper mode) — self-manages warmup
    #   cosine   : per-EPOCH (S paper mode) — manual linear warmup first
    #   plateau  : per-EPOCH on val loss (default early-stopping mode)
    onecycle = cosine = plateau = None
    if paper_schedule:
        if recipe["paper_sched"] == "onecycle":
            steps_per_epoch = max(1, math.ceil(len(train_ds) / batch_size))
            onecycle = torch.optim.lr_scheduler.OneCycleLR(
                base_optim, max_lr=max_lr, epochs=epochs,
                steps_per_epoch=steps_per_epoch)
        else:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                base_optim, T_max=max(1, epochs - WARMUP_EPOCHS),
                eta_min=MIN_LR)
    else:
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            base_optim, mode="min", patience=LR_SCHEDULER_PATIENCE,
            factor=LR_SCHEDULER_FACTOR, min_lr=MIN_LR)
    manual_warmup = onecycle is None    # OneCycle self-manages its warmup

    base_lrs = [g["lr"] for g in base_optim.param_groups]
    def apply_warmup(epoch):
        if epoch <= WARMUP_EPOCHS and WARMUP_EPOCHS > 0:
            scale = epoch / WARMUP_EPOCHS
            for g, b in zip(base_optim.param_groups, base_lrs):
                g["lr"] = b * scale

    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def compute_loss(x, y, ep):
        logits = model(x)
        if loss_type == "edl":
            y_oh = F.one_hot(y, NUM_CLASSES).float()
            loss = edl_loss(logits, y_oh, NUM_CLASSES, epoch=ep)
        else:
            loss = ce(logits, y)
        if use_physics:
            loss = loss + 0.05 * physics_doppler_symmetry_loss(model, x)
        if pretrain_mode == "denoising":
            loss = loss + 0.1 * consistency_denoising_loss(model, x)
        return loss

    best_val = float("inf"); best_state = None
    patience_left = EARLY_STOPPING_PATIENCE
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    for ep in range(1, epochs + 1):
        if manual_warmup:
            apply_warmup(ep)
        model.train(); tot, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_sam:
                loss = compute_loss(x, y, ep)
                loss.backward()
                optim.first_step(zero_grad=True)
                loss2 = compute_loss(x, y, ep)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.second_step(zero_grad=True)
            else:
                loss = compute_loss(x, y, ep)
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optim.step()
            if onecycle is not None:
                onecycle.step()          # 1-cycle steps per BATCH (B/16)
            tot += loss.item() * x.size(0); n += x.size(0)
        train_loss = tot / max(n, 1)

        model.eval(); vl, vn, correct = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                if loss_type == "edl":
                    probs = edl_predict_probs(logits)
                    vl += F.nll_loss(torch.log(probs.clamp(min=1e-9)),
                                     y).item() * x.size(0)
                else:
                    vl += F.cross_entropy(logits, y).item() * x.size(0)
                vn += x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        val_loss = vl / max(vn, 1); val_acc = correct / max(vn, 1)
        if ep > WARMUP_EPOCHS:
            if cosine is not None:
                cosine.step()
            if plateau is not None:
                plateau.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(base_optim.param_groups[0]["lr"])
        if verbose:
            print(f"[{model_name}] ep {ep:3d}/{epochs} tr={train_loss:.4f} "
                  f"val={val_loss:.4f} acc={val_acc:.4f} "
                  f"lr={base_optim.param_groups[0]['lr']:.2e}", flush=True)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0 and not paper_schedule:
                if verbose:
                    print(f"[{model_name}] early stop @ ep {ep} (val loss "
                          f"stalled {EARLY_STOPPING_PATIENCE} epochs)",
                          flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- temperature scaling (CE only; never changes argmax/accuracy) ----
    def _fit_temperature():
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_logits.append(model(x.to(device)))
                all_targets.append(y.to(device))
        logits = torch.cat(all_logits); targets = torch.cat(all_targets)
        T = nn.Parameter(torch.ones(1, device=device) * 1.5)
        opt = torch.optim.LBFGS([T], lr=0.05, max_iter=80)
        crit = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad()
            loss = crit(logits / T.clamp(min=1e-2), targets)
            loss.backward()
            return loss
        opt.step(closure)
        return float(T.detach().clamp(min=0.1, max=10.0).item())
    if use_temperature and loss_type == "ce":
        try:
            temperature = _fit_temperature()
        except Exception:
            temperature = 1.0
    else:
        temperature = 1.0

    # ---- DUAL inference: plain softmax AND TTA+MC-dropout -----------------
    def _enable_mc_dropout(m):
        for mod in m.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()

    def _probs_from_logits(lg):
        if loss_type == "edl":
            return edl_predict_probs(lg)
        return F.softmax(lg / temperature, dim=1)

    def _tta_mc_probs(x, n_mc):
        passes = [x, torch.flip(x, dims=[-1]),
                  torch.roll(x, shifts=8,  dims=-1),
                  torch.roll(x, shifts=-8, dims=-1)]
        probs_sum = 0; count = 0
        for xi in passes:
            for _ in range(max(1, n_mc)):
                probs_sum = probs_sum + _probs_from_logits(model(xi))
                count += 1
        return probs_sum / count

    def _pack(y_true, y_pred, probs):
        return {"y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
                "test_probs": probs.tolist(),
                "test_acc": float((y_true == y_pred).mean()),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                           zero_division=0))}

    model.eval()
    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs = _probs_from_logits(model(x.to(device)))
            all_y.append(y.numpy())
            all_p.append(probs.argmax(1).cpu().numpy())
            all_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(all_y)
    sm = _pack(y_true, np.concatenate(all_p), np.concatenate(all_prob, axis=0))

    model.eval()
    if mc_passes > 1:
        _enable_mc_dropout(model)
    all_p, all_prob = [], []
    with torch.no_grad():
        for x, _y in test_loader:
            probs = _tta_mc_probs(x.to(device), n_mc=mc_passes)
            all_p.append(probs.argmax(1).cpu().numpy())
            all_prob.append(probs.cpu().numpy())
    model.eval()
    mc = _pack(y_true, np.concatenate(all_p), np.concatenate(all_prob, axis=0))

    return {
        "model_name": model_name,
        "hyperparams": {"optimizer": opt_name + ("+SAM" if use_sam else ""),
                        "schedule": (recipe["paper_sched"] if paper_schedule
                                     else "plateau+early_stop"),
                        "base_lr": max_lr,
                        "momentum": FT_MOMENTUM if opt_name == "sgd" else None,
                        "weight_decay": weight_decay,
                        "label_smoothing": label_smoothing,
                        "batch_size": batch_size,
                        "sam_rho": SAM_RHO if use_sam else None,
                        "model_dropout": MODEL_DROPOUT,
                        "mc_passes": mc_passes,
                        "epochs_budget": epochs,
                        "epochs_run": len(history["train_loss"])},
        "history": history,
        "best_val_loss": float(best_val),
        "val_acc": float(max(history["val_acc"])) if history["val_acc"] else 0.0,
        "temperature": temperature,
        "softmax": sm,
        "tta_mcdropout": mc,
        "_state_dict": best_state,
    }


# ====================================================================
# 7. SPLITS + CONFIG GRID
# ====================================================================
def _nv_split(entries):
    from sklearn.model_selection import train_test_split
    labels = np.array([e[1] for e in entries]); idx = np.arange(len(entries))
    itrv, ite, ytrv, _ = train_test_split(idx, labels, test_size=TEST_SIZE,
                                          stratify=labels,
                                          random_state=RANDOM_SEED)
    itr, iva = train_test_split(itrv, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                stratify=ytrv, random_state=RANDOM_SEED)
    return ([entries[i] for i in itr], [entries[i] for i in iva],
            [entries[i] for i in ite])

def _nv_datasets(band_arg):
    bands = ["10ghz", "24ghz", "77ghz"] if band_arg == "combined" else [band_arg]
    entries = index_files(DATA_ROOT, bands)
    if not entries:
        return None
    tr, va, te = _nv_split(entries)
    return (RadarSpectrogramDataset(tr, augment=True, cache=True),
            RadarSpectrogramDataset(va, augment=False, cache=True),
            RadarSpectrogramDataset(te, augment=False, cache=True))

def _nv_configs(quick):
    """{ce,edl} x {phys off,on} x {none,denoising} = 8; x3 downscale = 24."""
    if quick:
        base = [{"loss": l, "physics": p, "pretrain": "none"}
                for l in ("ce", "edl") for p in (False, True)]
    else:
        base = [{"loss": l, "physics": p, "pretrain": pt}
                for l in ("ce", "edl") for p in (False, True)
                for pt in ("none", "denoising")]
    modes = DOWNSCALE_MODES_TO_COMPARE if COMPARE_DOWNSCALING else [DOWNSCALE_MODE]
    return [{**c, "downscale": ds} for ds in modes for c in base]

def _data_sig(dsmode):
    return dsmode   # mixed / multichannel / autoencoder each cache separately


# ====================================================================
# 8. TEXT CONFUSION MATRIX
# ====================================================================
def confusion_matrix_text(y_true, y_pred, class_names):
    from sklearn.metrics import confusion_matrix
    K = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    short = [c[:6] for c in class_names]
    rows = ["| true\\pred | " + " | ".join(short) + " | total |",
            "|" + "---|" * (K + 2)]
    for i in range(K):
        cells = " | ".join(str(int(cm[i, j])) for j in range(K))
        rows.append(f"| {short[i]} | {cells} | {int(cm[i].sum())} |")
    rows.append("")
    rows.append("Per-class recall (diagonal): " +
                ", ".join(f"{short[i]}={cm_norm[i, i]:.3f}" for i in range(K)))
    js = {"labels": list(class_names), "matrix_counts": cm.tolist(),
          "matrix_normalized": np.round(cm_norm, 4).tolist()}
    return "\n".join(rows), js


# ====================================================================
# 9. RUN ALL CONFIGS FOR ONE (model, band)
# ====================================================================
def nv_run_model_band(model_name, band_arg, out_root, quick, epochs):
    import torch
    configs = _nv_configs(quick)

    # Per-model augmentation: ViT-B/16 (IN-21K recipe) uses Cutout; the ViT-S
    # (SAM-paper recipe) use basic aug only. Set BEFORE building datasets — the
    # cache stores raw spectrograms and augmentation is re-applied per __getitem__.
    _rec = MODEL_RECIPE.get(model_name, {"cutout": 0.0})
    globals()["AUG_CUTOUT_FRAC"] = _rec.get("cutout", 0.0)
    print(f"[{band_arg}] {model_name} recipe: {_rec.get('note', '')} "
          f"(cutout={AUG_CUTOUT_FRAC})", flush=True)

    sigs = sorted(set(_data_sig(c["downscale"]) for c in configs))
    ds_bundles = {}
    for sig in sigs:
        globals()["DOWNSCALE_MODE"] = sig
        b = _nv_datasets(band_arg)
        if b is None:
            print(f"[!] no data for {band_arg}; skipping {model_name}.",
                  flush=True)
            return []
        ds_bundles[sig] = b

    out_dir = out_root / band_arg / f"ablation_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints" / band_arg / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ci, cfg in enumerate(configs):
        dsmode = cfg["downscale"]
        globals()["DOWNSCALE_MODE"] = dsmode
        cls = _get_model_cls(model_name)
        train_ds, val_ds, test_ds = ds_bundles[_data_sig(dsmode)]
        cfg_id = (f"DS{_DS_TAG.get(dsmode, dsmode)}_L{cfg['loss']}_"
                  f"PH{int(cfg['physics'])}_PT{cfg['pretrain']}")
        print(f"[{band_arg}] {model_name}  cfg {ci + 1}/{len(configs)} "
              f"({cfg_id})", flush=True)
        t0 = time.time()
        try:
            result = train_one_model(
                cls, model_name, train_ds, val_ds, test_ds, epochs=epochs,
                loss_type=cfg["loss"], use_physics=cfg["physics"],
                pretrain_mode=cfg["pretrain"],
                paper_schedule=PAPER_SCHEDULE, mc_passes=MC_DROPOUT_PASSES,
                use_sam=USE_SAM)
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        dt = round(time.time() - t0, 1)
        sm, mc = result["softmax"], result["tta_mcdropout"]
        if ci == 0:
            report_load_stats()

        state = result.pop("_state_dict", None)
        if state is not None:
            torch.save({"model": model_name, "band": band_arg, "config": cfg,
                        "state_dict": state,
                        "test_acc_softmax": sm["test_acc"],
                        "test_acc_tta_mcdropout": mc["test_acc"],
                        "input_size": list(SPECTROGRAM_SIZE)},
                       ckpt_dir / f"{cfg_id}.pth")
        print(f"   -> softmax acc={sm['test_acc']:.4f} "
              f"f1={sm['macro_f1']:.4f} | tta+mcdrop "
              f"acc={mc['test_acc']:.4f} f1={mc['macro_f1']:.4f} | "
              f"ep={result['hyperparams']['epochs_run']} t={dt}s  (ckpt saved)",
              flush=True)

        with open(out_dir / f"{cfg_id}_result.json", "w") as f:
            json.dump({"model": model_name, "band": band_arg, "config": cfg,
                       "config_id": cfg_id,
                       "val_acc": result.get("val_acc", 0),
                       "temperature": result.get("temperature", 1.0),
                       "epochs": result["hyperparams"]["epochs_run"],
                       "schedule": result["hyperparams"]["schedule"],
                       "optimizer": result["hyperparams"]["optimizer"],
                       "softmax": sm, "tta_mcdropout": mc}, f)
        rows.append({"model": model_name, "band": band_arg, **cfg,
                     "config_id": cfg_id,
                     "acc_softmax": sm["test_acc"],
                     "f1_softmax": sm["macro_f1"],
                     "acc_mcdropout": mc["test_acc"],
                     "f1_mcdropout": mc["macro_f1"],
                     "val_acc": result.get("val_acc", 0),
                     "temperature": result.get("temperature", 1.0),
                     "epochs": result["hyperparams"]["epochs_run"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda r: -r["acc_mcdropout"])
    sched_note = (f"paper cosine, {FT_EPOCHS} ep fixed" if PAPER_SCHEDULE
                  else f"plateau + early stop (patience "
                       f"{EARLY_STOPPING_PATIENCE}, budget {EPOCHS_BUDGET} ep)")
    sam_note = f", SAM rho={SAM_RHO}" if USE_SAM else ""
    md = [f"# Ablation -- band {band_arg}  model {model_name} "
          f"(SGD-m lr={MODEL_LR.get(model_name, FT_MAX_LR)}, {sched_note}"
          f"{sam_note})", "",
          "| Rank | Downscale | Loss | Physics | Pretrain | Acc(softmax) | "
          "F1(softmax) | Acc(TTA+MCdrop) | F1(TTA+MCdrop) | val_acc | T | "
          "epochs |",
          "|------|-----------|------|---------|----------|--------------|"
          "-------------|-----------------|----------------|---------|---|"
          "--------|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['downscale']} | {r['loss']} | "
                  f"{'on' if r['physics'] else 'off'} | {r['pretrain']} | "
                  f"{r['acc_softmax']:.4f} | {r['f1_softmax']:.4f} | "
                  f"{r['acc_mcdropout']:.4f} | {r['f1_mcdropout']:.4f} | "
                  f"{r['val_acc']:.4f} | {r['temperature']:.2f} | "
                  f"{r['epochs']} |")
    (out_dir / "comparison.md").write_text("\n".join(md))
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


# ====================================================================
# 10. EFFICIENCY PROFILE (params / GFLOPs / latency / memory)
# ====================================================================
def nv_profile(size, out_root):
    import torch, torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for model_name in ALL_MODELS:
        for dsmode in DOWNSCALE_MODES_TO_COMPARE:
            globals()["DOWNSCALE_MODE"] = dsmode
            cls = _get_model_cls(model_name)
            try:
                model = cls().to(device).eval()
            except Exception as e:
                print(f"[profile] {model_name}/{dsmode} build failed: {e}",
                      flush=True)
                continue
            n_params = sum(p.numel() for p in model.parameters())
            param_mb = sum(p.numel() * p.element_size()
                           for p in model.parameters()) / 1e6

            macs = [0]; handles = []
            def conv2d_hook(m, inp, out):
                oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
                kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple)
                          else (m.kernel_size,) * 2)
                macs[0] += oc * oh * ow * (inp[0].shape[1] // m.groups) * kh * kw
            def lin_hook(m, inp, out):
                macs[0] += (m.in_features * m.out_features
                            * (out.numel() // out.shape[-1]))
            for mod in model.modules():
                if isinstance(mod, nn.Conv2d):
                    handles.append(mod.register_forward_hook(conv2d_hook))
                elif isinstance(mod, nn.Linear):
                    handles.append(mod.register_forward_hook(lin_hook))
            in_hw = AE_INPUT_SIZE if dsmode == "autoencoder" else size
            x = torch.randn(1, 3, *in_hw, device=device)
            with torch.no_grad():
                _ = model(x)
            for h in handles:
                h.remove()
            gflops = 2 * macs[0] / 1e9

            with torch.no_grad():
                for _ in range(5):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(20):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat_ms = (time.time() - t0) / 20 * 1000

            peak_mb = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    model(x)
                torch.cuda.synchronize()
                peak_mb = torch.cuda.max_memory_allocated() / 1e6

            results.append({"model": model_name, "downscale": dsmode,
                            "params": int(n_params),
                            "param_mb": round(param_mb, 2),
                            "gflops": round(gflops, 3),
                            "latency_ms": round(lat_ms, 3),
                            "peak_activation_mb":
                                round(peak_mb, 2) if peak_mb else None})
            print(f"[profile] {model_name:20s}/{dsmode:12s} "
                  f"params={n_params/1e6:6.2f}M GFLOPs={gflops:8.2f} "
                  f"lat={lat_ms:7.2f}ms", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "efficiency_profile.json", "w") as f:
        json.dump({"input_size": list(size), "device": str(device),
                   "rows": results}, f, indent=2)
    md = [f"# Efficiency profile (input {size[0]}x{size[1]}, device={device})",
          "",
          "| Model | Downscale | Params (M) | Param mem (MB) | GFLOPs | "
          "Latency (ms) | Peak act. (MB) |",
          "|-------|-----------|-----------|----------------|--------|"
          "--------------|----------------|"]
    for r in results:
        md.append(f"| {r['model']} | {r['downscale']} | "
                  f"{r['params']/1e6:.2f} | {r['param_mb']:.1f} | "
                  f"{r['gflops']:.2f} | {r['latency_ms']:.2f} | "
                  f"{r['peak_activation_mb'] if r['peak_activation_mb'] else 'n/a'} |")
    (out_root / "efficiency_profile.md").write_text("\n".join(md))
    return results


# ====================================================================
# 11. AGGREGATE — master tables + metrics + text CMs + downscaling table
# ====================================================================
def nv_aggregate(out_root, size):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    cm_dir = out_root / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    band_summaries = {}; metrics_rows = []; all_rows = []
    cm_json_all = {}; ds_rows = []

    for band in NARVAL_BANDS:
        band_dir = out_root / band
        if not band_dir.exists():
            continue
        band_entries = []
        for model_name in ALL_MODELS:
            model_dir = band_dir / f"ablation_{model_name}"
            if not model_dir.exists():
                continue
            best = None
            for jf in model_dir.glob("*_result.json"):
                d = json.loads(jf.read_text())
                sm, mc = d["softmax"], d["tta_mcdropout"]
                all_rows.append({"model": d["model"], "band": d["band"],
                                 "config_id": d["config_id"],
                                 "acc_softmax": sm["test_acc"],
                                 "f1_softmax": sm["macro_f1"],
                                 "acc_mcdropout": mc["test_acc"],
                                 "f1_mcdropout": mc["macro_f1"],
                                 "val_acc": d.get("val_acc", 0)})
                ds_rows.append({"band": band, "model": d["model"],
                                "downscale": d["config"].get("downscale", "?"),
                                "acc_softmax": sm["test_acc"],
                                "acc_mcdropout": mc["test_acc"],
                                "val_acc": d.get("val_acc", 0)})
                band_entries.append({
                    "model": d["model"], "config_id": d["config_id"],
                    "loss": d["config"]["loss"],
                    "physics": d["config"]["physics"],
                    "pretrain": d["config"]["pretrain"],
                    "downscale": d["config"].get("downscale", "?"),
                    "acc_softmax": sm["test_acc"],
                    "f1_softmax": sm["macro_f1"],
                    "acc_mcdropout": mc["test_acc"],
                    "f1_mcdropout": mc["macro_f1"],
                    "val_acc": d.get("val_acc", 0)})
                if best is None or d["val_acc"] > best["val_acc"]:
                    best = d
            if best is None:
                continue
            # metrics + TEXT CMs for BOTH inference modes (best cfg per model)
            for mode in ("softmax", "tta_mcdropout"):
                y_true = np.array(best[mode]["y_true"])
                y_pred = np.array(best[mode]["y_pred"])
                acc = float(accuracy_score(y_true, y_pred))
                pr, rc, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0)
                metrics_rows.append({"band": band, "model": model_name,
                                     "inference": mode,
                                     "best_config": best["config_id"],
                                     "accuracy": round(acc, 4),
                                     "precision_macro": round(float(pr), 4),
                                     "recall_macro": round(float(rc), 4),
                                     "f1_macro": round(float(f1), 4)})
                cm_md, cm_js = confusion_matrix_text(y_true, y_pred,
                                                     CLASS_NAMES)
                (cm_dir / f"{band}_{model_name}_{mode}_cm.md").write_text(
                    f"# Confusion matrix -- {model_name} on {band} [{mode}] "
                    f"(best config {best['config_id']})\n\n{cm_md}\n")
                cm_json_all[f"{band}_{model_name}_{mode}"] = {
                    "best_config": best["config_id"], **cm_js}

        band_summaries[band] = sorted(band_entries,
                                      key=lambda e: -e["acc_mcdropout"])

    sched_note = (f"paper cosine, {FT_EPOCHS} ep fixed" if PAPER_SCHEDULE
                  else f"plateau + early stop (patience "
                       f"{EARLY_STOPPING_PATIENCE}, budget {EPOCHS_BUDGET} ep)")
    md = [f"# Master comparison — ViT-S/16 (DINO), ViT-S/8 (DINO), "
          f"ViT-S16+ (DINOv3)  (input {size[0]}x{size[1]})", "",
          f"Fine-tuning (arXiv:2106.01548 Table 9): SGD momentum "
          f"{FT_MOMENTUM}, no weight decay, grad-clip {GRAD_CLIP}, "
          f"{sched_note}{', SAM rho=' + str(SAM_RHO) if USE_SAM else ''}. "
          f"Inference: softmax vs TTA+MC-dropout ({MC_DROPOUT_PASSES} passes, "
          f"dropout {MODEL_DROPOUT}).", ""]
    for band, entries in band_summaries.items():
        md += [f"## Band: {band}", "",
               "| Rank | Model | Downscale | Loss | Phys | Pretrain | "
               "Acc(softmax) | F1(softmax) | Acc(TTA+MCdrop) | "
               "F1(TTA+MCdrop) |",
               "|------|-------|-----------|------|------|----------|"
               "--------------|-------------|-----------------|"
               "----------------|"]
        for i, e in enumerate(entries[:30], 1):
            md.append(f"| {i} | {e['model']} | {e['downscale']} | "
                      f"{e['loss']} | {'on' if e['physics'] else 'off'} | "
                      f"{e['pretrain']} | {e['acc_softmax']:.4f} | "
                      f"{e['f1_softmax']:.4f} | {e['acc_mcdropout']:.4f} | "
                      f"{e['f1_mcdropout']:.4f} |")
        md.append("")
    (out_root / "master_comparison.md").write_text("\n".join(md))
    with open(out_root / "master_comparison.json", "w") as f:
        json.dump({"models": ALL_MODELS, "input_size": list(size),
                   "band_summaries": band_summaries,
                   "all_configs": sorted(all_rows,
                                         key=lambda r: -r["acc_mcdropout"])},
                  f, indent=2)

    with open(out_root / "confusion_matrices_all.json", "w") as f:
        json.dump(cm_json_all, f, indent=2)

    mmd = ["# Best-config metrics per model (selected by val accuracy; "
           "both inference modes)", "",
           "| Band | Model | Inference | Best config | Accuracy | "
           "Precision (macro) | Recall (macro) | F1 (macro) |",
           "|------|-------|-----------|-------------|----------|"
           "-------------------|----------------|------------|"]
    for r in sorted(metrics_rows,
                    key=lambda r: (r["band"], r["model"], r["inference"])):
        mmd.append(f"| {r['band']} | {r['model']} | {r['inference']} | "
                   f"{r['best_config']} | {r['accuracy']:.4f} | "
                   f"{r['precision_macro']:.4f} | {r['recall_macro']:.4f} | "
                   f"{r['f1_macro']:.4f} |")
    (out_root / "metrics_best_per_model.md").write_text("\n".join(mmd))
    with open(out_root / "metrics_best_per_model.json", "w") as f:
        json.dump(metrics_rows, f, indent=2)

    _write_downscaling_comparison(out_root, ds_rows, size)
    print(f"[aggregate] wrote master_comparison, metrics_best_per_model, "
          f"{len(cm_json_all)} text confusion matrices, downscaling "
          f"comparison.", flush=True)


def _write_downscaling_comparison(out_root, rows, size):
    if not rows:
        return
    modes = sorted(set(r["downscale"] for r in rows))
    bands = [b for b in NARVAL_BANDS if any(r["band"] == b for r in rows)]
    models = sorted(set(r["model"] for r in rows))
    def best(pred, key="acc_mcdropout"):
        vals = [r[key] for r in rows if pred(r)]
        return max(vals) if vals else None
    md = [f"# Downscaling-method comparison (input {size[0]}x{size[1]}; "
          f"best TTA+MC-dropout accuracy — softmax column in the result "
          f"JSONs)", "",
          "## Best per band (max over models & configs)", "",
          "| Band | " + " | ".join(modes) + " | best method |",
          "|------|" + "------|" * (len(modes) + 1)]
    for band in bands:
        cells = []; bestmode = None; bestv = -1
        for m in modes:
            v = best(lambda r: r["downscale"] == m and r["band"] == band)
            cells.append(f"{v:.4f}" if v is not None else "-")
            if v is not None and v > bestv:
                bestv, bestmode = v, m
        md.append(f"| {band} | " + " | ".join(cells) + f" | **{bestmode}** |")
    md += ["", "## Best per model (max over bands & configs)", "",
           "| Model | " + " | ".join(modes) + " |",
           "|-------|" + "------|" * len(modes)]
    for model in models:
        cells = []
        for m in modes:
            v = best(lambda r: r["downscale"] == m and r["model"] == model)
            cells.append(f"{v:.4f}" if v is not None else "-")
        md.append(f"| {model} | " + " | ".join(cells) + " |")
    overall = {m: best(lambda r: r["downscale"] == m) for m in modes}
    overall = {k: v for k, v in overall.items() if v is not None}
    if overall:
        win = max(overall, key=overall.get)
        md += ["", f"**Overall best downscaling: {win} "
                   f"({overall[win]:.4f} peak accuracy).**", ""]
    (out_root / "downscaling_comparison.md").write_text("\n".join(md))
    with open(out_root / "downscaling_comparison.json", "w") as f:
        json.dump({"models": models, "input_size": list(size),
                   "modes": modes, "rows": rows}, f, indent=2)


# ====================================================================
# 12. MAIN
# ====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="ViT-S/16 + ViT-S/8 (DINO) + ViT-S16+ (DINOv3) on CI4R "
                    "radar HAR — arXiv:2106.01548 fine-tuning protocol, "
                    "8-config ablation x 3 smart-downscaling modes")
    ap.add_argument("--data-root", default=NARVAL_DATA_ROOT)
    ap.add_argument("--out-dir", default=NARVAL_OUT_DIR)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Epoch budget. Default: EPOCHS_BUDGET (200) with "
                         "early stopping, or 40 fixed under --paper-schedule.")
    ap.add_argument("--paper-schedule", action="store_true",
                    help="The article's exact fine-tuning schedule: fixed "
                         "cosine decay + warmup, NO early stopping. Default "
                         "mode instead trains until overfitting.")
    ap.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    ap.add_argument("--max-lr", type=float, default=None,
                    help="Override the per-model base LR (paper grid: 1e-3, "
                         "3e-3, 1e-2, 3e-2; default 3e-3).")
    ap.add_argument("--sam", action="store_true",
                    help="Enable SAM (the article's core technique; ~2x "
                         "compute). NOTE: the article itself does NOT use SAM "
                         "during fine-tuning.")
    ap.add_argument("--sam-rho", type=float, default=SAM_RHO,
                    help="SAM perturbation strength (paper Table 11: 0.1 for "
                         "ViT-S/16).")
    ap.add_argument("--mc-passes", type=int, default=MC_DROPOUT_PASSES)
    ap.add_argument("--model-dropout", type=float, default=MODEL_DROPOUT)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Reduce for ViT-S/8 if it OOMs (784 tokens).")
    ap.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    ap.add_argument("--bands", nargs="+", default=None)
    ap.add_argument("--task-id",
                    type=lambda v: None if (v is None or str(v).strip() == "")
                    else int(v), default=None,
                    help="SLURM array index 0..11 -> one (model, band) pair "
                         "(3 models x 4 bands; model-major order).")
    ap.add_argument("--downscale-mode",
                    choices=["mixed", "multichannel", "autoencoder"],
                    default=None,
                    help="Run ONLY this downscaling mode (default: compare "
                         "all three).")
    ap.add_argument("--mix-alpha", type=float, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="4 configs (drops the pretrain axis) x downscale "
                         "modes.")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--download-weights", action="store_true",
                    help="Cache all three models' weights (run ONCE on a LOGIN "
                         "node; DINO/DINOv3 via timm are NOT gated).")
    ap.add_argument("--allow-random-init", action="store_true",
                    help="Permit training from RANDOM weights if pretrained "
                         "weights can't be loaded. Default is to ABORT (an empty "
                         "cache should stop the job, not waste days).")
    ap.add_argument("--torch-home", default=None)
    args = ap.parse_args()

    if args.allow_random_init:
        globals()["ALLOW_RANDOM_INIT"] = True

    if args.torch_home:
        os.makedirs(args.torch_home, exist_ok=True)
        os.environ["TORCH_HOME"] = str(args.torch_home)
        os.environ.setdefault("HF_HOME", str(Path(args.torch_home) / "hf"))
        print(f"[cfg] TORCH_HOME={args.torch_home}", flush=True)
    if args.download_weights:
        _download_all_weights(args.torch_home)
        return

    # Narval compute nodes have NO internet: once weights are cached (login
    # node, --download-weights), force offline mode inside SLURM jobs so
    # timm/HuggingFace load from cache instead of hanging on network timeouts.
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        print("[cfg] SLURM job detected -> HF_HUB_OFFLINE=1 "
              "TRANSFORMERS_OFFLINE=1 (weights load from cache)", flush=True)

    globals()["SPECTROGRAM_SIZE"] = (args.size, args.size)
    globals()["AE_INPUT_SIZE"] = (int(args.size * AE_UPSCALE),
                                  int(args.size * AE_UPSCALE))
    globals()["DATA_ROOT"] = Path(args.data_root)
    globals()["PAPER_SCHEDULE"] = bool(args.paper_schedule)
    globals()["EARLY_STOPPING_PATIENCE"] = int(args.patience)
    globals()["MC_DROPOUT_PASSES"] = int(args.mc_passes)
    globals()["MODEL_DROPOUT"] = float(args.model_dropout)
    globals()["BATCH_SIZE"] = int(args.batch_size)
    globals()["USE_SAM"] = bool(args.sam)
    globals()["SAM_RHO"] = float(args.sam_rho)
    if args.max_lr is not None:
        for k in MODEL_LR:
            MODEL_LR[k] = float(args.max_lr)
    if args.epochs is not None:
        run_epochs = int(args.epochs)
    else:
        run_epochs = FT_EPOCHS if PAPER_SCHEDULE else EPOCHS_BUDGET
    if PAPER_SCHEDULE:
        globals()["FT_EPOCHS"] = run_epochs
    else:
        globals()["EPOCHS_BUDGET"] = run_epochs
    if args.mix_alpha is not None:
        globals()["MIX_ALPHA"] = args.mix_alpha
    if args.downscale_mode:
        globals()["DOWNSCALE_MODE"] = args.downscale_mode
        globals()["COMPARE_DOWNSCALING"] = False
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError:
        print("torch not installed.")
        return
    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    n_cfg = len(_nv_configs(args.quick))
    print(f"[cfg] models={ALL_MODELS}", flush=True)
    print(f"[cfg] data_root={DATA_ROOT} out_dir={out_root} "
          f"size={SPECTROGRAM_SIZE} ae_input={AE_INPUT_SIZE} "
          f"batch={BATCH_SIZE}", flush=True)
    if PAPER_SCHEDULE:
        print(f"[cfg] schedule: PAPER — SGD(m={FT_MOMENTUM}) + cosine x "
              f"{run_epochs} ep fixed, warmup {WARMUP_EPOCHS}, no early stop",
              flush=True)
    else:
        print(f"[cfg] schedule: train-until-overfitting — SGD(m={FT_MOMENTUM}),"
              f" ReduceLROnPlateau, early stop patience="
              f"{EARLY_STOPPING_PATIENCE}, budget={run_epochs} ep", flush=True)
    if USE_SAM:
        sam_status = f"ON, rho={SAM_RHO}"
    else:
        sam_status = ("off (the article does not use SAM at fine-tuning; "
                      "enable with --sam)")
    print(f"[cfg] SAM: {sam_status}", flush=True)
    print(f"[cfg] inference: softmax AND TTA+MC-dropout "
          f"({MC_DROPOUT_PASSES} passes, model dropout p={MODEL_DROPOUT})",
          flush=True)
    print(f"[cfg] configs per (model, band): {n_cfg}", flush=True)

    if args.profile_only:
        nv_profile(SPECTROGRAM_SIZE, out_root)
        return
    if args.aggregate:
        nv_aggregate(out_root, SPECTROGRAM_SIZE)
        nv_profile(SPECTROGRAM_SIZE, out_root)
        return

    if args.task_id is not None:
        mi, bi = divmod(args.task_id, len(NARVAL_BANDS))
        if mi >= len(ALL_MODELS):
            print(f"[!] task-id {args.task_id} out of range "
                  f"(max {len(ALL_MODELS) * len(NARVAL_BANDS) - 1}).")
            return
        pairs = [(ALL_MODELS[mi], NARVAL_BANDS[bi])]
    else:
        models = args.models or ALL_MODELS
        bands = args.bands or NARVAL_BANDS
        pairs = [(m, b) for m in models for b in bands]

    print(f"Running {len(pairs)} (model, band) pair(s).", flush=True)

    # ---- PREFLIGHT: verify every model's pretrained weights load BEFORE we
    # start any (multi-hour) training. Fails in seconds on an empty cache
    # instead of silently random-init training for days. ----
    if not ALLOW_RANDOM_INIT:
        print("[preflight] checking pretrained weights for all models …",
              flush=True)
        models_needed = sorted({m for m, _ in pairs})
        for mn in models_needed:
            try:
                _tmp = _build_backbone(mn, pretrained=True)
                del _tmp
                print(f"[preflight]   {mn}: OK", flush=True)
            except Exception as e:
                print(f"[preflight]   {mn}: FAILED", flush=True)
                raise
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
        print("[preflight] all weights present — starting training.", flush=True)

    for model_name, band_arg in pairs:
        nv_run_model_band(model_name, band_arg, out_root, args.quick,
                          run_epochs)
    print("\nTraining complete. Run with --aggregate to build the master "
          "comparison, metrics, and text confusion matrices.", flush=True)


if __name__ == "__main__":
    main()