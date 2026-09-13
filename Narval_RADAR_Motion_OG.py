"""
Narval_radar_motion_224.py
==========================
*** 224x224 VARIANT *** of Narval_RADAR_Motion.py.
Identical pipeline, but runs at 224x224 (the baseline resolution) instead of
384x384:  --size defaults to 224 and AE_INPUT_SIZE = (224, 224). Results go to
RADAR_Motion_Results_224/ so they don't overwrite the 384 run. Use this to
compare 224 (lossy baseline) vs the 384 run head-to-head. All models are
resolution-agnostic (AdaptiveAvgPool / sequence pooling), so no architecture
changes are needed — only the size constants differ.

radar_har_single.py
===================
Single-file pipeline for OpenRadar / CI4R-MULTI3 cross-frequency Human
Activity Recognition (11 classes: WLKT, WALKA, PICK, BEND, SIT, KNEEL,
CRWL, WTOES, LIMP, SHTEPS, SCSSR) across 10 / 24 / 77 GHz and combined.

Outputs (per model, per protocol) into ./results/<protocol>/:
  <model>_result.json   - hyperparams, history, y_true, y_pred, metrics
                          (re-usable input for replotting)
  <model>_cm.svg        - normalized confusion matrix  (vector)
  <model>_cm_raw.svg    - raw-count confusion matrix    (vector)
  <model>_curves.svg    - train/val loss + val acc      (vector)
  <model>_report.txt    - sklearn per-class report
  comparison.json + comparison.md  - ranked table across all 10 models

Modes:
    python radar_har_single.py train      [--bands ... --models ... --tune]
    python radar_har_single.py replot     [--results-dir ./results]
        # regenerates SVGs from saved JSON without retraining

Hyperparameters (already wired in):
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-7
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ====================================================================
# Narval-safe pretrained weight loader
# ====================================================================
def _safe_weights(builder_fn, weights_arg, **extra):
    """Try to load pretrained weights; fall back to random init with a clear
    warning if the compute node can't reach the internet.  On Narval, weights
    should be pre-cached on the login node first — see --download-weights."""
    try:
        return builder_fn(weights=weights_arg, **extra)
    except Exception as e:
        print(f"\n[WARNING] Cannot download pretrained weights: {e}"
              f"\n          Pre-cache them on the login node:"
              f"\n            python Narval_RADAR_Motion.py --download-weights"
              f"\n          Continuing with random (un-pretrained) weights."
              f"\n          Results will be WORSE without pretrained initialisation.\n",
              flush=True)
        return builder_fn(weights=None, **extra)


def _download_all_weights(torch_home=None):
    """Download all backbone weights the training uses. Run this ONCE on a
    Narval LOGIN NODE (which has internet) before submitting the array job.
    Weights land in torch_home (defaults to ~/.cache/torch) which is on the
    shared home NFS, visible from every compute node."""
    import os, torchvision.models as tvm
    if torch_home:
        os.makedirs(torch_home, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
    dst = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    print(f"Downloading all pretrained backbone weights to {dst} …", flush=True)
    downloads = [
        ("resnet18",      lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)),
        ("resnet50 V2",   lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)),
        ("convnext_tiny", lambda: tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)),
        ("swin_t",        lambda: tvm.swin_t(weights=tvm.Swin_T_Weights.DEFAULT)),
        ("densenet121",   lambda: tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)),
        ("inception_v3",  lambda: tvm.inception_v3(
                              weights=tvm.Inception_V3_Weights.IMAGENET1K_V1,
                              aux_logits=True, init_weights=False)),
    ]
    ok = 0
    for name, fn in downloads:
        try:
            m = fn(); del m
            print(f"  [OK]   {name}", flush=True); ok += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}", flush=True)
    print(f"\nDownloaded {ok}/{len(downloads)} weight files."
          f"\nNow submit the array job — compute nodes will load from {dst}", flush=True)


# ====================================================================
# 1. CONFIG
# ====================================================================
DATA_ROOT = Path(r"Data")
# On Linux/Colab override with e.g.: DATA_ROOT = Path("/content/Data")

FREQ_BANDS = ["10ghz", "24ghz", "77ghz"]

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

# Maximum files loaded per activity class per band folder.
# None = use ALL available files (recommended now that the folders have more
# data — more samples = more robust averaged results). Set to an int (e.g. 50)
# to cap per class for a quick balanced subset.
MAX_FILES_PER_ACT = None

# Signal processing
STFT_WINDOW = 256
STFT_OVERLAP = 200
STFT_NFFT = 512
SPECTROGRAM_SIZE = (224, 224)
MTI_FILTER = True
DB_FLOOR = -40
SAMPLING_RATES = {"10ghz": 1000, "24ghz": 1000, "77ghz": 1000}

# CONFIRMED via diagnose_data.py: CI4R cross-frequency files store a complex
# (freq x time) micro-Doppler matrix under variable 'sx1', shape ~(4096, 538).
# We take |sx1| -> dB -> normalize. Do NOT run another STFT on it.
#   "auto"        -> 2-D array (real or complex) = spectrogram; 1-D / Nx2 = raw IQ
#   "spectrogram" -> force: file is already a micro-Doppler matrix
#   "iq"          -> force: file is raw I/Q; apply STFT
DATA_FORMAT = "auto"   # auto correctly detects the sx1 2-D matrix as a spectrogram

# Dynamic range (dB below peak) kept when displaying a magnitude spectrogram.
DB_DYNAMIC_RANGE = 50

# Optional center-crop along the Doppler (frequency) axis to drop the noise-only
# top/bottom bands and concentrate on the signature. 1.0 = no crop. For the 77GHz
# sx1 data the signature sits well within the central ~60%, so 0.6 is a good,
# safe value that usually IMPROVES accuracy. Try 0.5-0.7.
DOPPLER_CROP_FRAC = 0.6

# ---- Band-specific Doppler energy regions (RAW pixel rows, applied BEFORE resize) ----
# VERIFIED against the provided spectrogram PNGs (Panel A = raw sx1, 4096 Doppler
# bins, averaged over all 11 activities per band):
#   10 GHz : single central band, peak ~2050, energy 1700-2700        -> CONFIRMED
#   24 GHz : THREE bands (Doppler wrapping at the higher carrier):
#            0-300 (bottom edge) + 1500-2500 (centre) + 3700-4096 (top) -> CONFIRMED
#   77 GHz : strong central torso line ~2050 with the micro-Doppler blob
#            spread ~1100-3000 (edges also show some energy)
# All bands have 4096 Doppler rows, so these row ranges apply to every band.
# Regions are symmetric about the centre, so the imshow origin convention does
# not change which rows are kept. Switch to BAND_REGION_MODE="auto" to detect
# per file, or run --diagnose-bands to re-measure.
BAND_REGION_MODE = "manual"          # "manual" | "auto" | "center"
# How manual regions are applied to the spectrogram:
#   "mask" : keep the FULL Doppler axis, zero out the non-signal rows. Preserves
#            spatial continuity & the physical Doppler positions — no artificial
#            edges. RECOMMENDED for multi-band cases (24 GHz) where concatenation
#            glued 3 non-adjacent bands and created discontinuities the CNN saw
#            as fake signal (a likely cause of the degradation you noticed).
#   "crop" : cut out the bands and CONCATENATE them (smaller image, signal fills
#            the frame at higher resolution, but introduces discontinuities for
#            multi-band frequencies). This was the previous behavior.
BAND_REGION_LAYOUT = "mask"
BAND_DOPPLER_REGIONS = {
   #"10ghz": [(1700, 2700)],                              # verified: signal 1758-2417 (68% energy)
   #"24ghz": [(0, 400), (1500, 2500), (3700, 4096)],      # verified: wrapped 3 bands (93% energy)
   #"77ghz": [(1100, 3000)],                              # verified: central signature
}
AUTO_REGION_DB       = 8.0      # keep rows within this many dB of the peak row-energy
AUTO_REGION_MIN_FRAC = 0.04     # ignore detected bands narrower than this frac of rows

# ---- Smart downscaling (overcomes the 4096x538 -> target bottleneck) ----
# "bilinear"     : baseline PIL bilinear (smears sharp micro-Doppler peaks)
# "mixed"        : alpha*max_pool + (1-alpha)*avg_pool (peak + energy balanced)
# "multichannel" : Ch0 = area-avg (energy context),
#                  Ch1 = mixed     (balanced features),
#                  Ch2 = max-pool  (peak detector)              [RECOMMENDED]
# "autoencoder"  : multichannel at AE_INPUT_SIZE -> a small learned conv stem
#                  refines it to SPECTROGRAM_SIZE, trained jointly (task-aware).
#                  AE_INPUT_SIZE defaults to == --size (384), so the stem learns
#                  to FUSE the avg/mix/max channels; set it larger to also downscale.
DOWNSCALE_MODE = "multichannel"
MIX_ALPHA      = 0.5
AE_INPUT_SIZE  = (224, 224)     # autoencoder-stem input size; matches --size (224)

# When COMPARE_DOWNSCALING is on, the downscaling strategy becomes an extra
# ablation axis: every config is trained under each method below, and the
# results include a head-to-head downscaling_comparison.md / .json.
DOWNSCALE_MODES_TO_COMPARE = ["mixed", "multichannel", "autoencoder"]
COMPARE_DOWNSCALING = True     # set by --compare-downscaling
_DS_TAG = {"bilinear": "bil", "mixed": "mix", "multichannel": "multi", "autoencoder": "ae"}

# Candidate variable names to look for inside .mat files. 'sx1' confirmed for
# this dataset; others kept as fallbacks for the 10/24 GHz bands.
MAT_VAR_CANDIDATES = ("sx1", "data", "rawData", "raw_data", "iq", "x", "signal",
                      "spectrogram", "spectro", "micro_doppler", "md", "S", "img")

# Per-model base learning rate. Pretrained transformers/heavy backbones need a
# much smaller LR than CNNs or they collapse to predicting one class.
MODEL_BASE_LR = {
    "01_CNN_LSTM":             1e-3,
    "02_CNN_BiLSTM_Attn":      1e-3,
    "03_ResNet18_Head":        1e-3,    # simple head can take full LR
    "04_ViT_GRU":              5e-5,
    "05_Swin_LSTM":            1e-4,
    "06_Contrastive_ResNet":   5e-4,   # contrastive + classification, gentle LR
    "07_ConvNeXt_Transformer": 1e-4,
    "08_CNN_Transformer_VAE":  5e-4,
    "09_Denoising_AE":         8e-4,   # from-scratch encoder, slightly higher LR
    "10_DenseNet_GRU":         5e-4,    # heavier pretrained backbone -> safer LR
}
# Backbone params train at BACKBONE_LR_MULT x the head LR (differential LR).
BACKBONE_LR_MULT = 0.1
WARMUP_EPOCHS = 3

# Training
LEARNING_RATE = 0.001
# Was 15 -> too tight for transformer-heavy models that overfit early. Giving
# them 25 epochs of patience lets their LR scheduler kick in before early stop.
EARLY_STOPPING_PATIENCE = 25
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-7
BATCH_SIZE = 32
EPOCHS = 200
# DataLoader workers. On Windows, >0 uses 'spawn' which pickles the dataset
# and can crash with "cannot pickle 'module' object" or EOFError. Default to 0
# on Windows for reliability; bump on Linux/Mac if you want speedup.
import platform as _platform
NUM_WORKERS = 0 if _platform.system() == "Windows" else 2
# Was 5e-4 — over-regularized the clean 77 GHz data and dropped accuracy by
# ~5 points. 2e-4 still regularizes the heavy models without crushing the
# easy ones. Tuned per evidence from the v1 run.
WEIGHT_DECAY = 2e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
# After training all 10 models, build a post-hoc ensemble from the top-K by
# averaging their softmax outputs (now WEIGHTED by validation accuracy).
ENSEMBLE_TOP_K = 5     # widened from 3 — more diversity helps once we weight
# Monte-Carlo Dropout passes during test inference. 1 = standard inference.
# 5-10 gives a meaningful uncertainty estimate at 5-10x the inference cost.
MC_DROPOUT_PASSES = 5

# ====================================================================
# WHAT HAPPENS WHEN YOU PRESS "Run Python File" IN VSCODE (no CLI args)
# --------------------------------------------------------------------
# "ablate" -> run the FULL 2x2x3 ablation grid (loss × physics × pretrain)
#             on the 8 models below, across all 4 bands. This is what you
#             want for the full comparison. ~384 trainings.
# "train"  -> run a normal single training pass over all models/bands.
# Change this one line to switch the default behavior.
# ====================================================================
DEFAULT_ACTION = "ablate"

# Default models for the ablation grid: 8 plain classifiers. We exclude
# 06_Contrastive_ResNet and 09_Denoising_AE because their built-in auxiliary
# losses are disabled under an explicit pretrain_mode, so they'd produce
# rows redundant with the plain models.
ABLATION_DEFAULT_MODELS = [
    "08_CNN_Transformer_VAE",
    "01_CNN_LSTM",
    "02_CNN_BiLSTM_Attn",
    "07_ConvNeXt_Transformer",
    "03_ResNet18_Head",
    "05_Swin_LSTM",
    "10_DenseNet_GRU",
]

# Splits
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

# Augmentation
AUG_TIME_MASK = 0.15        # was 0.20 — was hurting clean 77 GHz data
AUG_FREQ_MASK = 0.15        # was 0.20
AUG_NOISE_STD = 0.02        # was 0.03
AUG_MIXUP_ALPHA = 0.3       # was 0.4 — too strong, hurt clean bands
USE_MIXUP = True

# Optuna
TUNE_TRIALS = 30
TUNE_EPOCHS = 25
TUNE_SEARCH_SPACE = {
    "lr":           {"low": 1e-5, "high": 5e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
    "dropout":      {"low": 0.1,  "high": 0.6,  "log": False},
    "batch_size":   {"choices": [16, 32, 64]},
}

RESULTS_DIR = Path("./results")

# Where the ablation writes all its output (per-band tables, per-model detail,
# master comparison). Change this one line to relocate all ablation output.
ABLATION_RESULTS_DIR = Path(r"C:\Users\atikm\Downloads\Radar Motion Detection\Results")


# ====================================================================
# 2. DATA LOADING
# ====================================================================
def _normalize(s: str) -> str:
    return re.sub(r"[\s_]+", "_", s.strip().lower())

def _folder_to_class(folder_name: str) -> int:
    """Match by the 2-digit activity number embedded in the folder name."""
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
    """Return the most likely data array from a .mat file (raw, un-coerced).

    Supports both classic (<v7.3) and HDF5 (v7.3) .mat files.
    """
    import scipy.io as sio
    try:
        mat = sio.loadmat(str(path), squeeze_me=True)
        candidates = {k: v for k, v in mat.items()
                      if not k.startswith("__") and isinstance(v, np.ndarray)}
    except NotImplementedError:
        # v7.3 / HDF5 mat file
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

    # prefer a known variable name, else the largest array
    for name in MAT_VAR_CANDIDATES:
        if name in candidates and candidates[name].size > 100:
            return np.asarray(candidates[name]).squeeze()
    k = max(candidates, key=lambda kk: candidates[kk].size)
    return np.asarray(candidates[k]).squeeze()


def _load_dat_array(path: Path) -> np.ndarray:
    """Return raw array from a .dat file as complex IQ (int16 then float32)."""
    raw = np.fromfile(str(path), dtype=np.int16)
    if raw.size and raw.size % 2 == 0:
        iq = raw.astype(np.float32)
        return (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size and raw.size % 2 == 0:
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    raise ValueError(f"Unable to parse {path.name}")

def _mti(iq: np.ndarray) -> np.ndarray:
    if not MTI_FILTER: return iq
    from scipy.signal import butter, filtfilt
    b, a = butter(4, 0.02, btype="high")
    return filtfilt(b, a, iq.real) + 1j * filtfilt(b, a, iq.imag)

def _normalize_db_image(S: np.ndarray) -> np.ndarray:
    """dB-scale + min-max normalize + resize to SPECTROGRAM_SIZE, return [0,1] float."""
    from PIL import Image
    S = np.asarray(S, dtype=np.float64)
    S = 10 * np.log10(S + 1e-12) if np.all(S >= 0) else S  # already-power -> dB
    S = np.clip(S, S.max() + DB_FLOOR, S.max())
    S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    img = Image.fromarray((S * 255).astype(np.uint8)).resize(
        SPECTROGRAM_SIZE[::-1], Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def _iq_to_spectrogram(iq: np.ndarray, fs: int) -> np.ndarray:
    from scipy.signal import stft
    iq = _mti(iq.astype(np.complex64).ravel())
    _, _, Z = stft(iq, fs=fs, nperseg=STFT_WINDOW,
                   noverlap=STFT_OVERLAP, nfft=STFT_NFFT, return_onesided=False)
    Z = np.fft.fftshift(Z, axes=0)
    return _normalize_db_image(np.abs(Z) ** 2)

def _spectrogram_image(arr: np.ndarray, band: str = "77ghz") -> np.ndarray:
    """Turn an already-computed (freq x time) magnitude/complex matrix into a
    normalized image. Applies band-specific Doppler-region selection, then the
    configured smart-downscaling. Returns (H,W) for bilinear/mixed or
    (3,H,W) for multichannel/autoencoder."""
    mag = np.abs(np.asarray(arr, dtype=np.complex128))
    # CI4R sx1 is (freq=4096, time=538): frequency on the vertical (row) axis.
    if mag.ndim == 2:
        mag = _apply_band_regions(mag, band)
    db = 20 * np.log10(mag + 1e-9)
    db = np.clip(db, db.max() - DB_DYNAMIC_RANGE, db.max())
    norm = ((db - db.min()) / (db.max() - db.min() + 1e-9)).astype(np.float32)
    target = AE_INPUT_SIZE if DOWNSCALE_MODE == "autoencoder" else SPECTROGRAM_SIZE
    return _downscale(norm, target)


# ------------------------------------------------------------------
# Band-specific Doppler region selection (keep only signal-carrying rows)
# ------------------------------------------------------------------
def _apply_band_regions(mag: np.ndarray, band: str) -> np.ndarray:
    """Select the energy-carrying Doppler rows for this band, per BAND_REGION_MODE.
    BAND_REGION_LAYOUT decides HOW:
      "mask" -> keep full height, zero non-signal rows (no discontinuities)
      "crop" -> cut & concatenate the bands (smaller, but discontinuous)."""
    h = mag.shape[0]
    mode = BAND_REGION_MODE
    if mode == "auto":
        return _auto_band_crop(mag)
    if mode == "manual":
        regions = BAND_DOPPLER_REGIONS.get(band)
        if regions:
            clipped = [(max(0, min(h, int(lo))), max(0, min(h, int(hi))))
                       for lo, hi in regions]
            clipped = [(lo, hi) for lo, hi in clipped if hi > lo]
            if clipped:
                if BAND_REGION_LAYOUT == "mask":
                    # full height; suppress everything outside the signal bands
                    out = np.full_like(mag, float(mag.min()))
                    for lo, hi in clipped:
                        out[lo:hi] = mag[lo:hi]
                    return out
                # "crop": concatenate the bands (legacy; discontinuous for >1 band)
                return np.concatenate([mag[lo:hi] for lo, hi in clipped], axis=0)
    # "center" mode, or manual with no region for this band -> central crop
    if 0 < DOPPLER_CROP_FRAC < 1.0:
        keep = max(8, int(h * DOPPLER_CROP_FRAC)); lo = (h - keep) // 2
        return mag[lo:lo + keep]
    return mag


def _auto_band_crop(mag: np.ndarray) -> np.ndarray:
    """Detect signal bands automatically: keep rows whose mean energy is within
    AUTO_REGION_DB of the peak row-energy. Falls back to the central crop if the
    detected region is implausibly sparse."""
    row_e = 20 * np.log10(mag.mean(axis=1) + 1e-9)
    keep = row_e >= (row_e.max() - AUTO_REGION_DB)
    if keep.sum() < max(8, int(AUTO_REGION_MIN_FRAC * len(row_e))):
        h = len(row_e); k = max(8, int(h * DOPPLER_CROP_FRAC)); lo = (h - k) // 2
        return mag[lo:lo + k]
    return mag[keep]


# ------------------------------------------------------------------
# Smart downscaling (pooling-based; pure numpy/torch, picklable & cacheable)
# ------------------------------------------------------------------
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
    """Apply DOWNSCALE_MODE to a full-res normalized image.
    Returns (H,W) for bilinear/mixed, or (3,H,W) for multichannel/autoencoder."""
    mode = DOWNSCALE_MODE
    if mode == "mixed":
        return _mixed(norm, target, MIX_ALPHA).astype(np.float32)
    if mode in ("multichannel", "autoencoder"):
        ch0 = _area_avg(norm, target)            # energy context
        ch1 = _mixed(norm, target, MIX_ALPHA)    # balanced
        ch2 = _max_pool(norm, target)            # peak detector
        return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)
    # bilinear (default / fallback)
    from PIL import Image
    img = Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8)).resize(
        tuple(target)[::-1], Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def array_to_spectrogram(arr: np.ndarray, fs: int, band: str = "77ghz") -> np.ndarray:
    """Turn a loaded file array into a normalized spectrogram image, deciding
    whether it is an already-computed 2-D map or raw 1-D IQ that needs an STFT."""
    arr = np.asarray(arr).squeeze()
    fmt = DATA_FORMAT
    if fmt == "auto":
        if arr.ndim == 2 and np.iscomplexobj(arr) and arr.shape[-1] != 2:
            fmt = "spectrogram"
        elif arr.ndim == 2 and arr.shape[-1] != 2 and min(arr.shape) >= 8:
            fmt = "spectrogram"
        else:
            fmt = "iq"

    if fmt == "spectrogram":
        return _spectrogram_image(arr, band)

    # ---- raw IQ path (apply STFT) ----
    if arr.ndim == 2 and arr.shape[-1] == 2 and not np.iscomplexobj(arr):
        arr = arr[..., 0] + 1j * arr[..., 1]
    if arr.ndim > 1:
        arr = arr.sum(axis=0)          # collapse fast-time -> slow-time IQ
    return _iq_to_spectrogram(arr, fs)


# Global counters so silent parse failures become VISIBLE.
_LOAD_STATS = {"ok": 0, "failed": 0, "errors": {}}

def load_spectrogram(path: Path, band: str) -> np.ndarray:
    """Read a file and return a spectrogram image. On failure, record it and
    return zeros (so one bad file can't crash a long run) — but the failure is
    counted and reported by report_load_stats()."""
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
        # match the shape the current downscale mode would produce
        if DOWNSCALE_MODE in ("multichannel", "autoencoder"):
            tgt = AE_INPUT_SIZE if DOWNSCALE_MODE == "autoencoder" else SPECTROGRAM_SIZE
            return np.zeros((3, *tgt), dtype=np.float32)
        return np.zeros(SPECTROGRAM_SIZE, dtype=np.float32)

def report_load_stats():
    ok, failed = _LOAD_STATS["ok"], _LOAD_STATS["failed"]
    total = ok + failed
    if total == 0:
        return
    pct = 100.0 * failed / total
    print(f"\n[data] loaded {ok}/{total} files OK, {failed} FAILED ({pct:.1f}%).")
    if failed:
        print("[data] !! Failed files were replaced with BLANK images. "
              "This will cripple accuracy. Top errors:")
        for err, n in sorted(_LOAD_STATS["errors"].items(),
                             key=lambda kv: -kv[1])[:5]:
            print(f"        {n:5d}x  {err}")
        print("[data] -> Run diagnose_data.py and fix MAT_VAR_CANDIDATES / "
              "DATA_FORMAT / _load_dat_array accordingly.")

def index_files(root: Path, bands: List[str]) -> List[Tuple[Path, int, str]]:
    """Walk root, collect all .mat/.dat files, and apply MAX_FILES_PER_ACT so
    every (activity, band) combination contributes at most MAX_FILES_PER_ACT
    samples.  This keeps the dataset balanced across classes regardless of how
    many recordings exist per activity in the raw folders."""
    # Collect into per-(class, band) buckets first
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        band = _folder_to_band(folder.name)
        cid  = _folder_to_class(folder.name)
        if band not in bands or cid < 0: continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in (".mat", ".dat"):
                buckets[(cid, band)].append((f, cid, band))
    # Apply per-class cap then flatten
    entries = []
    for (cid, band), files in sorted(buckets.items()):
        if MAX_FILES_PER_ACT is not None:
            files = files[:MAX_FILES_PER_ACT]
        entries.extend(files)
    return entries


def _torch_imports():
    """Lazy-import torch so `replot` mode works without it installed."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    return torch, Dataset, DataLoader


class RadarSpectrogramDataset:
    """Picklable PyTorch-compatible dataset. Caches spectrograms after first read.

    IMPORTANT: This class must remain picklable for multi-worker DataLoaders on
    Windows (which use 'spawn'). That means: store NO module references, no open
    file handles, no torch tensors on self. Lazy-import torch inside __getitem__.
    """
    def __init__(self, entries, augment=False, cache=True):
        self.entries = entries
        self.augment = augment
        self.cache = cache
        self._cache = {}

    def __len__(self): return len(self.entries)

    def _augment(self, S):
        # S may be (H,W) or (3,H,W); apply the same masks across channels.
        if S.ndim == 3:
            _, H, W = S.shape
        else:
            H, W = S.shape
        tm = int(np.random.uniform(0, AUG_TIME_MASK) * W)
        if tm:
            t0 = np.random.randint(0, W - tm)
            S[..., :, t0:t0+tm] = 0
        fm = int(np.random.uniform(0, AUG_FREQ_MASK) * H)
        if fm:
            f0 = np.random.randint(0, H - fm)
            S[..., f0:f0+fm, :] = 0
        S = S + np.random.normal(0, AUG_NOISE_STD, S.shape).astype(np.float32)
        return np.clip(S, 0, 1)

    def __getitem__(self, idx):
        import torch  # local import — keeps the dataset object picklable
        path, cid, band = self.entries[idx]
        if self.cache and idx in self._cache:
            S = self._cache[idx].copy()
        else:
            S = load_spectrogram(path, band)
            if self.cache:
                self._cache[idx] = S.copy()
        if self.augment:
            S = self._augment(S)
        # multichannel/autoencoder modes already return (3,H,W); otherwise stack.
        if S.ndim == 3:
            x = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32))
        else:
            x = torch.from_numpy(np.stack([S, S, S], axis=0).astype(np.float32))
        return x, cid


def mixup(x, y, alpha=AUG_MIXUP_ALPHA, num_classes=NUM_CLASSES):
    import torch, torch.nn.functional as F
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[idx]
    y_oh = F.one_hot(y, num_classes).float()
    y_m = lam * y_oh + (1 - lam) * y_oh[idx]
    return x_m, y_m


# ====================================================================
# 3. MODELS (10 hybrid architectures)
# ====================================================================
def _build_models_module():
    """Builds the 10 models. Returns a dict {name: class}. Imports torch lazily."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tvm

    def _cnn_backbone(name, pretrained=True):
        if name == "resnet50":
            n = _safe_weights(tvm.resnet50, tvm.ResNet50_Weights.DEFAULT if pretrained else None)
            return nn.Sequential(*list(n.children())[:-2]), 2048
        if name == "resnet18":
            n = _safe_weights(tvm.resnet18, tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            return nn.Sequential(*list(n.children())[:-2]), 512
        if name == "efficientnet_b0":
            n = _safe_weights(tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            return n.features, 1280
        if name == "convnext_tiny":
            n = _safe_weights(tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            return n.features, 768
        if name == "mobilenet_v3_small":
            n = _safe_weights(tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            return n.features, 576
        raise ValueError(name)

    def _pos_encoding(L, D, device):
        pe = torch.zeros(L, D, device=device)
        pos = torch.arange(L, dtype=torch.float, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=device).float() * -(math.log(10000.0)/D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    class TxBlock(nn.Module):
        def __init__(self, d, h=8, ff=1024, p=0.3):
            super().__init__()
            self.a = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
            self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
            self.f = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p),
                                   nn.Linear(ff, d), nn.Dropout(p))
        def forward(self, x):
            a, _ = self.a(x, x, x, need_weights=False)
            x = self.n1(x + a); return self.n2(x + self.f(x))

    # ---- 1. CNN + LSTM ----------------------------------------------
    # CNN backbone collapses Doppler axis with .mean(dim=2), producing a
    # time-step sequence fed to the LSTM. AdaptiveAvgPool2d is not needed here:
    # .mean(dim=2) is inherently resolution-invariant. At 384x384 the ResNet18
    # backbone produces (B,512,12,12); mean over Doppler -> (B,512,12) -> 12
    # time-step sequence (vs 7 at 224). Same parameter count, more operations.
    class CNN_LSTM(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            self.cnn, feat = _cnn_backbone("resnet18")
            self.lstm = nn.LSTM(feat, 256, 2, batch_first=True, dropout=dropout)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES))
        def forward(self, x):
            f = self.cnn(x).mean(dim=2).permute(0, 2, 1)   # (B, W, feat)
            o, _ = self.lstm(f); return self.head(o[:, -1])

    # ---- 2. CNN + BiLSTM + Attention --------------------------------
    class CNN_BiLSTM_Attn(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            self.cnn, feat = _cnn_backbone("resnet18")
            self.bi = nn.LSTM(feat, 256, 2, batch_first=True, dropout=dropout, bidirectional=True)
            self.attn = nn.Linear(512, 1)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, NUM_CLASSES))
        def forward(self, x):
            f = self.cnn(x).mean(dim=2).permute(0, 2, 1)   # (B, W, feat)
            h, _ = self.bi(f); w = torch.softmax(self.attn(h), dim=1)
            return self.head((w * h).sum(dim=1))

    # ---- 3. ResNet18 + Linear Head -----------------------------------
    # nn.AdaptiveAvgPool2d((1, 1)) pools the CNN spatial output to a single
    # vector regardless of input resolution, so the FC layer dimensions are
    # fixed at (feat -> 256 -> NUM_CLASSES) whether input is 224 or 384.
    # Parameter count is unchanged; only FLOPs increase with resolution.
    class ResNet18_Head(nn.Module):
        def __init__(self, dropout=0.5):
            super().__init__()
            self.cnn, feat = _cnn_backbone("resnet18")
            self.pool = nn.AdaptiveAvgPool2d((1, 1))    # <- key: size-agnostic FC
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(feat, 256), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, NUM_CLASSES))
        def forward(self, x):
            return self.head(self.pool(self.cnn(x)))

    # ---- 4. ViT + GRU -----------------------------------------------
    class ViT_GRU(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
            vit.heads = nn.Identity()
            self.vit = vit
            self.gru = nn.GRU(768, 256, 2, batch_first=True, dropout=dropout, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, NUM_CLASSES))
        def forward(self, x):
            chunks = torch.chunk(x, 4, dim=-1)
            feats = [self.vit(F.interpolate(c, (224, 224), mode="bilinear", align_corners=False)) for c in chunks]
            h, _ = self.gru(torch.stack(feats, dim=1))
            return self.head(h[:, -1])

    # ---- 5. Swin + LSTM ---------------------------------------------
    class Swin_LSTM(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            swin = _safe_weights(tvm.swin_t, tvm.Swin_T_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(swin.children())[:-3])
            self.norm = nn.LayerNorm(768)
            self.lstm = nn.LSTM(768, 256, 2, batch_first=True, dropout=dropout, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, NUM_CLASSES))
        def forward(self, x):
            # Swin-T uses windowed attention with a relative-position bias tuned
            # for 224x224; it cannot exploit 512 the way the conv models do, so we
            # feed it at its native resolution. The LSTM still runs over Swin's
            # token sequence. (This is the honest choice for a fixed-res backbone.)
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            f = self.backbone(x)
            if f.dim() == 4 and f.shape[-1] == 768:
                seq = f.reshape(f.size(0), -1, 768)
            else:
                seq = f.flatten(2).permute(0, 2, 1)
            seq = self.norm(seq); h, _ = self.lstm(seq)
            return self.head(h[:, -1])

    # ---- 6. Contrastive ResNet18 (SimCLR-style aux loss) -------------
    # REPLACED SimpleCNN (43-69% range — no pretrain + too small).
    # ResNet18 backbone + classifier head + projection head trained jointly
    # with an NT-Xent contrastive loss between two augmented views of each
    # batch. The contrastive objective forces the encoder to learn invariant
    # features. Typical gain on small datasets: +2-5 points.
    class Contrastive_ResNet(nn.Module):
        def __init__(self, dropout=0.4, proj_dim=128, temperature=0.2):
            super().__init__()
            self.cnn, feat = _cnn_backbone("resnet18")
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(dropout),
                nn.Linear(feat, 256), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, NUM_CLASSES))
            self.projector = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat, 256), nn.GELU(),
                nn.Linear(256, proj_dim))
            self.temperature = temperature

        def _embed(self, x):
            return self.pool(self.cnn(x))

        def forward(self, x):
            return self.classifier(self._embed(x))

        def auxiliary_loss(self, x):
            """NT-Xent contrastive loss between two random augmentations of x."""
            # Lightweight on-GPU view augmentation
            def augment(t):
                # Random horizontal flip
                if torch.rand(1, device=t.device) < 0.5:
                    t = torch.flip(t, dims=[-1])
                # Random shift in time
                shift = int(torch.randint(-12, 13, (1,)).item())
                t = torch.roll(t, shifts=shift, dims=-1)
                # Random noise
                t = t + 0.03 * torch.randn_like(t)
                return t
            x1, x2 = augment(x), augment(x)
            z1 = F.normalize(self.projector(self._embed(x1)), dim=1)
            z2 = F.normalize(self.projector(self._embed(x2)), dim=1)
            z = torch.cat([z1, z2], dim=0)               # (2B, D)
            sim = z @ z.t() / self.temperature           # (2B, 2B)
            B = z1.size(0)
            mask = torch.eye(2 * B, dtype=torch.bool, device=x.device)
            sim.masked_fill_(mask, float("-inf"))
            # positive index: i <-> i+B
            tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(x.device)
            return F.cross_entropy(sim, tgt)

    # ---- 7. ConvNeXt + Transformer ----------------------------------
    # ConvNeXt-tiny backbone outputs (B, 768, H/32, W/32). At 384: 12x12 = 144
    # tokens; at 224: 7x7 = 49 tokens. flatten(2) passes ALL tokens to the
    # Transformer, which handles variable sequence length inherently.
    # AdaptiveAvgPool2d is NOT needed: the Transformer aggregates via .mean(1).
    # Same parameter count; 144 tokens at 384 vs 49 at 224 = more FLOPs.
    class ConvNeXt_Transformer(nn.Module):
        def __init__(self, dropout=0.3, n_layers=3):
            super().__init__()
            self.cnn, feat = _cnn_backbone("convnext_tiny")
            self.proj = nn.Linear(feat, 384)
            self.blocks = nn.ModuleList([TxBlock(384, 8, 1536, dropout) for _ in range(n_layers)])
            self.head = nn.Sequential(nn.LayerNorm(384), nn.Dropout(dropout),
                                      nn.Linear(384, NUM_CLASSES))
        def forward(self, x):
            seq = self.proj(self.cnn(x).flatten(2).permute(0, 2, 1))
            seq = seq + _pos_encoding(seq.size(1), seq.size(2), x.device)
            for blk in self.blocks: seq = blk(seq)
            return self.head(seq.mean(dim=1))

    # ---- 8. CNN + Transformer + VAE (GenAI hybrid) ------------------
    class _VAEEnc(nn.Module):
        def __init__(self, latent=64):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.mu = nn.Linear(128, latent); self.lv = nn.Linear(128, latent)
        def forward(self, x):
            h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
            return mu + torch.exp(0.5 * lv) * torch.randn_like(lv), mu, lv

    class _VAEDec(nn.Module):
        def __init__(self, latent=64):
            super().__init__()
            self.fc = nn.Linear(latent, 128 * 7 * 7)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())
        def forward(self, z):
            h = self.fc(z).view(-1, 128, 7, 7)
            return F.interpolate(self.dec(h), size=SPECTROGRAM_SIZE, mode="bilinear", align_corners=False)

    class CNN_Transformer_VAE(nn.Module):
        def __init__(self, dropout=0.3, latent=64):
            super().__init__()
            self.cnn, feat = _cnn_backbone("resnet18")
            self.proj = nn.Linear(feat, 256)        # feat=512, fixed by resnet18 backbone
            self.tx = TxBlock(256, 8, 1024, dropout)
            self.enc = _VAEEnc(latent); self.dec = _VAEDec(latent)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(256 + latent, NUM_CLASSES))
        def forward(self, x):
            # .mean(dim=2) collapses Doppler: adaptive, same params at 384 or 224
            f = self.tx(self.proj(self.cnn(x).mean(dim=2).permute(0, 2, 1))).mean(dim=1)
            z, mu, lv = self.enc(x); x_hat = self.dec(z)   # dec interpolates to SPECTROGRAM_SIZE
            return self.head(torch.cat([f, z], dim=1)), x_hat, mu, lv

    # ---- 9. Denoising Autoencoder + Classifier  ----------------------
    # REPLACED MobileNet_BiLSTM (always collapsed to 10-36%).
    # Encoder-decoder with joint classification + reconstruction loss. We feed
    # a NOISY version of the spectrogram and ask the model to reconstruct the
    # CLEAN one. This pushes the encoder to learn denoised, class-discriminative
    # features. Aux loss with weight 0.1 typically buys +1-3 points.
    class Denoising_AE(nn.Module):
        def __init__(self, dropout=0.4, latent=256, noise_std=0.10):
            super().__init__()
            self.noise_std = noise_std
            # Encoder: 4-stage CNN
            def enc_block(ci, co):
                return nn.Sequential(
                    nn.Conv2d(ci, co, 3, padding=1, bias=False),
                    nn.BatchNorm2d(co), nn.GELU(),
                    nn.Conv2d(co, co, 3, padding=1, bias=False),
                    nn.BatchNorm2d(co), nn.GELU(),
                    nn.MaxPool2d(2))
            self.encoder = nn.Sequential(
                enc_block(3, 32), enc_block(32, 64),
                enc_block(64, 128), enc_block(128, 256))   # -> (B,256,14,14)
            # Decoder
            def dec_block(ci, co):
                return nn.Sequential(
                    nn.ConvTranspose2d(ci, co, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(co), nn.GELU())
            self.decoder = nn.Sequential(
                dec_block(256, 128), dec_block(128, 64),
                dec_block(64, 32),   dec_block(32, 16),
                nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())
            # Classifier
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(256, 128), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, NUM_CLASSES))

        def forward(self, x):
            z = self.encoder(x)
            return self.head(z)

        def auxiliary_loss(self, x):
            """Reconstruction loss: corrupt x with noise, recover clean x."""
            x_noisy = (x + self.noise_std * torch.randn_like(x)).clamp(0, 1)
            z = self.encoder(x_noisy)
            x_hat = self.decoder(z)
            # decoder output matches input size (encoder /16, decoder x16 = net 1:1)
            return F.mse_loss(x_hat, x)

    # ---- 10. DenseNet121 + GRU --------------------------------------
    class DenseNet_GRU(nn.Module):
        def __init__(self, dropout=0.4):
            super().__init__()
            dn = _safe_weights(tvm.densenet121, tvm.DenseNet121_Weights.DEFAULT)
            self.cnn = dn.features
            self.proj = nn.Linear(1024, 256)     # 1024 fixed by densenet121 output channels
            self.gru = nn.GRU(256, 192, 2, batch_first=True,
                              dropout=dropout, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(384),
                                      nn.Dropout(dropout),
                                      nn.Linear(384, NUM_CLASSES))
        def forward(self, x):
            # F.relu + .mean(dim=2) collapses Doppler adaptively; at 384 the
            # time sequence is 12 steps (vs 7 at 224). Same params, more FLOPs.
            f = F.relu(self.cnn(x)).mean(dim=2).permute(0, 2, 1)   # (B, W, 1024)
            f = self.proj(f)
            h, _ = self.gru(f)
            return self.head(h[:, -1])

    return {
        "01_CNN_LSTM":             CNN_LSTM,
        "02_CNN_BiLSTM_Attn":      CNN_BiLSTM_Attn,
        "03_ResNet18_Head":        ResNet18_Head,
        "04_ViT_GRU":              ViT_GRU,
        "05_Swin_LSTM":            Swin_LSTM,
        "06_Contrastive_ResNet":   Contrastive_ResNet,
        "07_ConvNeXt_Transformer": ConvNeXt_Transformer,
        "08_CNN_Transformer_VAE":  CNN_Transformer_VAE,
        "09_Denoising_AE":         Denoising_AE,
        "10_DenseNet_GRU":         DenseNet_GRU,
    }


# ====================================================================
# 4. TRAINING
# ====================================================================
def _vae_loss(x, x_hat, mu, lv, beta=0.001):
    import torch.nn.functional as F
    import torch
    rec = F.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return rec + beta * kld


# ====================================================================
# Evidential Deep Learning (Sensoy et al., NeurIPS 2018)
# Replaces softmax with a Dirichlet evidence head. Each model's K logits
# become evidence values: alpha = softplus(logits) + 1, then a Dirichlet(alpha)
# distribution over the K-simplex. This gives a principled per-sample
# uncertainty (high uncertainty = small alphas / concentration close to uniform).
# Loss: expected squared error + KL-to-uniform regularizer on incorrect classes.
# ====================================================================
def _kl_dirichlet_to_uniform(alpha, K):
    """KL(Dir(alpha) || Dir(1,...,1))."""
    import torch
    S = alpha.sum(dim=1, keepdim=True)
    K_t = torch.tensor(float(K), device=alpha.device)
    lg_S = torch.lgamma(S).squeeze(1)
    lg_K = torch.lgamma(K_t)
    lg_alpha_sum = torch.lgamma(alpha).sum(dim=1)
    first = lg_S - lg_K - lg_alpha_sum
    second = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return first + second

def edl_loss(logits, y_oh, num_classes, epoch=0, max_anneal_epochs=10, lam=0.1):
    """Evidential Deep Learning loss with KL annealing. y_oh is one-hot OR soft
    labels (e.g. from mixup)."""
    import torch.nn.functional as F
    import torch
    evidence = F.softplus(logits)                 # ≥ 0
    alpha = evidence + 1                          # ≥ 1
    S = alpha.sum(dim=1, keepdim=True)
    # Expected MSE between Dirichlet mean and target + variance penalty
    err = (y_oh - alpha / S) ** 2
    var = alpha * (S - alpha) / (S ** 2 * (S + 1))
    data_term = (err + var).sum(dim=1).mean()
    # KL on wrong-class evidence: pull alpha for wrong classes towards 1 (uniform).
    alpha_tilde = y_oh + (1 - y_oh) * alpha
    kl = _kl_dirichlet_to_uniform(alpha_tilde, num_classes).mean()
    annealing = min(1.0, max(epoch, 0) / max(max_anneal_epochs, 1))
    return data_term + annealing * lam * kl

def edl_predict_probs(logits):
    """Class probabilities from EDL: alpha / S."""
    import torch.nn.functional as F
    alpha = F.softplus(logits) + 1
    return alpha / alpha.sum(dim=1, keepdim=True)


# ====================================================================
# Physics-Informed Doppler-symmetry consistency
# For cyclic human motion (walking, gait), the micro-Doppler spectrogram has
# approximate symmetry across the zero-Doppler axis. Encourage the model to
# give consistent predictions when we flip the spectrogram vertically (Doppler
# axis). This is a soft prior — it's only approximately true — so we keep
# the weight small.
# ====================================================================
def physics_doppler_symmetry_loss(model, x, is_vae=False):
    import torch
    import torch.nn.functional as F
    # Get model logits on original x (frozen target)
    with torch.no_grad():
        lg_orig = model(x)
        if isinstance(lg_orig, tuple): lg_orig = lg_orig[0]
        target = F.softmax(lg_orig, dim=1)
    x_flip = torch.flip(x, dims=[-2])              # flip Doppler (vertical) axis
    lg_flip = model(x_flip)
    if isinstance(lg_flip, tuple): lg_flip = lg_flip[0]
    return F.kl_div(F.log_softmax(lg_flip, dim=1), target, reduction="batchmean")


# ====================================================================
# Model-agnostic consistency losses (the "pretrain" toggles for the ablation)
# These work on ANY classifier — they just regularize the model to be
# invariant to specific perturbations:
#   - contrastive: predictions match between two augmented views (flip/shift/noise)
#   - denoising:   predictions match between clean input and noised input
# Both are sometimes called "consistency regularization" in the SSL literature.
# ====================================================================
def _grab_logits(out):
    return out[0] if isinstance(out, tuple) else out

def consistency_contrastive_loss(model, x):
    """Symmetric KL between predictions on two augmented views of x.
    Works for any classifier (no need for explicit feature heads)."""
    import torch
    import torch.nn.functional as F
    def aug(t):
        if torch.rand(1, device=t.device).item() < 0.5:
            t = torch.flip(t, dims=[-1])
        shift = int(torch.randint(-12, 13, (1,)).item())
        t = torch.roll(t, shifts=shift, dims=-1)
        t = t + 0.03 * torch.randn_like(t)
        return t.clamp(0, 1)
    x1, x2 = aug(x), aug(x)
    lg1 = _grab_logits(model(x1))
    lg2 = _grab_logits(model(x2))
    p1 = F.softmax(lg1, dim=1); p2 = F.softmax(lg2, dim=1)
    kl12 = F.kl_div(p1.clamp(min=1e-9).log(), p2, reduction="batchmean")
    kl21 = F.kl_div(p2.clamp(min=1e-9).log(), p1, reduction="batchmean")
    return 0.5 * (kl12 + kl21)

def consistency_denoising_loss(model, x, noise_std=0.10):
    """Predictions on noisy input should match predictions on clean input.
    Clean prediction is detached (used as soft target)."""
    import torch
    import torch.nn.functional as F
    x_noisy = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
    with torch.no_grad():
        target = F.softmax(_grab_logits(model(x)), dim=1)
    lg_noisy = _grab_logits(model(x_noisy))
    return F.kl_div(F.log_softmax(lg_noisy, dim=1), target, reduction="batchmean")


def train_one_model(model_cls, model_name, train_ds, val_ds, test_ds,
                    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    dropout=0.3, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    verbose=True,
                    loss_type="ce",                # "ce" or "edl"
                    use_physics=False,             # add Doppler-symmetry aux loss
                    use_temperature=True,          # post-hoc temperature scaling
                    mc_passes=MC_DROPOUT_PASSES,   # MC dropout passes at test
                    pretrain_mode="auto"):         # "auto"/"none"/"contrastive"/"denoising"
    """Train one model with optional Evidential head, Doppler-symmetry physics
    prior, temperature scaling, MC Dropout inference, and pretrain-style aux loss.

    pretrain_mode semantics:
      "auto"        -> use built-in aux loss if model has .auxiliary_loss() / is VAE
      "none"        -> NO auxiliary loss (override built-in)
      "contrastive" -> consistency_contrastive_loss applied to ANY model (override built-in)
      "denoising"   -> consistency_denoising_loss   applied to ANY model (override built-in)
    """
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:    model = model_cls(dropout=dropout)
    except TypeError: model = model_cls()
    model = model.to(device)

    # The VAE model's forward() ALWAYS returns a tuple (logits, x_hat, mu, lv),
    # regardless of whether we use its auxiliary loss. We must track these two
    # facts separately:
    #   model_returns_tuple -> forward() yields a tuple; unwrap logits before use
    #   use_vae_aux         -> add the VAE reconstruction loss (only in "auto" mode)
    model_returns_tuple = model_name.endswith("VAE")
    use_vae_aux = model_returns_tuple and pretrain_mode == "auto"
    # Kept for backward-compat references below
    is_vae = model_returns_tuple

    def fwd_logits(xx):
        """Forward pass returning ONLY the class logits (unwraps tuple outputs)."""
        out = model(xx)
        return out[0] if isinstance(out, tuple) else out

    train_loader = DataLoader(_DS(train_ds), batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(_DS(val_ds),   batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(_DS(test_ds),  batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ---- Differential learning rate -------------------------------------
    # Pretrained backbones (cnn/vit/swin/s1/s2/backbone) train at a fraction of
    # the head LR. This is critical: a flat 1e-3 across a pretrained transformer
    # destroys its features and collapses the model to one class.
    backbone_keys = ("cnn", "vit", "backbone", "s1", "s2")
    bb_params, head_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if any(pname.startswith(k) or f".{k}." in pname
                          or pname.split(".")[0] in backbone_keys
                          for k in backbone_keys) else head_params).append(p)

    param_groups = [{"params": head_params, "lr": lr}]
    if bb_params:
        param_groups.append({"params": bb_params, "lr": lr * BACKBONE_LR_MULT})

    optim = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optim, mode="min",
                                  patience=LR_SCHEDULER_PATIENCE,
                                  factor=LR_SCHEDULER_FACTOR, min_lr=MIN_LR)

    base_lrs = [g["lr"] for g in optim.param_groups]
    def apply_warmup(epoch):
        """Linear LR warmup for the first WARMUP_EPOCHS epochs."""
        if epoch <= WARMUP_EPOCHS and WARMUP_EPOCHS > 0:
            scale = epoch / WARMUP_EPOCHS
            for g, base in zip(optim.param_groups, base_lrs):
                g["lr"] = base * scale
    eval_crit = nn.CrossEntropyLoss()
    if USE_MIXUP:
        criterion = lambda logits, ys: -(ys * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val = float("inf"); best_state = None
    patience_left = EARLY_STOPPING_PATIENCE
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}

    # Detect a model-defined auxiliary loss (e.g. Contrastive_ResNet, Denoising_AE).
    has_builtin_aux = hasattr(model, "auxiliary_loss")
    aux_weight = 0.1   # weight for any auxiliary / consistency loss

    # When pretrain_mode is explicit, IGNORE built-in aux losses so the ablation
    # comparison is clean — every model gets the same regularization.
    # NOTE: this does NOT change the fact that the VAE forward() returns a tuple;
    # we still unwrap logits via fwd_logits() everywhere.
    use_builtin_aux = has_builtin_aux and pretrain_mode == "auto"

    for ep in range(1, epochs + 1):
        apply_warmup(ep)
        # train
        model.train(); tot, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)

            def _compute_classification_loss(logits, y_hard, y_soft_or_oh):
                """CE OR EDL on logits. y_hard for plain CE, y_soft_or_oh for mixup/EDL."""
                if loss_type == "edl":
                    return edl_loss(logits, y_soft_or_oh, NUM_CLASSES, epoch=ep)
                return criterion(logits, y_soft_or_oh) if USE_MIXUP else criterion(logits, y_hard)

            if USE_MIXUP:
                x_in, y_target = mixup(x, y)
            else:
                x_in, y_target = x, F.one_hot(y, NUM_CLASSES).float()

            out = model(x_in)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
            loss = _compute_classification_loss(logits, y, y_target)

            # VAE reconstruction aux loss (only in auto mode for the VAE model)
            if use_vae_aux and isinstance(out, tuple) and len(out) >= 4:
                _, xh, mu, lv = out[0], out[1], out[2], out[3]
                loss = loss + aux_weight * _vae_loss(x_in, xh, mu, lv)
            # Other models' built-in aux loss (contrastive/denoising architectures)
            elif use_builtin_aux:
                loss = loss + aux_weight * model.auxiliary_loss(x)

            # Physics-informed Doppler-symmetry aux loss (light weight)
            if use_physics:
                loss = loss + 0.05 * physics_doppler_symmetry_loss(model, x)

            # Pretrain-mode aux loss (model-agnostic consistency regularization)
            if pretrain_mode == "contrastive":
                loss = loss + aux_weight * consistency_contrastive_loss(model, x)
            elif pretrain_mode == "denoising":
                loss = loss + aux_weight * consistency_denoising_loss(model, x)

            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        train_loss = tot / max(n, 1)

        # val
        model.eval(); vl, vn, correct = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = fwd_logits(x)
                if loss_type == "edl":
                    # NLL of Dirichlet mean for fair comparison across loss types
                    probs = edl_predict_probs(logits)
                    vl += F.nll_loss(torch.log(probs.clamp(min=1e-9)), y).item() * x.size(0)
                else:
                    vl += eval_crit(logits, y).item() * x.size(0)
                vn += x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        val_loss = vl / max(vn, 1); val_acc = correct / max(vn, 1)
        if ep > WARMUP_EPOCHS:
            scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optim.param_groups[0]["lr"])

        if verbose:
            print(f"[{model_name}] ep {ep:3d}  tr={train_loss:.4f}  "
                  f"val={val_loss:.4f}  acc={val_acc:.4f}  "
                  f"lr={optim.param_groups[0]['lr']:.2e}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose: print(f"[{model_name}] early stop @ ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Temperature scaling: calibrate the softmax using val set --------
    # Train a single scalar T so that softmax(logits/T) is well-calibrated
    # on the validation set. This improves both individual model accuracy
    # (by sharpening or smoothing predictions appropriately) and especially
    # ensemble accuracy (since we're averaging softmax across models).
    def _fit_temperature():
        # Collect logits on val set
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                lg = fwd_logits(x)
                all_logits.append(lg); all_targets.append(y)
        logits = torch.cat(all_logits); targets = torch.cat(all_targets)
        T = nn.Parameter(torch.ones(1, device=device) * 1.5)
        opt = torch.optim.LBFGS([T], lr=0.05, max_iter=80)
        ce = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad()
            loss = ce(logits / T.clamp(min=1e-2), targets)
            loss.backward()
            return loss
        opt.step(closure)
        return float(T.detach().clamp(min=0.1, max=10.0).item())
    # Temperature scaling is only meaningful for softmax/CE models; skip for EDL.
    if use_temperature and loss_type == "ce":
        try:
            temperature = _fit_temperature()
            if verbose: print(f"[{model_name}] fitted temperature T={temperature:.3f}")
        except Exception:
            temperature = 1.0
    else:
        temperature = 1.0

    # ---- Test inference: TTA + (optional) MC Dropout + Temperature ------
    # - For CE models: softmax(logits / T)
    # - For EDL models: alpha/S (Dirichlet mean, no temperature)
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
                lg = fwd_logits(xi)
                probs_sum = probs_sum + _probs_from_logits(lg)
                count += 1
        return probs_sum / count

    model.eval()
    if mc_passes > 1:
        _enable_mc_dropout(model)   # keep dropout active for MC sampling

    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            probs = _tta_mc_probs(x, n_mc=mc_passes)
            all_y.append(y.numpy())
            all_p.append(probs.argmax(1).cpu().numpy())
            all_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
    test_probs = np.concatenate(all_prob, axis=0)
    test_acc = float((y_true == y_pred).mean())

    return {
        "model_name": model_name,
        "hyperparams": {"lr": lr, "weight_decay": weight_decay,
                        "dropout": dropout, "batch_size": batch_size,
                        "epochs_run": len(history["train_loss"])},
        "history": history,
        "best_val_loss": float(best_val),
        "test_acc": test_acc,
        "val_acc": float(max(history["val_acc"])) if history["val_acc"] else 0.0,
        "temperature": temperature,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "test_probs": test_probs.tolist(),    # for post-hoc ensembling
        "_state_dict": best_state,            # for .pth checkpointing (popped before JSON)
    }


class _DS:
    """Thin torch Dataset wrapper around RadarSpectrogramDataset."""
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i): return self.ds[i]


def tune_model(model_cls, model_name, train_ds, val_ds, n_trials=TUNE_TRIALS):
    import optuna
    def objective(trial):
        sp = TUNE_SEARCH_SPACE
        lr   = trial.suggest_float("lr",           sp["lr"]["low"],   sp["lr"]["high"],   log=True)
        wd   = trial.suggest_float("weight_decay", sp["weight_decay"]["low"], sp["weight_decay"]["high"], log=True)
        do   = trial.suggest_float("dropout",      sp["dropout"]["low"], sp["dropout"]["high"])
        bs   = trial.suggest_categorical("batch_size", sp["batch_size"]["choices"])
        r = train_one_model(model_cls, model_name, train_ds, val_ds, val_ds,
                            lr=lr, weight_decay=wd, dropout=do,
                            batch_size=bs, epochs=TUNE_EPOCHS, verbose=False)
        return r["best_val_loss"]
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


# ====================================================================
# 5. EVALUATION + SVG PLOTTING
# ====================================================================
def _plot_imports():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def plot_confusion_matrix_svg(y_true, y_pred, class_names, out_svg: Path,
                              normalize=True, title=""):
    from sklearn.metrics import confusion_matrix
    plt = _plot_imports()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        d = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1); fmt = ".2f"
    else:
        d = cm; fmt = "d"
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(d, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label",
           title=title or "Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thr = d.max() / 2.0
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, format(d[i, j], fmt), ha="center", va="center",
                    color="white" if d[i, j] > thr else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)

def plot_training_curves_svg(history, out_svg: Path, title=""):
    plt = _plot_imports()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} loss"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].plot(history["val_acc"], color="green", label="val acc")
    axes[1].set_title(f"{title} val accuracy"); axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)

def evaluate_and_save(result: dict, class_names, out_dir: Path):
    """Persist JSON + SVGs + per-class report. Returns short summary."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result["model_name"]
    y_true = np.array(result["y_true"]); y_pred = np.array(result["y_pred"])

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4, zero_division=0)
    print(f"\n=== {name} ===\nTest acc: {acc:.4f}   Macro F1: {macro_f1:.4f}\n{report}")

    # SVGs
    plot_confusion_matrix_svg(y_true, y_pred, class_names,
                              out_dir / f"{name}_cm.svg",
                              normalize=True, title=f"{name} (normalized)")
    plot_confusion_matrix_svg(y_true, y_pred, class_names,
                              out_dir / f"{name}_cm_raw.svg",
                              normalize=False, title=f"{name} (counts)")
    plot_training_curves_svg(result["history"],
                             out_dir / f"{name}_curves.svg",
                             title=name)

    # Persist full result JSON (re-usable for replotting)
    full = {**result,
            "metrics": {"test_acc": acc, "macro_f1": macro_f1},
            "class_names": class_names}
    with open(out_dir / f"{name}_result.json", "w") as f:
        json.dump(full, f, indent=2)
    with open(out_dir / f"{name}_report.txt", "w") as f:
        f.write(report)

    return {"model": name, "test_acc": acc, "macro_f1": macro_f1,
            "best_val_loss": result["best_val_loss"],
            "epochs_trained": len(result["history"]["train_loss"]),
            "hyperparams": result["hyperparams"]}


def write_comparison(summaries, out_dir: Path):
    summaries = sorted(summaries, key=lambda s: -s["test_acc"])
    md = ["| Rank | Model | Test Acc | Macro F1 | Epochs | LR | Dropout |",
          "|------|-------|----------|----------|--------|------|---------|"]
    for i, s in enumerate(summaries, 1):
        hp = s["hyperparams"]
        md.append(f"| {i} | {s['model']} | {s['test_acc']:.4f} | {s['macro_f1']:.4f} | "
                  f"{s['epochs_trained']} | {hp['lr']:.1e} | {hp['dropout']:.2f} |")
    (out_dir / "comparison.md").write_text("\n".join(md))
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\n".join(md))


# ====================================================================
# 6. REPLOT MODE — regenerate SVGs from JSON without retraining
# ====================================================================
def replot_from_json(results_root: Path):
    """Walks ./results/ for *_result.json, rebuilds all SVGs."""
    n = 0
    for jp in results_root.rglob("*_result.json"):
        with open(jp) as f:
            r = json.load(f)
        out_dir = jp.parent; name = r["model_name"]
        cls_names = r.get("class_names", CLASS_NAMES)
        plot_confusion_matrix_svg(np.array(r["y_true"]), np.array(r["y_pred"]),
                                  cls_names, out_dir / f"{name}_cm.svg",
                                  normalize=True, title=f"{name} (normalized)")
        plot_confusion_matrix_svg(np.array(r["y_true"]), np.array(r["y_pred"]),
                                  cls_names, out_dir / f"{name}_cm_raw.svg",
                                  normalize=False, title=f"{name} (counts)")
        plot_training_curves_svg(r["history"],
                                 out_dir / f"{name}_curves.svg", title=name)
        n += 1
        print(f"replotted: {jp}")
    print(f"\nDone. Regenerated SVGs for {n} model results.")


# ====================================================================
# 7. ORCHESTRATION
# ====================================================================
def stratified_split(entries, seed=RANDOM_SEED):
    from sklearn.model_selection import train_test_split
    labels = np.array([e[1] for e in entries]); idx = np.arange(len(entries))
    idx_trv, idx_te, y_trv, _ = train_test_split(
        idx, labels, test_size=TEST_SIZE, stratify=labels, random_state=seed)
    val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
    idx_tr, idx_va = train_test_split(
        idx_trv, test_size=val_rel, stratify=y_trv, random_state=seed)
    return ([entries[i] for i in idx_tr],
            [entries[i] for i in idx_va],
            [entries[i] for i in idx_te])

def run_protocol(bands, protocol_name, model_names, do_tune, model_registry,
                 train_kwargs=None):
    train_kwargs = train_kwargs or {}
    print(f"\n{'='*70}\nProtocol: {protocol_name}\n{'='*70}")
    entries = index_files(DATA_ROOT, bands)
    print(f"Found {len(entries)} files for {protocol_name}")
    if not entries:
        print("No files indexed - check DATA_ROOT in config section."); return []
    print("Class distribution:", Counter(e[1] for e in entries))

    train_e, val_e, test_e = stratified_split(entries)
    print(f"split: train={len(train_e)} val={len(val_e)} test={len(test_e)}")

    train_ds = RadarSpectrogramDataset(train_e, augment=True,  cache=True)
    val_ds   = RadarSpectrogramDataset(val_e,   augment=False, cache=True)
    test_ds  = RadarSpectrogramDataset(test_e,  augment=False, cache=True)

    out_dir = RESULTS_DIR / protocol_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    # Stash per-model soft probabilities for the ensemble step at the end.
    ensemble_inputs = []

    for mi, mname in enumerate(model_names):
        t0 = time.time()
        cls = model_registry[mname]
        if do_tune:
            print(f"\n--- HP tuning {mname} ---")
            best = tune_model(cls, mname, train_ds, val_ds)
            print(f"Best HPs: {best}")
        else:
            best = {"lr": MODEL_BASE_LR.get(mname, LEARNING_RATE),
                    "weight_decay": WEIGHT_DECAY,
                    "dropout": 0.3, "batch_size": BATCH_SIZE}

        print(f"\n--- training {mname} on {protocol_name}  (base lr={best['lr']:.1e}) ---")
        result = train_one_model(cls, mname, train_ds, val_ds, test_ds,
                                 lr=best["lr"], weight_decay=best["weight_decay"],
                                 dropout=best["dropout"], batch_size=best["batch_size"],
                                 **train_kwargs)
        # After the first model, the dataset cache is populated -> report parse health.
        if mi == 0:
            report_load_stats()
        s = evaluate_and_save(result, CLASS_NAMES, out_dir)
        s["protocol"] = protocol_name
        s["elapsed_sec"] = round(time.time() - t0, 1)
        summaries.append(s)
        # Keep soft predictions in memory for the ensemble.
        ensemble_inputs.append({
            "model": mname,
            "test_acc": s["test_acc"],
            "val_acc": result.get("val_acc", s["test_acc"]),
            "y_true": np.array(result["y_true"]),
            "probs": np.array(result["test_probs"]),
        })
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- POST-HOC ENSEMBLE -----------------------------------------------
    # Pick top-K individual models by VAL accuracy (NOT test acc — that would
    # be selection bias / leakage), then combine their per-sample softmax
    # predictions with weights proportional to their val accuracies. Models
    # have already been temperature-calibrated in train_one_model, so the
    # softmaxes are comparable across architectures.
    # Typical lift: +2 to +5 points absolute over the best single model.
    K = min(ENSEMBLE_TOP_K, len(ensemble_inputs))
    if K >= 2:
        # Rank by val_acc to avoid peeking at test
        ranked = sorted(ensemble_inputs, key=lambda e: -e.get("val_acc", 0))
        top = ranked[:K]
        # Weighted average: weights = softmax(val_acc / temperature)
        val_accs = np.array([t.get("val_acc", t["test_acc"]) for t in top])
        # Sharpen weights so the best model dominates a bit but the ensemble
        # still gets diversity. tau=0.05 makes a +5pt val_acc gap give ~3x weight.
        tau = 0.05
        w = np.exp(val_accs / tau); w = w / w.sum()
        print(f"\n[ensemble] member weights (sorted by val_acc):")
        for t, wi in zip(top, w):
            print(f"  {t['model']:28s}  val_acc={t.get('val_acc', 0):.3f}  weight={wi:.3f}")
        y_true = top[0]["y_true"]
        avg_probs = sum(wi * t["probs"] for wi, t in zip(w, top))
        y_pred_ens = avg_probs.argmax(axis=1)
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        acc_ens = float(accuracy_score(y_true, y_pred_ens))
        f1_ens  = float(f1_score(y_true, y_pred_ens, average="macro"))
        ens_name = f"00_Ensemble_w{K}"
        print(f"\n=== {ens_name} ===")
        print(f"Test acc: {acc_ens:.4f}   Macro F1: {f1_ens:.4f}")

        # Render the ensemble's CM + report and save a result JSON
        plot_confusion_matrix_svg(y_true, y_pred_ens, CLASS_NAMES,
                                  out_dir / f"{ens_name}_cm.svg",
                                  normalize=True, title=f"{ens_name} (normalized)")
        plot_confusion_matrix_svg(y_true, y_pred_ens, CLASS_NAMES,
                                  out_dir / f"{ens_name}_cm_raw.svg",
                                  normalize=False, title=f"{ens_name} (counts)")
        rep = classification_report(y_true, y_pred_ens, target_names=CLASS_NAMES,
                                    digits=4, zero_division=0)
        (out_dir / f"{ens_name}_report.txt").write_text(rep)
        with open(out_dir / f"{ens_name}_result.json", "w") as f:
            json.dump({"model_name": ens_name, "members": [t["model"] for t in top],
                       "y_true": y_true.tolist(),
                       "y_pred": y_pred_ens.tolist(),
                       "test_probs": avg_probs.tolist(),
                       "metrics": {"test_acc": acc_ens, "macro_f1": f1_ens},
                       "class_names": CLASS_NAMES}, f, indent=2)
        summaries.append({"model": ens_name, "test_acc": acc_ens, "macro_f1": f1_ens,
                          "best_val_loss": float("nan"),
                          "epochs_trained": 0,
                          "hyperparams": {"lr": 0, "weight_decay": 0,
                                          "dropout": 0, "batch_size": 0},
                          "protocol": protocol_name, "elapsed_sec": 0})

    write_comparison(summaries, out_dir)
    return summaries


def cmd_train(args):
    np.random.seed(RANDOM_SEED)
    try:
        import torch
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
    except ImportError:
        print("torch not installed — install it before running 'train'.")
        return

    registry = _build_models_module()
    model_names = list(registry.keys()) if args.models == ["all"] else args.models

    train_kwargs = {
        "loss_type":       getattr(args, "loss", "ce"),
        "use_physics":     getattr(args, "physics_informed", False),
        "mc_passes":       getattr(args, "mc_passes", MC_DROPOUT_PASSES),
        "use_temperature": not getattr(args, "no_temperature", False),
    }
    print(f"[info] training options: {train_kwargs}")

    all_s = []
    for b in args.bands:
        bands = ["10ghz", "24ghz", "77ghz"] if b == "combined" else [b]
        all_s.extend(run_protocol(bands, b, model_names, args.tune, registry,
                                  train_kwargs=train_kwargs))

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_DIR / "all_protocols_summary.json", "w") as f:
        json.dump(all_s, f, indent=2)
    print(f"\nDone. Results in {RESULTS_DIR.resolve()}")


def cmd_replot(args):
    replot_from_json(Path(args.results_dir))


# ====================================================================
# ABLATION: run the 12-config grid on each (model, band) the user picks
# ====================================================================
def cmd_ablate(args):
    """Run the 2x2x3 ablation grid:
        loss     in {ce, edl}
        physics  in {off, on}
        pretrain in {none, contrastive, denoising}
    on every (model, band) combination requested.

    For each (model, band): 12 trainings, then a ranked comparison table.
    """
    np.random.seed(RANDOM_SEED)
    import torch
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

    registry = _build_models_module()
    model_names = list(registry.keys()) if args.models == ["all"] else args.models
    bands_to_run = args.bands

    # Generate the 12 configs
    if args.quick:
        # 4 configs: loss × physics, no pretrain
        configs = [{"loss": l, "physics": p, "pretrain": "none"}
                   for l in ("ce", "edl") for p in (False, True)]
    else:
        configs = [{"loss": l, "physics": p, "pretrain": pt}
                   for l in ("ce", "edl")
                   for p in (False, True)
                   for pt in ("none", "contrastive", "denoising")]

    n_trainings = len(configs) * len(model_names) * len(bands_to_run)
    est_min = n_trainings * 10
    print(f"\n{'='*70}")
    print(f"ABLATION: {len(model_names)} models × {len(bands_to_run)} bands × "
          f"{len(configs)} configs = {n_trainings} trainings")
    print(f"Estimated time: ~{est_min}-{est_min*2} minutes "
          f"(~{est_min//60}-{est_min*2//60} hours)")
    print(f"{'='*70}\n")

    ablate_root = ABLATION_RESULTS_DIR; ablate_root.mkdir(parents=True, exist_ok=True)
    master_rows = []
    band_summaries = {}    # band -> ordered list of summary entries (models + ensembles)

    for band_arg in bands_to_run:
        bands = ["10ghz", "24ghz", "77ghz"] if band_arg == "combined" else [band_arg]
        entries = index_files(DATA_ROOT, bands)
        if not entries:
            print(f"[!] No files for band={band_arg}; skipping."); continue
        from sklearn.model_selection import train_test_split
        labels = np.array([e[1] for e in entries]); idx = np.arange(len(entries))
        idx_trv, idx_te, y_trv, _ = train_test_split(
            idx, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_SEED)
        val_rel = VAL_SIZE / (1.0 - TEST_SIZE)
        idx_tr, idx_va = train_test_split(
            idx_trv, test_size=val_rel, stratify=y_trv, random_state=RANDOM_SEED)
        train_ds = RadarSpectrogramDataset([entries[i] for i in idx_tr], augment=True,  cache=True)
        val_ds   = RadarSpectrogramDataset([entries[i] for i in idx_va], augment=False, cache=True)
        test_ds  = RadarSpectrogramDataset([entries[i] for i in idx_te], augment=False, cache=True)

        # All trained configs for this band (kept in memory only — for ensembling).
        band_all_configs = []

        for mname in model_names:
            cls = registry[mname]
            out_dir = ablate_root / band_arg / f"ablation_{mname}"
            out_dir.mkdir(parents=True, exist_ok=True)
            band_model_rows = []

            for ci, cfg in enumerate(configs):
                cfg_id = f"L{cfg['loss']}_PH{int(cfg['physics'])}_PT{cfg['pretrain']}"
                t0 = time.time()
                print(f"\n[ablate] band={band_arg}  model={mname}  "
                      f"cfg={ci+1}/{len(configs)} ({cfg_id})")
                try:
                    result = train_one_model(
                        cls, mname, train_ds, val_ds, test_ds,
                        lr=MODEL_BASE_LR.get(mname, LEARNING_RATE),
                        weight_decay=WEIGHT_DECAY,
                        dropout=0.3, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        verbose=False,
                        loss_type=cfg["loss"],
                        use_physics=cfg["physics"],
                        use_temperature=True,
                        mc_passes=MC_DROPOUT_PASSES,
                        pretrain_mode=cfg["pretrain"],
                    )
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    continue
                elapsed = round(time.time() - t0, 1)
                acc = result["test_acc"]
                f1 = float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                    result["y_true"], result["y_pred"], average="macro", zero_division=0))
                temp = result.get("temperature", 1.0)
                print(f"  -> acc={acc:.4f}  f1={f1:.4f}  T={temp:.2f}  "
                      f"ep={result['hyperparams']['epochs_run']}  time={elapsed}s")

                # Persist the per-config result JSON
                with open(out_dir / f"{cfg_id}_result.json", "w") as f:
                    json.dump({**result, "config": cfg, "model": mname,
                               "band": band_arg, "elapsed_sec": elapsed}, f, indent=2)

                row = {"model": mname, "band": band_arg, **cfg,
                       "test_acc": acc, "macro_f1": f1,
                       "val_acc": result.get("val_acc", 0),
                       "temperature": temp,
                       "epochs": result["hyperparams"]["epochs_run"],
                       "elapsed_sec": elapsed,
                       # Kept in memory for the band-level ensemble step
                       "_probs": np.array(result["test_probs"]),
                       "_y_true": np.array(result["y_true"]),
                       "config_id": cfg_id}
                band_model_rows.append(row)
                band_all_configs.append(row)
                master_rows.append({k: v for k, v in row.items()
                                    if not k.startswith("_")})

                # Free GPU after each
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Per-(band, model) comparison table
            band_model_rows = sorted(band_model_rows, key=lambda r: -r["test_acc"])
            md = [f"# Ablation — band: {band_arg}   model: {mname}", "",
                  "| Rank | Loss | Physics | Pretrain    | Test Acc | Macro F1 | val_acc | T    | epochs |",
                  "|------|------|---------|-------------|----------|----------|---------|------|--------|"]
            for i, r in enumerate(band_model_rows, 1):
                md.append(f"| {i} | {r['loss']} | "
                          f"{'on' if r['physics'] else 'off'} | "
                          f"{r['pretrain']:11s} | "
                          f"{r['test_acc']:.4f} | {r['macro_f1']:.4f} | "
                          f"{r['val_acc']:.4f} | {r['temperature']:.2f} | "
                          f"{r['epochs']} |")
            (out_dir / "comparison.md").write_text("\n".join(md))
            # Strip the in-memory tensors before writing JSON
            json_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in band_model_rows]
            with open(out_dir / "comparison.json", "w") as f:
                json.dump(json_rows, f, indent=2)
            print(f"\n[ablate] wrote {out_dir / 'comparison.md'}")

        # ============================================================
        # BAND-LEVEL SUMMARY: best config per model + top-3/top-5 ensembles
        # ============================================================
        from sklearn.metrics import accuracy_score, f1_score
        # Best config per model, ranked by validation accuracy (not test, to avoid
        # selection bias on test set).
        best_per_model = {}
        for r in band_all_configs:
            m = r["model"]
            if m not in best_per_model or r["val_acc"] > best_per_model[m]["val_acc"]:
                best_per_model[m] = r
        ranked = sorted(best_per_model.values(), key=lambda r: -r["val_acc"])

        # Build top-3 and top-5 ensembles by averaging softmax (weighted by val_acc).
        band_summary_entries = []
        for r in ranked:
            band_summary_entries.append({
                "kind": "model",
                "model": r["model"],
                "best_config_id": r["config_id"],
                "loss": r["loss"],
                "physics": r["physics"],
                "pretrain": r["pretrain"],
                "test_acc": r["test_acc"],
                "macro_f1": r["macro_f1"],
                "val_acc": r["val_acc"],
                "temperature": r["temperature"],
                "epochs": r["epochs"],
            })

        for K in (3, 5):
            top = ranked[:min(K, len(ranked))]
            if len(top) < 2:
                continue
            val_accs = np.array([t["val_acc"] for t in top])
            tau = 0.05
            w = np.exp(val_accs / tau); w = w / w.sum()
            y_true = top[0]["_y_true"]
            avg_probs = sum(wi * t["_probs"] for wi, t in zip(w, top))
            y_pred_ens = avg_probs.argmax(axis=1)
            acc_ens = float(accuracy_score(y_true, y_pred_ens))
            f1_ens  = float(f1_score(y_true, y_pred_ens, average="macro", zero_division=0))
            ens_name = f"Ensemble_top{K}"
            print(f"\n[{band_arg}] {ens_name}: acc={acc_ens:.4f}  f1={f1_ens:.4f}")
            print(f"           members: {[t['model']+'/'+t['config_id'] for t in top]}")
            band_summary_entries.insert(0, {
                "kind": "ensemble",
                "model": ens_name,
                "members": [f"{t['model']}/{t['config_id']}" for t in top],
                "weights": [float(x) for x in w],
                "test_acc": acc_ens,
                "macro_f1": f1_ens,
                "loss": None, "physics": None, "pretrain": None,
                "val_acc": None, "temperature": None, "epochs": None,
            })

        # Re-sort the band summary table by test_acc so the best rows are on top
        band_summary_entries = sorted(band_summary_entries,
                                      key=lambda e: -e["test_acc"])

        # ---- write per-band comparison.md and .json (the deliverable) ----
        band_dir = ablate_root / band_arg
        band_md = [f"# Ablation summary — band: {band_arg}",
                   "",
                   "Best config per model (selected by val_acc) plus weighted top-3 and top-5 ensembles.",
                   "",
                   "| Rank | Entry                    | Loss | Phys | Pretrain    | Test Acc | Macro F1 | val_acc | T    |",
                   "|------|--------------------------|------|------|-------------|----------|----------|---------|------|"]
        for i, e in enumerate(band_summary_entries, 1):
            if e["kind"] == "ensemble":
                row = (f"| {i} | **{e['model']}** | - | - | - | "
                       f"**{e['test_acc']:.4f}** | **{e['macro_f1']:.4f}** | - | - |")
            else:
                row = (f"| {i} | {e['model']} | {e['loss']} | "
                       f"{'on' if e['physics'] else 'off'} | "
                       f"{e['pretrain']:11s} | "
                       f"{e['test_acc']:.4f} | {e['macro_f1']:.4f} | "
                       f"{e['val_acc']:.4f} | {e['temperature']:.2f} |")
            band_md.append(row)
        # Append the ensemble member lists for traceability
        band_md.append("")
        band_md.append("### Ensemble members (model / best_config_id)")
        for e in band_summary_entries:
            if e["kind"] == "ensemble":
                band_md.append(f"- **{e['model']}** weights={['%.2f'%w for w in e['weights']]}")
                for mem in e["members"]:
                    band_md.append(f"    - {mem}")
        (band_dir / "comparison.md").write_text("\n".join(band_md))
        with open(band_dir / "comparison.json", "w") as f:
            json.dump({"band": band_arg, "entries": band_summary_entries}, f, indent=2)
        print(f"\n[ablate] wrote {band_dir / 'comparison.md'}")
        band_summaries[band_arg] = band_summary_entries

    # ====================================================================
    # MASTER COMPARISON: per-band summary tables side by side
    # ====================================================================
    md = ["# Master ablation comparison", "",
          f"Models ablated: {model_names}",
          f"Configs per model: {len(configs)} ({'loss × physics × pretrain' if not args.quick else 'loss × physics'})",
          f"Bands: {bands_to_run}",
          ""]
    for band, entries in band_summaries.items():
        md.append(f"## Band: {band}")
        md.append("")
        md.append("| Rank | Entry                    | Loss | Phys | Pretrain    | Test Acc | Macro F1 | val_acc | T    |")
        md.append("|------|--------------------------|------|------|-------------|----------|----------|---------|------|")
        for i, e in enumerate(entries, 1):
            if e["kind"] == "ensemble":
                md.append(f"| {i} | **{e['model']}** | - | - | - | "
                          f"**{e['test_acc']:.4f}** | **{e['macro_f1']:.4f}** | - | - |")
            else:
                md.append(f"| {i} | {e['model']} | {e['loss']} | "
                          f"{'on' if e['physics'] else 'off'} | "
                          f"{e['pretrain']:11s} | "
                          f"{e['test_acc']:.4f} | {e['macro_f1']:.4f} | "
                          f"{e['val_acc']:.4f} | {e['temperature']:.2f} |")
        md.append("")
    md.append("---")
    md.append("")
    md.append("## Full master table (every config across every band, top 50 by test_acc)")
    md.append("")
    md.append("| Rank | Band | Model | Loss | Physics | Pretrain    | Test Acc | Macro F1 |")
    md.append("|------|------|-------|------|---------|-------------|----------|----------|")
    master_rows_sorted = sorted(master_rows, key=lambda r: -r["test_acc"])
    for i, r in enumerate(master_rows_sorted[:50], 1):
        md.append(f"| {i} | {r['band']} | {r['model']} | {r['loss']} | "
                  f"{'on' if r['physics'] else 'off'} | "
                  f"{r['pretrain']:11s} | "
                  f"{r['test_acc']:.4f} | {r['macro_f1']:.4f} |")
    (ablate_root / "master_comparison.md").write_text("\n".join(md))

    # Master JSON: nested per-band summaries + flat config list
    with open(ablate_root / "master_comparison.json", "w") as f:
        json.dump({
            "models_ablated": model_names,
            "bands": list(bands_to_run),
            "configs_per_model": len(configs),
            "band_summaries": band_summaries,
            "all_configs": master_rows_sorted,
        }, f, indent=2)
    print(f"\nAblation complete. Top-level summary: {ablate_root.resolve()}/master_comparison.md")




# ====================================================================================
# ====================================================================================
#  NARVAL CONSOLIDATED SECTION
#  Combines: comparison_ablation.py (3 high-res models) + narval_full_ablation.py
#  Everything below uses the module itself as "R" so the existing R.* references
#  from the original standalone files work unchanged.
# ====================================================================================
# ====================================================================================
R = sys.modules[__name__]          # self-reference: R.NUM_CLASSES, R.train_one_model, ...

# Honor SLURM's allocated CPUs for the DataLoader workers if present.
import os as _os
try:
    _ncpu = int(_os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if _ncpu > 0:
        globals()["NUM_WORKERS"] = max(0, _ncpu - 1)
except Exception:
    pass

# ---- Linux / Narval paths (override with --data-root / --out-dir) ------------------
# These are relative to your home dir via the ~/projects symlink on Narval.
# If you prefer absolute, use "/project/def-shervinv/atikmahabub/...".
NARVAL_DATA_ROOT = Path(r"Data")
NARVAL_OUT_DIR   = Path(r"RADAR_Motion_Results_224_without_regions")

# Force a headless matplotlib backend (Narval compute nodes have no display).
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


# ====================================================================================
#  THREE HIGH-RESOLUTION COMPARISON MODELS
#  (from comparison_ablation.py: resnet50_up, inceptionv3_up, radmamba_plus)
# ====================================================================================
def _make_resnet50_up(dropout=0.3):
    import torch.nn as nn, torchvision.models as tvm
    net = _safe_weights(tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, NUM_CLASSES))
    return net


def _make_inceptionv3_up(dropout=0.3):
    import torch, torch.nn as nn, torch.nn.functional as F, torchvision.models as tvm
    net = _safe_weights(tvm.inception_v3, tvm.Inception_V3_Weights.IMAGENET1K_V1,
                        aux_logits=True, init_weights=False)
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, NUM_CLASSES))
    net.aux_logits = False; net.AuxLogits = None

    class Wrap(nn.Module):
        def __init__(self): super().__init__(); self.net = net
        def forward(self, x):
            if x.shape[-1] < 75 or x.shape[-2] < 75:
                x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            out = self.net(x)
            if isinstance(out, tuple): return out[0]
            return out.logits if hasattr(out, "logits") else out
    return Wrap()


def _build_enhanced_radmamba():
    import torch, torch.nn as nn, torch.nn.functional as F

    class SelectiveSSM(nn.Module):
        """Minimal pure-PyTorch Mamba selective state-space block."""
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_inner = expand * d_model
            self.d_state = d_state
            self.in_proj  = nn.Linear(d_model, self.d_inner * 2)
            self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                      groups=self.d_inner, padding=d_conv - 1)
            self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1)
            self.dt_proj  = nn.Linear(1, self.d_inner)
            A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
            self.A_log    = nn.Parameter(torch.log(A))
            self.D        = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model)

        def forward(self, x):
            B, L, _ = x.shape
            x_, z = self.in_proj(x).chunk(2, dim=-1)
            x_ = self.conv1d(x_.transpose(1, 2))[..., :L].transpose(1, 2)
            x_ = F.silu(x_)
            proj = self.x_proj(x_)
            Bm = proj[..., :self.d_state]
            Cm = proj[..., self.d_state:2 * self.d_state]
            dt = F.softplus(self.dt_proj(proj[..., -1:]))
            A = -torch.exp(self.A_log)
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
            ys = []
            for t in range(L):
                dt_t = dt[:, t]
                dA = torch.exp(dt_t.unsqueeze(-1) * A)
                dB = dt_t.unsqueeze(-1) * Bm[:, t].unsqueeze(1)
                h = dA * h + dB * x_[:, t].unsqueeze(-1)
                ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, dim=1) + x_ * self.D
            return self.out_proj(y * F.silu(z))

    class BiCPMambaBlock(nn.Module):
        def __init__(self, d_model, d_state=16):
            super().__init__()
            self.norm = nn.LayerNorm(d_model)
            self.conv_proj = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
            self.ssm_fwd = SelectiveSSM(d_model, d_state=d_state)
            self.ssm_bwd = SelectiveSSM(d_model, d_state=d_state)
            self.merge = nn.Linear(2 * d_model, d_model)
        def forward(self, x):
            r = x
            x = self.norm(x)
            x = self.conv_proj(x.transpose(1, 2)).transpose(1, 2)
            f = self.ssm_fwd(x)
            b = torch.flip(self.ssm_bwd(torch.flip(x, dims=[1])), dims=[1])
            return self.merge(torch.cat([f, b], dim=-1)) + r

    class RadMambaPlus(nn.Module):
        def __init__(self, dropout=0.3, d_model=96, depth=3,
                     n_doppler_keep=96, time_tokens=96):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.SiLU(),
                nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.SiLU())
            self.down  = nn.AdaptiveAvgPool2d((n_doppler_keep, time_tokens))
            self.embed = nn.Linear(48 * n_doppler_keep, d_model)
            self.blocks = nn.ModuleList([BiCPMambaBlock(d_model) for _ in range(depth)])
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, NUM_CLASSES)

        def forward(self, x):
            x = self.stem(x)
            x = self.down(x)
            B, C, Dk, T = x.shape
            x = x.permute(0, 3, 1, 2).reshape(B, T, C * Dk)
            x = self.embed(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x).mean(dim=1)
            return self.head(self.dropout(x))

    return RadMambaPlus


class ResNet50_Up:
    def __new__(cls, dropout=0.3): return _make_resnet50_up(dropout)
class InceptionV3_Up:
    def __new__(cls, dropout=0.3): return _make_inceptionv3_up(dropout)
class RadMamba_Plus:
    def __new__(cls, dropout=0.3): return _build_enhanced_radmamba()(dropout=dropout)


# ====================================================================================
#  MODEL SET (6 models, per user request)
#    radar  : 01_CNN_LSTM, 02_CNN_BiLSTM_Attn, 03_ResNet18_Head, 07_ConvNeXt_Transformer
#    compare: inceptionv3_up, radmamba_plus
#  (Inception-v4 is not in torchvision's offline wheelhouse on Narval, so
#   "inceptionv3 and v4" is served by inceptionv3_up. Add resnet50_up back to
#   COMPARISON_MODELS if you want it.)
# ====================================================================================
RADAR_MODELS = ["01_CNN_LSTM", "02_CNN_BiLSTM_Attn",
                "03_ResNet18_Head", "07_ConvNeXt_Transformer"]
COMPARISON_MODELS = {"inceptionv3_up": InceptionV3_Up,
                     "radmamba_plus": RadMamba_Plus}
ALL_MODELS = RADAR_MODELS + list(COMPARISON_MODELS.keys())   # 6 models
NARVAL_BANDS = ["10ghz", "24ghz", "77ghz", "combined"]
COMPARISON_LR = {"resnet50_up": 3e-4, "inceptionv3_up": 3e-4, "radmamba_plus": 8e-4}


# ====================================================================================
#  Learned autoencoder stem (DOWNSCALE_MODE == "autoencoder")
#  Takes the (3, AE_INPUT_SIZE) multichannel image (avg / mixed / max pooling
#  channels) and produces a refined (3, SPECTROGRAM_SIZE) representation with a
#  small conv encoder trained jointly with the classifier (task-aware).
#
#  With AE_INPUT_SIZE == SPECTROGRAM_SIZE (== 384) the convs run at 384 and the
#  AdaptiveAvgPool is an identity in size — so the stem LEARNS HOW TO FUSE the
#  three pooling views into the best 3-channel input for the backbone. If you set
#  AE_INPUT_SIZE larger than the final size, the same AdaptiveAvgPool cleanly
#  downsamples to SPECTROGRAM_SIZE (no wasteful stride-then-upsample).
# ====================================================================================
def _make_ae_stem(base_module, out_size):
    import torch.nn as nn
    class _AEStem(nn.Module):
        def __init__(self, base, out_hw):
            super().__init__()
            # convolutions stay at the input resolution; the adaptive pool sets
            # the exact output size (identity when in==out, downsample when in>out)
            self.enc = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
                nn.Conv2d(16, 3, 3, padding=1),
                nn.AdaptiveAvgPool2d(tuple(out_hw)),   # exact target size
                nn.Sigmoid())                          # keep output in [0,1]
            self.base = base
        def forward(self, x):
            return self.base(self.enc(x))
    return _AEStem(base_module, out_size)


def _get_model_cls(name, registry):
    base = COMPARISON_MODELS[name] if name in COMPARISON_MODELS else registry[name]
    if DOWNSCALE_MODE == "autoencoder":
        out_hw = SPECTROGRAM_SIZE
        def factory(dropout=0.3, _base=base, _out=out_hw):
            return _make_ae_stem(_base(dropout=dropout), _out)
        return factory
    return base


def _get_lr(name):
    if name in COMPARISON_LR:
        return COMPARISON_LR[name]
    return MODEL_BASE_LR.get(name, LEARNING_RATE)


# ====================================================================================
#  DATA SPLIT + DATASETS
# ====================================================================================
def _nv_split(entries):
    from sklearn.model_selection import train_test_split
    labels = np.array([e[1] for e in entries]); idx = np.arange(len(entries))
    itrv, ite, ytrv, _ = train_test_split(idx, labels, test_size=TEST_SIZE,
                                           stratify=labels, random_state=RANDOM_SEED)
    itr, iva = train_test_split(itrv, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                stratify=ytrv, random_state=RANDOM_SEED)
    return ([entries[i] for i in itr], [entries[i] for i in iva], [entries[i] for i in ite])


def _nv_datasets(band_arg):
    bands = ["10ghz", "24ghz", "77ghz"] if band_arg == "combined" else [band_arg]
    entries = index_files(DATA_ROOT, bands)
    if not entries:
        return None
    tr, va, te = _nv_split(entries)
    return (RadarSpectrogramDataset(tr, augment=True,  cache=True),
            RadarSpectrogramDataset(va, augment=False, cache=True),
            RadarSpectrogramDataset(te, augment=False, cache=True))


def _nv_configs(quick):
    # Ablation grid (per user request):
    #   Loss          : {ce, edl}
    #   Physics-prior : {off, on}        (Doppler-symmetry consistency)
    #   Pretrain      : {none, denoising}  ("contrastive" removed)
    # => 2 x 2 x 2 = 8 configs.  With COMPARE_DOWNSCALING, each config is also
    # run under every downscaling method (x3) so the results compare them.
    if quick:
        base = [{"loss": l, "physics": p, "pretrain": "none"}
                for l in ("ce", "edl") for p in (False, True)]
    else:
        base = [{"loss": l, "physics": p, "pretrain": pt}
                for l in ("ce", "edl") for p in (False, True)
                for pt in ("none", "denoising")]
    modes = DOWNSCALE_MODES_TO_COMPARE if COMPARE_DOWNSCALING else [DOWNSCALE_MODE]
    return [{**c, "downscale": ds} for ds in modes for c in base]


# ====================================================================================
#  TEXT CONFUSION MATRIX (no images — Narval-friendly)
# ====================================================================================
def confusion_matrix_text(y_true, y_pred, class_names):
    """Return (markdown_string, json_dict) for a confusion matrix. No image."""
    from sklearn.metrics import confusion_matrix
    K = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    # short labels for header
    short = [c[:6] for c in class_names]
    header = "| true\\pred | " + " | ".join(short) + " | total |"
    sep    = "|" + "---|" * (K + 2)
    rows = [header, sep]
    for i in range(K):
        cells = " | ".join(str(int(cm[i, j])) for j in range(K))
        rows.append(f"| {short[i]} | {cells} | {int(cm[i].sum())} |")
    # per-class recall (diagonal of normalized)
    rows.append("")
    rows.append("Per-class recall (diagonal): " +
                ", ".join(f"{short[i]}={cm_norm[i, i]:.3f}" for i in range(K)))
    md = "\n".join(rows)
    js = {"labels": list(class_names),
          "matrix_counts": cm.tolist(),
          "matrix_normalized": np.round(cm_norm, 4).tolist()}
    return md, js


# ====================================================================================
#  TRAIN ONE (model, band): all configs, save .pth checkpoints + result JSONs
# ====================================================================================
def nv_run_model_band(model_name, band_arg, out_root, quick, epochs):
    import torch
    from sklearn.metrics import f1_score
    registry = _build_models_module()
    configs = _nv_configs(quick)

    # Build one dataset per unique data representation. "autoencoder" uses the
    # same avg/mix/max channels as "multichannel" (it differs only by the learned
    # stem on the model side), so they share a dataset.
    def _data_sig(dsmode):
        return "multichannel" if dsmode in ("multichannel", "autoencoder") else dsmode
    sigs = sorted(set(_data_sig(c["downscale"]) for c in configs))
    ds_bundles = {}
    for sig in sigs:
        globals()["DOWNSCALE_MODE"] = sig
        b = _nv_datasets(band_arg)
        if b is None:
            print(f"[!] no data for {band_arg}; skipping {model_name}.", flush=True); return []
        ds_bundles[sig] = b

    out_dir = out_root / band_arg / f"ablation_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints" / band_arg / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ci, cfg in enumerate(configs):
        dsmode = cfg["downscale"]
        # Set DOWNSCALE_MODE so _get_model_cls wraps with the AE stem iff needed.
        globals()["DOWNSCALE_MODE"] = dsmode
        cls = _get_model_cls(model_name, registry)
        train_ds, val_ds, test_ds = ds_bundles[_data_sig(dsmode)]
        cfg_id = (f"DS{_DS_TAG.get(dsmode, dsmode)}_L{cfg['loss']}_"
                  f"PH{int(cfg['physics'])}_PT{cfg['pretrain']}")
        print(f"[{band_arg}] {model_name}  cfg {ci+1}/{len(configs)} ({cfg_id})", flush=True)
        t0 = time.time()
        try:
            result = train_one_model(
                cls, model_name, train_ds, val_ds, test_ds,
                lr=_get_lr(model_name), weight_decay=WEIGHT_DECAY, dropout=0.3,
                batch_size=BATCH_SIZE, epochs=epochs, verbose=False,
                loss_type=cfg["loss"], use_physics=cfg["physics"],
                use_temperature=True, mc_passes=MC_DROPOUT_PASSES,
                pretrain_mode=cfg["pretrain"])
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}", flush=True); continue
        dt = round(time.time() - t0, 1)
        acc = result["test_acc"]
        f1 = float(f1_score(result["y_true"], result["y_pred"], average="macro", zero_division=0))

        # --- save the trained weights so we never retrain ---
        state = result.pop("_state_dict", None)
        if state is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)   # ensure exists before save
            torch.save({"model": model_name, "band": band_arg, "config": cfg,
                        "state_dict": state, "test_acc": acc, "macro_f1": f1,
                        "input_size": list(SPECTROGRAM_SIZE)},
                       ckpt_dir / f"{cfg_id}.pth")
        print(f"   -> acc={acc:.4f} f1={f1:.4f} ep={result['hyperparams']['epochs_run']} "
              f"t={dt}s  (ckpt saved)", flush=True)

        out_dir.mkdir(parents=True, exist_ok=True)   # ensure exists before write
        with open(out_dir / f"{cfg_id}_result.json", "w") as f:
            json.dump({"model": model_name, "band": band_arg, "config": cfg,
                       "config_id": cfg_id, "test_acc": acc, "macro_f1": f1,
                       "val_acc": result.get("val_acc", 0),
                       "temperature": result.get("temperature", 1.0),
                       "epochs": result["hyperparams"]["epochs_run"],
                       "y_true": list(map(int, result["y_true"])),
                       "y_pred": list(map(int, result["y_pred"])),
                       "test_probs": np.asarray(result["test_probs"]).tolist()}, f)
        rows.append({"model": model_name, "band": band_arg, **cfg, "config_id": cfg_id,
                     "test_acc": acc, "macro_f1": f1, "val_acc": result.get("val_acc", 0),
                     "temperature": result.get("temperature", 1.0),
                     "epochs": result["hyperparams"]["epochs_run"]})
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda r: -r["test_acc"])
    md = [f"# Ablation -- band {band_arg}  model {model_name}", "",
          "| Rank | Downscale | Loss | Physics | Pretrain | Test Acc | Macro F1 | val_acc | T |",
          "|------|-----------|------|---------|----------|----------|----------|---------|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r.get('downscale','-')} | {r['loss']} | "
                  f"{'on' if r['physics'] else 'off'} | {r['pretrain']} | "
                  f"{r['test_acc']:.4f} | {r['macro_f1']:.4f} | "
                  f"{r['val_acc']:.4f} | {r['temperature']:.2f} |")
    (out_dir / "comparison.md").write_text("\n".join(md))
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


# ====================================================================================
#  EFFICIENCY PROFILE: params, FLOPs, latency, memory (best/avg/worst)
# ====================================================================================
def nv_profile_all(size, out_root, models=None):
    import torch, torch.nn as nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = _build_models_module()
    model_names = models or ALL_MODELS
    results = []
    for name in model_names:
        cls = _get_model_cls(name, registry)
        try:
            model = cls(dropout=0.3).to(device).eval()
        except Exception as e:
            print(f"[profile] {name} build failed: {e}", flush=True); continue
        n_params = sum(p.numel() for p in model.parameters())
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

        macs = [0]; handles = []
        def conv2d_hook(m, inp, out):
            oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
            kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2)
            macs[0] += oc * oh * ow * (inp[0].shape[1] // m.groups) * kh * kw
        def conv1d_hook(m, inp, out):
            oc, ol = out.shape[1], out.shape[2]
            k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            macs[0] += oc * ol * (inp[0].shape[1] // m.groups) * k
        def lin_hook(m, inp, out):
            macs[0] += m.in_features * m.out_features
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d):   handles.append(mod.register_forward_hook(conv2d_hook))
            elif isinstance(mod, nn.Conv1d): handles.append(mod.register_forward_hook(conv1d_hook))
            elif isinstance(mod, nn.Linear): handles.append(mod.register_forward_hook(lin_hook))
        x = torch.randn(1, 3, *size, device=device)
        with torch.no_grad(): _ = model(x)
        for h in handles: h.remove()
        gflops = 2 * macs[0] / 1e9

        with torch.no_grad():
            for _ in range(10): model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(30): model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            lat_ms = (time.time() - t0) / 30 * 1000

        peak_mb = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(): model(x)
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1e6

        results.append({"model": name, "params": int(n_params),
                        "param_mb": round(param_mb, 2), "gflops": round(gflops, 3),
                        "latency_ms": round(lat_ms, 3),
                        "peak_activation_mb": round(peak_mb, 2) if peak_mb else None})
        print(f"[profile] {name:24s} params={n_params/1e6:6.2f}M GFLOPs={gflops:8.2f} "
              f"lat={lat_ms:7.2f}ms mem={param_mb:7.1f}MB", flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    def baw(key):
        vals = [(r["model"], r[key]) for r in results if r[key] is not None]
        if not vals: return None
        best = min(vals, key=lambda t: t[1]); worst = max(vals, key=lambda t: t[1])
        return {"best": {"model": best[0], "value": best[1]},
                "worst": {"model": worst[0], "value": worst[1]},
                "average": round(float(np.mean([v for _, v in vals])), 3)}
    summary = {k: baw(k) for k in ("params", "param_mb", "gflops", "latency_ms",
                                   "peak_activation_mb")}
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "efficiency_profile.json", "w") as f:
        json.dump({"input_size": list(size), "device": str(device),
                   "per_model": results, "best_average_worst": summary}, f, indent=2)
    md = [f"# Efficiency profile  (input {size[0]}x{size[1]}, device={device})", "",
          "| Model | Params (M) | Param mem (MB) | GFLOPs/sample | Latency (ms) | Peak act. (MB) |",
          "|-------|-----------|----------------|---------------|--------------|----------------|"]
    for r in sorted(results, key=lambda r: r["gflops"]):
        md.append(f"| {r['model']} | {r['params']/1e6:.2f} | {r['param_mb']:.1f} | "
                  f"{r['gflops']:.2f} | {r['latency_ms']:.2f} | "
                  f"{r['peak_activation_mb'] if r['peak_activation_mb'] else 'n/a'} |")
    md += ["", "## Best / Average / Worst (best = lowest = most efficient)", "",
           "| Metric | Best (model) | Average | Worst (model) |",
           "|--------|--------------|---------|---------------|"]
    label = {"params": "Parameters", "param_mb": "Param memory (MB)",
             "gflops": "GFLOPs/sample", "latency_ms": "Latency (ms)",
             "peak_activation_mb": "Peak activation (MB)"}
    for k, lab in label.items():
        s = summary[k]
        if not s: continue
        md.append(f"| {lab} | {s['best']['value']} ({s['best']['model']}) | "
                  f"{s['average']} | {s['worst']['value']} ({s['worst']['model']}) |")
    md += ["", "Notes: FLOPs counts Conv1d/Conv2d/Linear MACs x2 (dominant cost); "
           "norm/activation/elementwise excluded. Latency = per-sample (batch=1) wall "
           "time, 30 runs after 10 warmups.", ""]
    (out_root / "efficiency_profile.md").write_text("\n".join(md))
    print(f"[profile] wrote {out_root / 'efficiency_profile.md'}", flush=True)
    return results, summary


# ====================================================================================
#  AGGREGATE: master comparison + best-config metrics + TEXT confusion matrices
# ====================================================================================
def nv_aggregate(out_root, size):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 f1_score)
    cm_dir = out_root / "confusion_matrices"; cm_dir.mkdir(parents=True, exist_ok=True)
    band_summaries = {}; metrics_rows = []; all_rows = []
    cm_json_all = {}
    ds_compare_rows = []     # for the downscaling-method comparison

    for band in NARVAL_BANDS:
        band_dir = out_root / band
        if not band_dir.exists(): continue
        per_model_best = {}
        for model_dir in sorted(band_dir.glob("ablation_*")):
            model = model_dir.name.replace("ablation_", "")
            best = None
            for jf in model_dir.glob("*_result.json"):
                d = json.loads(jf.read_text())
                all_rows.append({k: d[k] for k in ("model", "band", "config_id",
                                                   "test_acc", "macro_f1", "val_acc")})
                ds_compare_rows.append({
                    "band": band, "model": model,
                    "downscale": d["config"].get("downscale", "?"),
                    "test_acc": d["test_acc"], "macro_f1": d["macro_f1"],
                    "val_acc": d.get("val_acc", 0)})
                if best is None or d["val_acc"] > best["val_acc"]:
                    best = d
            if best is None: continue
            per_model_best[model] = best

            y_true = np.array(best["y_true"]); y_pred = np.array(best["y_pred"])
            acc = float(accuracy_score(y_true, y_pred))
            pr, rc, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0)
            metrics_rows.append({"band": band, "model": model,
                                 "best_config": best["config_id"],
                                 "accuracy": round(acc, 4),
                                 "precision_macro": round(float(pr), 4),
                                 "recall_macro": round(float(rc), 4),
                                 "f1_macro": round(float(f1), 4)})
            # TEXT confusion matrix (md + json) -- no image
            cm_md, cm_js = confusion_matrix_text(y_true, y_pred, CLASS_NAMES)
            (cm_dir / f"{band}_{model}_cm.md").write_text(
                f"# Confusion matrix -- {model} on {band} "
                f"(best config {best['config_id']})\n\n{cm_md}\n")
            cm_json_all[f"{band}_{model}"] = {"best_config": best["config_id"], **cm_js}

        ranked = sorted(per_model_best.values(), key=lambda d: -d["val_acc"])
        entries = [{"kind": "model", "model": d["model"], "best_config_id": d["config_id"],
                    "loss": d["config"]["loss"], "physics": d["config"]["physics"],
                    "pretrain": d["config"]["pretrain"], "test_acc": d["test_acc"],
                    "macro_f1": d["macro_f1"], "val_acc": d["val_acc"]} for d in ranked]
        for K in (3, 5):
            top = ranked[:min(K, len(ranked))]
            if len(top) < 2: continue
            w = np.exp(np.array([t["val_acc"] for t in top]) / 0.05); w = w / w.sum()
            yt = np.array(top[0]["y_true"])
            avg = sum(wi * np.array(t["test_probs"]) for wi, t in zip(w, top))
            yp = avg.argmax(1)
            # text CM for the ensemble too
            cm_md, cm_js = confusion_matrix_text(yt, yp, CLASS_NAMES)
            (cm_dir / f"{band}_Ensemble_top{K}_cm.md").write_text(
                f"# Confusion matrix -- Ensemble_top{K} on {band}\n\n{cm_md}\n")
            cm_json_all[f"{band}_Ensemble_top{K}"] = cm_js
            entries.insert(0, {"kind": "ensemble", "model": f"Ensemble_top{K}",
                               "members": [f"{t['model']}/{t['config_id']}" for t in top],
                               "test_acc": float(accuracy_score(yt, yp)),
                               "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                               "loss": None, "physics": None, "pretrain": None, "val_acc": None})
        entries = sorted(entries, key=lambda e: -e["test_acc"])
        band_summaries[band] = entries

        bmd = [f"# Ablation summary -- band {band}  (input {size[0]}x{size[1]})", "",
               "| Rank | Entry | Loss | Phys | Pretrain | Test Acc | Macro F1 |",
               "|------|-------|------|------|----------|----------|----------|"]
        for i, e in enumerate(entries, 1):
            if e["kind"] == "ensemble":
                bmd.append(f"| {i} | **{e['model']}** | - | - | - | "
                           f"**{e['test_acc']:.4f}** | **{e['macro_f1']:.4f}** |")
            else:
                bmd.append(f"| {i} | {e['model']} | {e['loss']} | "
                           f"{'on' if e['physics'] else 'off'} | {e['pretrain']} | "
                           f"{e['test_acc']:.4f} | {e['macro_f1']:.4f} |")
        (band_dir / "comparison.md").write_text("\n".join(bmd))
        with open(band_dir / "comparison.json", "w") as f:
            json.dump({"band": band, "input_size": list(size), "entries": entries}, f, indent=2)

    md = [f"# Master comparison -- input {size[0]}x{size[1]}", "",
          f"Models: {ALL_MODELS}", ""]
    for band, entries in band_summaries.items():
        md += [f"## Band: {band}", "",
               "| Rank | Entry | Loss | Phys | Pretrain | Test Acc | Macro F1 |",
               "|------|-------|------|------|----------|----------|----------|"]
        for i, e in enumerate(entries, 1):
            if e["kind"] == "ensemble":
                md.append(f"| {i} | **{e['model']}** | - | - | - | "
                          f"**{e['test_acc']:.4f}** | **{e['macro_f1']:.4f}** |")
            else:
                md.append(f"| {i} | {e['model']} | {e['loss']} | "
                          f"{'on' if e['physics'] else 'off'} | {e['pretrain']} | "
                          f"{e['test_acc']:.4f} | {e['macro_f1']:.4f} |")
        md.append("")
    (out_root / "master_comparison.md").write_text("\n".join(md))
    with open(out_root / "master_comparison.json", "w") as f:
        json.dump({"input_size": list(size), "models": ALL_MODELS,
                   "band_summaries": band_summaries,
                   "all_configs": sorted(all_rows, key=lambda r: -r["test_acc"])}, f, indent=2)

    # all confusion matrices in one JSON too
    with open(out_root / "confusion_matrices_all.json", "w") as f:
        json.dump(cm_json_all, f, indent=2)

    mmd = [f"# Best-config metrics per model (input {size[0]}x{size[1]})", "",
           "Test-set metrics for each model's BEST config (selected by val accuracy).", "",
           "| Band | Model | Best config | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
           "|------|-------|-------------|----------|-------------------|----------------|------------|"]
    for r in sorted(metrics_rows, key=lambda r: (r["band"], -r["accuracy"])):
        mmd.append(f"| {r['band']} | {r['model']} | {r['best_config']} | "
                   f"{r['accuracy']:.4f} | {r['precision_macro']:.4f} | "
                   f"{r['recall_macro']:.4f} | {r['f1_macro']:.4f} |")
    (out_root / "metrics_best_per_model.md").write_text("\n".join(mmd))
    with open(out_root / "metrics_best_per_model.json", "w") as f:
        json.dump(metrics_rows, f, indent=2)
    print(f"[aggregate] wrote master_comparison, metrics_best_per_model, "
          f"and {len(cm_json_all)} text confusion matrices.", flush=True)

    # ---- downscaling-method comparison ----
    _write_downscaling_comparison(out_root, ds_compare_rows, size)


def _write_downscaling_comparison(out_root, rows, size):
    """Head-to-head comparison of the smart-downscaling strategies
    (bilinear / mixed / multichannel / autoencoder). Reports the best test
    accuracy each method achieves per band and per model, plus band-averaged
    means, so you can see which downscaling helps most."""
    if not rows:
        return
    from collections import defaultdict
    modes = sorted(set(r["downscale"] for r in rows))
    bands = [b for b in NARVAL_BANDS if any(r["band"] == b for r in rows)]
    models = sorted(set(r["model"] for r in rows))
    label = {"bilinear": "bilinear", "mix": "mixed", "mixed": "mixed",
             "multichannel": "multichannel", "multi": "multichannel",
             "autoencoder": "autoencoder", "ae": "autoencoder"}

    def best(pred):
        vals = [r["test_acc"] for r in rows if pred(r)]
        return max(vals) if vals else None
    def mean_best_over_models(dsmode, band):
        # best per model for this (band, dsmode), then average across models
        per = []
        for m in models:
            v = [r["test_acc"] for r in rows if r["downscale"]==dsmode and r["band"]==band and r["model"]==m]
            if v: per.append(max(v))
        return float(np.mean(per)) if per else None

    md = [f"# Downscaling-method comparison  (input {size[0]}x{size[1]})", "",
          "Best test accuracy each smart-downscaling strategy achieves. "
          "Higher = the downscaling preserves more discriminative micro-Doppler.", ""]

    # Table 1: best accuracy per (band x downscale)
    md += ["## Best test accuracy per band (max over all models & configs)", "",
           "| Band | " + " | ".join(label.get(m, m) for m in modes) + " | best method |",
           "|------|" + "------|" * (len(modes) + 1)]
    for band in bands:
        cells = []
        bestmode = None; bestv = -1
        for m in modes:
            v = best(lambda r: r["downscale"]==m and r["band"]==band)
            cells.append(f"{v:.4f}" if v is not None else "-")
            if v is not None and v > bestv: bestv, bestmode = v, label.get(m,m)
        md.append(f"| {band} | " + " | ".join(cells) + f" | **{bestmode}** |")

    # Table 2: band-averaged best-per-model (fairer aggregate)
    md += ["", "## Band-averaged accuracy (mean of best-per-model)", "",
           "| Band | " + " | ".join(label.get(m, m) for m in modes) + " |",
           "|------|" + "------|" * len(modes)]
    for band in bands:
        cells = [f"{mean_best_over_models(m,band):.4f}" if mean_best_over_models(m,band) is not None else "-"
                 for m in modes]
        md.append(f"| {band} | " + " | ".join(cells) + " |")

    # Table 3: per-model best (averaged over bands)
    md += ["", "## Best accuracy per model x downscaling (max over bands & configs)", "",
           "| Model | " + " | ".join(label.get(m, m) for m in modes) + " |",
           "|-------|" + "------|" * len(modes)]
    for model in models:
        cells = []
        for m in modes:
            v = best(lambda r: r["downscale"]==m and r["model"]==model)
            cells.append(f"{v:.4f}" if v is not None else "-")
        md.append(f"| {model} | " + " | ".join(cells) + " |")

    # Overall winner
    overall = {m: best(lambda r: r["downscale"]==m) for m in modes}
    overall = {k: v for k, v in overall.items() if v is not None}
    if overall:
        win = max(overall, key=overall.get)
        md += ["", f"**Overall best downscaling: {label.get(win, win)} "
                   f"({overall[win]:.4f} peak test accuracy).**", ""]

    (out_root / "downscaling_comparison.md").write_text("\n".join(md))
    with open(out_root / "downscaling_comparison.json", "w") as f:
        json.dump({"input_size": list(size), "modes": modes,
                   "rows": rows,
                   "best_per_band_mode": {b: {m: best(lambda r: r["downscale"]==m and r["band"]==b)
                                              for m in modes} for b in bands}}, f, indent=2)
    print(f"[aggregate] wrote downscaling_comparison.md ({len(modes)} methods compared).",
          flush=True)



# ====================================================================================
#  Band-energy diagnostic — verify BAND_DOPPLER_REGIONS against real data
# ====================================================================================
def diagnose_band_energy(n_per_band=5):
    """Load a few files per band and report which Doppler row-ranges carry the
    energy, so you can confirm/correct BAND_DOPPLER_REGIONS. Prints contiguous
    bands whose mean row-energy is within AUTO_REGION_DB of the per-file peak."""
    print(f"\n{'='*70}\nBAND ENERGY DIAGNOSTIC  (data_root={DATA_ROOT})\n{'='*70}")
    print("Reports Doppler row-bands within "
          f"{AUTO_REGION_DB} dB of each file's peak row-energy.\n")
    for band in ("10ghz", "24ghz", "77ghz"):
        entries = index_files(DATA_ROOT, [band])[:n_per_band]
        if not entries:
            print(f"[{band}] no files found."); continue
        print(f"[{band}] manual regions in config: {BAND_DOPPLER_REGIONS.get(band)}")
        for path, cid, _ in entries:
            try:
                arr = (_load_mat_array(path) if path.suffix.lower()==".mat"
                       else _load_dat_array(path))
                mag = np.abs(np.asarray(arr, dtype=np.complex128)).squeeze()
                if mag.ndim != 2:
                    print(f"    {path.name}: not 2-D ({mag.shape}), skipped"); continue
                row_e = 20*np.log10(mag.mean(axis=1)+1e-9)
                keep = row_e >= (row_e.max() - AUTO_REGION_DB)
                # find contiguous True runs
                bands_found = []
                i = 0; H = len(keep)
                while i < H:
                    if keep[i]:
                        j = i
                        while j < H and keep[j]: j += 1
                        if (j - i) >= AUTO_REGION_MIN_FRAC*H:
                            bands_found.append((i, j))
                        i = j
                    else: i += 1
                print(f"    {path.name[:40]:40} rows={mag.shape[0]:5d}  "
                      f"energy-bands={bands_found}")
            except Exception as e:
                print(f"    {path.name}: FAILED {type(e).__name__}: {e}")
        print()
    print("Compare 'energy-bands' above to your BAND_DOPPLER_REGIONS. If they "
          "differ, edit BAND_DOPPLER_REGIONS (or use --region-mode auto).\n")


# ====================================================================================
#  MAIN
# ====================================================================================
def main():
    ap = argparse.ArgumentParser(description="Narval consolidated radar HAR ablation")
    ap.add_argument("--data-root", default=NARVAL_DATA_ROOT)
    ap.add_argument("--out-dir",   default=NARVAL_OUT_DIR)
    ap.add_argument("--size", type=int, default=224,
                    help="Square spectrogram size (this is the 224 variant; the "
                         "main file defaults to 384).")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--bands",  nargs="+", default=None)
    ap.add_argument("--task-id", type=lambda v: None if (v is None or str(v).strip()=="") else int(v),
                    default=None,
                    help="SLURM array index 0..39 -> one (model,band) pair. "
                         "Omit to run all pairs sequentially in one job.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--profile-only", action="store_true")
    # ---- Narval pretrained-weight helpers ----
    ap.add_argument("--download-weights", action="store_true",
                    help="Download all backbone weights to --torch-home and exit. "
                         "Run this ONCE on a Narval LOGIN NODE before submitting jobs.")
    ap.add_argument("--torch-home", default=None,
                    help="Override TORCH_HOME (where pretrained weights are cached). "
                         "e.g. ~/projects/def-shervinv/atikmahabub/torch_weights")
    ap.add_argument("--downscale-mode",
                    choices=["bilinear", "mixed", "multichannel", "autoencoder"],
                    default=None, help="Override DOWNSCALE_MODE (default multichannel).")
    ap.add_argument("--mix-alpha", type=float, default=None,
                    help="alpha for mixed/multichannel pooling (default 0.5).")
    ap.add_argument("--region-mode", choices=["manual", "auto", "center"], default=None,
                    help="Override BAND_REGION_MODE for Doppler-region selection.")
    ap.add_argument("--region-layout", choices=["mask", "crop"], default=None,
                    help="mask = keep full axis & zero non-signal (no discontinuities); "
                         "crop = cut & concatenate bands (legacy). Default mask.")
    ap.add_argument("--diagnose-bands", action="store_true",
                    help="Measure & print where Doppler energy concentrates per band "
                         "(to verify BAND_DOPPLER_REGIONS), then exit.")
    ap.add_argument("--compare-downscaling", action="store_true",
                    help="Make downscaling a comparison axis: train every config under "
                         "mixed/multichannel/autoencoder and emit downscaling_comparison.md. "
                         "Triples the trainings.")
    args = ap.parse_args()

    # Set TORCH_HOME FIRST so every subsequent torchvision call uses the cache.
    import os as _os
    if args.torch_home:
        _os.makedirs(args.torch_home, exist_ok=True)
        _os.environ["TORCH_HOME"] = str(args.torch_home)
        print(f"[cfg] TORCH_HOME={args.torch_home}", flush=True)

    if args.download_weights:
        _download_all_weights(args.torch_home)
        return

    # Global overrides (module-level config the rest of the code reads)
    globals()["SPECTROGRAM_SIZE"] = (args.size, args.size)
    globals()["DATA_ROOT"] = Path(args.data_root)
    if args.downscale_mode: globals()["DOWNSCALE_MODE"] = args.downscale_mode
    if args.mix_alpha is not None: globals()["MIX_ALPHA"] = args.mix_alpha
    if args.region_mode: globals()["BAND_REGION_MODE"] = args.region_mode
    if args.region_layout: globals()["BAND_REGION_LAYOUT"] = args.region_layout
    if args.compare_downscaling: globals()["COMPARE_DOWNSCALING"] = True
    # autoencoder stem operates at the same resolution as the final size (384):
    # the stem learns to fuse the avg/mix/max channels rather than upscale-then-pool
    globals()["AE_INPUT_SIZE"] = (args.size, args.size)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)   # create before anything writes

    try:
        import torch
    except ImportError:
        print("torch not installed."); return
    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

    print(f"[cfg] data_root={DATA_ROOT}  out_dir={out_root}  size={SPECTROGRAM_SIZE}  "
          f"epochs={args.epochs}", flush=True)
    print(f"[cfg] downscale={DOWNSCALE_MODE} (alpha={MIX_ALPHA})  "
          f"region_mode={BAND_REGION_MODE}  models={len(ALL_MODELS)}", flush=True)
    if COMPARE_DOWNSCALING:
        print(f"[cfg] compare-downscaling ON: sweeping {DOWNSCALE_MODES_TO_COMPARE} "
              f"-> {len(_nv_configs(False))} configs/model/band", flush=True)

    if args.diagnose_bands:
        diagnose_band_energy(); return

    if args.profile_only:
        nv_profile_all(SPECTROGRAM_SIZE, out_root); return
    if args.aggregate:
        nv_aggregate(out_root, SPECTROGRAM_SIZE)
        nv_profile_all(SPECTROGRAM_SIZE, out_root)
        return

    if args.task_id is not None:
        mi, bi = divmod(args.task_id, len(NARVAL_BANDS))
        if mi >= len(ALL_MODELS):
            print(f"[!] task-id {args.task_id} out of range "
                  f"(max {len(ALL_MODELS)*len(NARVAL_BANDS)-1})."); return
        pairs = [(ALL_MODELS[mi], NARVAL_BANDS[bi])]
    else:
        models = args.models or ALL_MODELS
        bands = args.bands or NARVAL_BANDS
        pairs = [(m, b) for m in models for b in bands]

    print(f"Running {len(pairs)} (model,band) pair(s).", flush=True)
    for model_name, band_arg in pairs:
        nv_run_model_band(model_name, band_arg, out_root, args.quick, args.epochs)
    print("\nPer-pair training complete. Run with --aggregate to build the master "
          "comparison, metrics, and text confusion matrices.", flush=True)


if __name__ == "__main__":
    main()