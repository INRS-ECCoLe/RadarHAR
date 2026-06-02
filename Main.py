"""
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
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ====================================================================
# 1. CONFIG
# ====================================================================
DATA_ROOT = Path(r"C:\Users\atikm\Downloads\CSI RADAR\Open Radar Datasets (OpenRadarInitiative)\Raw Data_raw_data\Cross-frequency\Data")
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

RESULTS_DIR = Path(r"C:\Users\atikm\Downloads\Radar Motion Detection\Results")

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

def _spectrogram_image(arr: np.ndarray) -> np.ndarray:
    """Turn an already-computed (freq x time) magnitude/complex matrix into a
    normalized [0,1] image — this reproduces 'Panel A' from diagnose_data.py."""
    from PIL import Image
    mag = np.abs(np.asarray(arr, dtype=np.complex128))
    # CI4R sx1 is (freq=4096, time=538): frequency on the vertical axis. Keep as-is.
    # Optional center-crop along the Doppler (vertical) axis to drop noise bands.
    if 0 < DOPPLER_CROP_FRAC < 1.0 and mag.ndim == 2:
        h = mag.shape[0]
        keep = max(8, int(h * DOPPLER_CROP_FRAC))
        lo = (h - keep) // 2
        mag = mag[lo:lo + keep]
    db = 20 * np.log10(mag + 1e-9)
    db = np.clip(db, db.max() - DB_DYNAMIC_RANGE, db.max())
    norm = (db - db.min()) / (db.max() - db.min() + 1e-9)
    img = Image.fromarray((norm * 255).astype(np.uint8)).resize(
        SPECTROGRAM_SIZE[::-1], Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def array_to_spectrogram(arr: np.ndarray, fs: int) -> np.ndarray:
    """Turn a loaded file array into a normalized spectrogram image, deciding
    whether it is an already-computed 2-D map or raw 1-D IQ that needs an STFT."""
    arr = np.asarray(arr).squeeze()
    fmt = DATA_FORMAT
    if fmt == "auto":
        # A 2-D complex matrix is always a time-frequency map -> take |.|.
        # For real 2-D: trailing dim of 2 is an [I, Q] pair (raw IQ); anything
        # else with both dims >= 8 is treated as a magnitude spectrogram.
        if arr.ndim == 2 and np.iscomplexobj(arr) and arr.shape[-1] != 2:
            fmt = "spectrogram"
        elif arr.ndim == 2 and arr.shape[-1] != 2 and min(arr.shape) >= 8:
            fmt = "spectrogram"
        else:
            fmt = "iq"

    if fmt == "spectrogram":
        return _spectrogram_image(arr)

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
        S = array_to_spectrogram(arr, fs=SAMPLING_RATES[band])
        _LOAD_STATS["ok"] += 1
        return S
    except Exception as e:
        _LOAD_STATS["failed"] += 1
        key = f"{type(e).__name__}: {e}"
        _LOAD_STATS["errors"][key] = _LOAD_STATS["errors"].get(key, 0) + 1
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
    entries = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        band = _folder_to_band(folder.name)
        cid = _folder_to_class(folder.name)
        if band not in bands or cid < 0: continue
        for f in folder.rglob("*"):
            if f.suffix.lower() in (".mat", ".dat"):
                entries.append((f, cid, band))
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
        H, W = S.shape
        tm = int(np.random.uniform(0, AUG_TIME_MASK) * W)
        if tm:
            t0 = np.random.randint(0, W - tm); S[:, t0:t0+tm] = 0
        fm = int(np.random.uniform(0, AUG_FREQ_MASK) * H)
        if fm:
            f0 = np.random.randint(0, H - fm); S[f0:f0+fm, :] = 0
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
        x = torch.from_numpy(np.stack([S, S, S], axis=0))
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
            n = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
            return nn.Sequential(*list(n.children())[:-2]), 2048
        if name == "resnet18":
            n = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            return nn.Sequential(*list(n.children())[:-2]), 512
        if name == "efficientnet_b0":
            n = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            return n.features, 1280
        if name == "convnext_tiny":
            n = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            return n.features, 768
        if name == "mobilenet_v3_small":
            n = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
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
    class CNN_LSTM(nn.Module):
        def __init__(self, dropout=0.3):
            super().__init__()
            self.cnn, feat = _cnn_backbone("resnet18")
            self.lstm = nn.LSTM(feat, 256, 2, batch_first=True, dropout=dropout)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES))
        def forward(self, x):
            f = self.cnn(x).mean(dim=2).permute(0, 2, 1)
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
            f = self.cnn(x).mean(dim=2).permute(0, 2, 1)
            h, _ = self.bi(f); w = torch.softmax(self.attn(h), dim=1)
            return self.head((w * h).sum(dim=1))

    # ---- 3. ResNet18 + Linear Head -----------------------------------
    # REPLACED ResNet50_Transformer (which overfit & collapsed to ~11%).
    # ResNet18 is much smaller -> less overfitting on ~600 train samples;
    # a plain linear head trains quickly and stably.
    class ResNet18_Head(nn.Module):
        def __init__(self, dropout=0.5):
            super().__init__()
            self.cnn, feat = _cnn_backbone("resnet18")
            self.pool = nn.AdaptiveAvgPool2d(1)
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
            swin = tvm.swin_t(weights=tvm.Swin_T_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(swin.children())[:-3])
            self.norm = nn.LayerNorm(768)
            self.lstm = nn.LSTM(768, 256, 2, batch_first=True, dropout=dropout, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, NUM_CLASSES))
        def forward(self, x):
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
            self.proj = nn.Linear(feat, 256)
            self.tx = TxBlock(256, 8, 1024, dropout)
            self.enc = _VAEEnc(latent); self.dec = _VAEDec(latent)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(256 + latent, NUM_CLASSES))
        def forward(self, x):
            f = self.tx(self.proj(self.cnn(x).mean(dim=2).permute(0, 2, 1))).mean(dim=1)
            z, mu, lv = self.enc(x); x_hat = self.dec(z)
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
            # decoder produces 224x224 — match input size
            return F.mse_loss(x_hat, x)

    # ---- 10. DenseNet121 + GRU --------------------------------------
    # REPLACED CNN_Transformer_LSTM (triple hybrid that collapsed to ~8%).
    # DenseNet's feature-reuse pattern handles small datasets well; a single
    # GRU is much cheaper than the stacked Transformer+LSTM that overfit.
    class DenseNet_GRU(nn.Module):
        def __init__(self, dropout=0.4):
            super().__init__()
            dn = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT)
            self.cnn = dn.features
            self.proj = nn.Linear(1024, 256)
            self.gru = nn.GRU(256, 192, 2, batch_first=True,
                              dropout=dropout, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(384),
                                      nn.Dropout(dropout),
                                      nn.Linear(384, NUM_CLASSES))
        def forward(self, x):
            f = self.cnn(x)
            f = F.relu(f).mean(dim=2).permute(0, 2, 1)  # (B, W, 1024)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # subcommand is optional now: running the script with no args defaults to
    # a full training run over all bands and all 10 models.
    sub = ap.add_subparsers(dest="cmd", required=False)

    t = sub.add_parser("train", help="Train all models on selected protocols")
    t.add_argument("--bands", nargs="+",
                   default=["10ghz", "24ghz", "77ghz", "combined"])
    t.add_argument("--models", nargs="+", default=["all"])
    t.add_argument("--tune", action="store_true",
                   help="Run Optuna hyperparameter search per model")
    # NEW: uncertainty + physics toggles
    t.add_argument("--loss", choices=["ce", "edl"], default="ce",
                   help="Classification loss. EDL = evidential deep learning "
                        "(Dirichlet head with KL regularizer).")
    t.add_argument("--physics-informed", action="store_true",
                   help="Add Doppler-symmetry consistency loss (5%% weight).")
    t.add_argument("--mc-passes", type=int, default=MC_DROPOUT_PASSES,
                   help="MC Dropout passes at test (1 = off).")
    t.add_argument("--no-temperature", action="store_true",
                   help="Skip post-hoc temperature scaling.")

    r = sub.add_parser("replot", help="Regenerate SVGs from saved JSON results")
    r.add_argument("--results-dir", default="./results")

    # NEW: the 2x2x3 ablation grid
    abl = sub.add_parser("ablate",
                         help="Run the 2x2x3 ablation grid "
                              "(loss × physics × pretrain) on selected models/bands. "
                              "Produces per-(band,model) detail tables, per-band summaries "
                              "with top-3/top-5 ensembles, and a master comparison.")
    abl.add_argument("--bands", nargs="+",
                     default=["10ghz", "24ghz", "77ghz", "combined"],
                     help="Bands to ablate (default: all four).")
    abl.add_argument("--models", nargs="+",
                     default=ABLATION_DEFAULT_MODELS,
                     help="Model names from the registry. "
                          f"Default: 8 plain classifiers ({len(ABLATION_DEFAULT_MODELS)} models).")
    abl.add_argument("--quick", action="store_true",
                     help="Only 4 configs (loss × physics, no pretrain axis) "
                          "instead of the full 12.")

    args = ap.parse_args()

    # No subcommand given (e.g. pressing "Run Python File" in VSCode) ->
    # use DEFAULT_ACTION set at the top of this file.
    if args.cmd is None:
        if DEFAULT_ACTION == "ablate":
            print("[info] No arguments given - running the FULL ablation grid:")
            print("       8 models × 12 configs (loss × physics × pretrain) × 4 bands")
            print("       = 384 trainings. This is a long run; results are saved")
            print(f"       saved to {ABLATION_RESULTS_DIR} band-by-band as it goes.")
            args.cmd = "ablate"
            args.bands = ["10ghz", "24ghz", "77ghz", "combined"]
            args.models = ABLATION_DEFAULT_MODELS
            args.quick = False
        else:
            print("[info] No arguments given - running a normal training pass "
                  "over all bands and models.")
            args.cmd = "train"
            args.bands = ["10ghz", "24ghz", "77ghz", "combined"]
            args.models = ["all"]
            args.tune = False
            args.loss = "ce"
            args.physics_informed = False
            args.mc_passes = MC_DROPOUT_PASSES
            args.no_temperature = False

    {"train": cmd_train, "replot": cmd_replot, "ablate": cmd_ablate}[args.cmd](args)


if __name__ == "__main__":
    main()
