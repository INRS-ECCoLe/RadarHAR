"""
diagnose_data.py
================
Run this BEFORE trusting any training results. It answers the only questions
that matter when every model is stuck near chance:

  1. Are the files actually being read, or silently turned into zeros?
  2. What is INSIDE each .mat/.dat? (variable names, shapes, dtypes)
  3. Is the data RAW IQ (complex / 1-D  -> needs STFT) or ALREADY A
     SPECTROGRAM (real 2-D -> must NOT be re-STFT'd)?
  4. Do the generated images actually look like micro-Doppler signatures?

It samples one file per class per band, prints a full report, and saves PNG
previews under ./diagnostics/ so you can eyeball them.

Usage:
    python diagnose_data.py
    python diagnose_data.py --root "C:\\path\\to\\Data" --per-class 2
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- keep these in sync with the main script ----
DATA_ROOT = Path(r"C:\Users\atikm\Downloads\CSI RADAR\Open Radar Datasets (OpenRadarInitiative)\Raw Data_raw_data\Cross-frequency\Data")
FREQ_BANDS = ["10ghz", "24ghz", "77ghz"]
ACTIVITY_IDS = ["05", "06", "07", "08", "09", "10", "11", "16", "17", "18", "19"]
OUT = Path("./diagnostics")


def normalize(s): return re.sub(r"[\s_]+", "_", s.strip().lower())

def folder_band(name):
    n = normalize(name)
    for b in FREQ_BANDS:
        if n.startswith(b): return b
    return ""

def folder_activity(name):
    m = re.search(r"(?:10|24|77)ghz[_\- ]?(\d{2})", normalize(name))
    return m.group(1) if m else ""


def describe_mat(path: Path):
    """Return (report_str, list_of_(varname, ndarray)) for a .mat file."""
    mat = sio.loadmat(str(path), squeeze_me=True)
    lines = [f"  scipy.io keys: {[k for k in mat if not k.startswith('__')]}"]
    arrays = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            arrays.append((k, v))
            lines.append(f"    '{k}': shape={v.shape}  dtype={v.dtype}  "
                         f"complex={np.iscomplexobj(v)}  "
                         f"min={np.nanmin(v.real):.3g}  max={np.nanmax(v.real):.3g}")
        else:
            lines.append(f"    '{k}': {type(v).__name__} = {repr(v)[:60]}")
    if not arrays:
        # maybe it's a v7.3 / HDF5 mat file
        lines.append("    (no plain arrays found - may be a v7.3/HDF5 .mat; "
                     "try h5py or `mat = mat73.loadmat(path)`)")
    return "\n".join(lines), arrays


def describe_dat(path: Path):
    size = path.stat().st_size
    lines = [f"  raw size: {size} bytes"]
    for dtype in (np.int16, np.float32, np.complex64):
        n = size // np.dtype(dtype).itemsize
        lines.append(f"    as {np.dtype(dtype).name}: {n} elements "
                     f"({'even' if n % 2 == 0 else 'odd'} -> "
                     f"{'IQ-interleavable' if n % 2 == 0 else 'not IQ-interleavable'})")
    arr16 = np.fromfile(str(path), dtype=np.int16)[:8]
    arrf = np.fromfile(str(path), dtype=np.float32)[:8]
    lines.append(f"    first int16 vals : {arr16.tolist()}")
    lines.append(f"    first float32 vals: {[round(float(x), 3) for x in arrf]}")
    return "\n".join(lines), None


def make_previews(name, arrays, out_dir):
    """Save candidate visualizations so the user can see which interpretation is right."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not arrays:
        return
    # pick the largest array - usually the data
    k, v = max(arrays, key=lambda kv: kv[1].size)
    v = np.asarray(v).squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{name}  (var '{k}', shape {v.shape}, "
                 f"{'complex' if np.iscomplexobj(v) else 'real'})")

    # Interpretation A: already a 2-D spectrogram -> show as-is (log scale)
    try:
        if v.ndim == 2:
            mag = np.abs(v).astype(float)
            disp = 20 * np.log10(mag + 1e-9)
            axes[0].imshow(disp, aspect="auto", cmap="jet", origin="lower")
            axes[0].set_title("A) treat as spectrogram (dB)")
        else:
            axes[0].text(0.5, 0.5, "not 2-D\n(not a ready spectrogram)",
                         ha="center", va="center"); axes[0].axis("off")
    except Exception as e:
        axes[0].text(0.5, 0.5, f"err: {e}", ha="center"); axes[0].axis("off")

    # Interpretation B: raw IQ -> STFT
    try:
        from scipy.signal import stft
        if np.iscomplexobj(v):
            iq = v.ravel()
        elif v.ndim == 2 and v.shape[-1] == 2:
            iq = v[..., 0] + 1j * v[..., 1]
        elif v.ndim == 2:
            iq = v.sum(axis=0).astype(complex)   # collapse fast-time
        else:
            iq = v.astype(complex)
        f, t, Z = stft(iq, nperseg=256, noverlap=200, nfft=512, return_onesided=False)
        Z = np.fft.fftshift(Z, axes=0)
        axes[1].imshow(20 * np.log10(np.abs(Z) + 1e-9), aspect="auto",
                       cmap="jet", origin="lower")
        axes[1].set_title("B) treat as raw IQ -> STFT")
    except Exception as e:
        axes[1].text(0.5, 0.5, f"err: {e}", ha="center"); axes[1].axis("off")

    # Interpretation C: raw 1-D time series plot
    try:
        flat = v.ravel()
        axes[2].plot(np.real(flat[:2000]))
        axes[2].set_title("C) first 2000 samples (real part)")
    except Exception as e:
        axes[2].text(0.5, 0.5, f"err: {e}", ha="center"); axes[2].axis("off")

    fig.tight_layout()
    safe = re.sub(r"[^\w\-]", "_", name)
    fig.savefig(out_dir / f"{safe}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_ROOT))
    ap.add_argument("--per-class", type=int, default=1,
                    help="files to sample per class/band")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: data root does not exist:\n  {root}")
        sys.exit(1)

    OUT.mkdir(exist_ok=True, parents=True)
    print(f"Scanning {root}\nPreviews will be saved to {OUT.resolve()}\n")

    total, ok, failed = 0, 0, 0
    seen = {}   # (band, activity) -> count

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        band = folder_band(folder.name)
        act = folder_activity(folder.name)
        if band not in FREQ_BANDS or act not in ACTIVITY_IDS:
            continue
        files = [f for f in folder.rglob("*") if f.suffix.lower() in (".mat", ".dat")]
        if not files:
            print(f"[!] {folder.name}: NO .mat/.dat files inside")
            continue
        key = (band, act)
        for f in files[:args.per_class]:
            if seen.get(key, 0) >= args.per_class:
                break
            seen[key] = seen.get(key, 0) + 1
            total += 1
            print(f"\n=== {folder.name} -> {f.name} ===")
            try:
                if f.suffix.lower() == ".mat":
                    report, arrays = describe_mat(f)
                else:
                    report, arrays = describe_dat(f)
                print(report)
                make_previews(f"{band}_{act}_{f.stem}", arrays or [], OUT)
                ok += 1
            except Exception as e:
                failed += 1
                print(f"  PARSE FAILED: {type(e).__name__}: {e}")

    # coverage summary
    print("\n" + "=" * 60)
    print(f"Files inspected : {total}   parsed OK: {ok}   failed: {failed}")
    missing = [(b, a) for b in FREQ_BANDS for a in ACTIVITY_IDS
               if (b, a) not in seen]
    if missing:
        print(f"\n[!] No folder found for these (band, activity) pairs:")
        for b, a in missing:
            print(f"    {b}  activity {a}")
    print(f"\nNow OPEN the PNGs in {OUT.resolve()} and check:")
    print("  - Does panel A (spectrogram-as-is) look like a micro-Doppler signature")
    print("    (a bright horizontal-ish band with limb 'flares' around it)?")
    print("    -> if YES, your files are ALREADY spectrograms. Skip the STFT.")
    print("  - Or does panel B (IQ->STFT) look like a signature instead?")
    print("    -> if YES, your files are raw IQ and the STFT path is correct.")
    print("  - If BOTH look like noise, the variable name / dtype guess is wrong.")


if __name__ == "__main__":
    main()
