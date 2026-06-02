"""
dat2mat.py
==========
Convert CI4R-MULTI3 10 GHz (XeThru) and 24 GHz (Ancortek) .dat files into
.mat files containing variable 'sx1' — the same convention used by the 77 GHz
files. Once converted, point DATA_ROOT in radar_har_single.py at the OUTPUT
directory and the training pipeline runs without any other change.

The processing pipelines exactly mirror the lab's MATLAB scripts:
  - datToImage_Anchortech.m  (24 GHz ASCII -> Ancortek pipeline)
  - XeThru_mDopp_bulk.m      (10 GHz binary -> XeThru pipeline)

Usage:
    python dat2mat.py --root  "C:\\Users\\atikm\\...\\Data" \\
                      --out   "C:\\Users\\atikm\\...\\Data_converted" \\
                      --copy-77 \\
                      --preview 1
"""
from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import butter, lfilter, stft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# 24 GHz - Ancortek SDR (FMCW, ASCII text .dat)
# =====================================================================
_FLOAT_RE = re.compile(rb"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?")

def _looks_like_text(path: Path, sniff_bytes: int = 4096) -> bool:
    """True if the first sniff_bytes of the file are mostly printable ASCII."""
    with open(path, "rb") as f:
        chunk = f.read(sniff_bytes)
    if not chunk:
        return False
    printable = sum(1 for b in chunk if b in b"\r\n\t " or 33 <= b <= 126)
    return printable / len(chunk) > 0.90

def _read_24ghz_floats(path: Path) -> np.ndarray:
    """MATLAB textscan('%f') equivalent. Reads every float token regardless of
    line layout. Falls back to binary if the file is not text."""
    if _looks_like_text(path):
        with open(path, "rb") as f:
            blob = f.read()
        toks = _FLOAT_RE.findall(blob)
        if not toks:
            raise ValueError(f"No numeric tokens found in {path.name}")
        return np.fromiter((float(t) for t in toks), dtype=np.float64, count=len(toks))
    # Binary fallback: Ancortek files are sometimes int16 (most common) or float32.
    raw_i16 = np.fromfile(str(path), dtype=np.int16).astype(np.float64)
    raw_f32 = np.fromfile(str(path), dtype=np.float32).astype(np.float64)
    # Pick whichever has more plausible dynamic range (i.e., not all zeros / all NaN)
    candidates = [(raw_i16, "int16"), (raw_f32, "float32")]
    candidates = [(arr, tag) for arr, tag in candidates
                  if arr.size > 100 and np.isfinite(arr).all() and arr.std() > 0]
    if not candidates:
        raise ValueError(f"Could not parse {path.name} as text or binary")
    # Prefer int16 for Ancortek by default
    return candidates[0][0]


def process_24ghz_ancortek(dat_path: Path) -> np.ndarray:
    """Replicates datToImage_Anchortech.m. Returns sx1 complex (freq, time)."""
    radar = _read_24ghz_floats(dat_path)
    if radar.size < 5:
        raise ValueError(f"Too few samples ({radar.size}) in {dat_path.name}")

    fc       = radar[0]                # Hz, center frequency
    Tsweep   = radar[1] / 1000.0       # ms -> s
    NTS      = int(radar[2])           # samples per sweep
    Bw       = radar[3]                # Hz
    Data     = radar[4:]

    if NTS <= 0 or NTS > 100_000:
        raise ValueError(f"Implausible NTS={NTS} (header may be wrong format)")
    if Tsweep <= 0:
        raise ValueError(f"Implausible Tsweep={Tsweep}s (header may be wrong format)")
    fs = NTS / Tsweep

    nc = len(Data) // NTS
    if nc < 8:
        raise ValueError(f"Only {nc} chirps (NTS={NTS}, data_len={len(Data)})")
    Data = Data[:NTS * nc]

    # MATLAB reshape(Data,[NTS nc]) is column-major. In numpy: order='F'.
    Data_time = Data.reshape((NTS, nc), order="F")

    # Range FFT (1st FFT) along the fast-time axis
    tmp = np.fft.fftshift(np.fft.fft(Data_time, axis=0), axes=0)
    # MATLAB: Data_range = tmp(NTS/2+1:NTS, :)  -> upper half, positive freqs
    Data_range = tmp[NTS // 2 : NTS, :]

    # MTI: 4th-order Butterworth high-pass with cutoff 0.01 (normalized).
    # MATLAB used `filter(b,a,...)` (single-pass), so we use lfilter here.
    b, a = butter(4, 0.01, btype="high")
    Data_range_MTI = np.zeros_like(Data_range, dtype=np.complex128)
    for k in range(Data_range.shape[0]):
        Data_range_MTI[k, :] = lfilter(b, a, Data_range[k, :])

    # Spectrogram from the K most-active range bins (was hard-coded bin 19).
    # Same robustness fix as 10 GHz — the best bin depends on subject distance.
    range_signal = _select_active_range_bins(Data_range_MTI,
                                             search_range=(3, 60), k=5)
    # MATLAB: window=128, noverlap=100, nfft=2^12=4096.
    _, _, sx = stft(range_signal, fs=fs,
                    nperseg=128, noverlap=100, nfft=4096,
                    return_onesided=False, boundary=None, padded=False)
    # MATLAB: sx1 = flipud(fftshift(sx, 1))
    sx1 = np.flipud(np.fft.fftshift(sx, axes=0))
    return sx1.astype(np.complex64)


# =====================================================================
# 10 GHz - XeThru UWB (binary frame stream .dat)
# =====================================================================
def _xethru_read_frames(path: Path) -> np.ndarray:
    """Read XeThru binary file. Each frame: 12-byte hdr + 182 float32. Returns
    a (182, n_frames) float32 array."""
    frames = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)             # 3 * uint32
            if len(hdr) < 12:
                break
            data = f.read(182 * 4)       # 182 float32
            if len(data) < 182 * 4:
                break
            frames.append(np.frombuffer(data, dtype="<f4").copy())
    if not frames:
        raise ValueError(f"No frames decoded from {path.name}")
    return np.stack(frames, axis=1)      # shape (182, n_frames)


def _select_active_range_bins(matrix_after_mti: np.ndarray,
                              search_range=(3, 60), k: int = 5) -> np.ndarray:
    """Pick the K range bins with the highest post-MTI temporal variance, then
    sum them coherently. This is the single biggest robustness fix vs. picking
    one fixed bin — the active bin depends on where the subject stood, and
    summing the top K gives a much stronger signal than guessing one.

    Returns a 1-D complex slow-time signal."""
    lo, hi = search_range
    hi = min(hi, matrix_after_mti.shape[0])
    bin_var = np.var(np.abs(matrix_after_mti[lo:hi]), axis=1)
    if not np.any(bin_var > 0):
        return matrix_after_mti[matrix_after_mti.shape[0] // 4]  # fallback
    top = np.argsort(-bin_var)[:k]
    chosen = matrix_after_mti[lo + top]
    # phase-align before summing: align each row's mean phase to the strongest row
    strongest = chosen[0]
    aligned = [strongest]
    for i in range(1, len(chosen)):
        ph = np.angle(np.vdot(strongest, chosen[i]))
        aligned.append(chosen[i] * np.exp(-1j * ph))
    return np.sum(aligned, axis=0)


def process_10ghz_xethru(dat_path: Path) -> np.ndarray:
    """Replicates XeThru_mDopp_bulk.m with adaptive range-bin selection.
    The original script hard-coded bin 27 ('corner') — fine for the lab's
    setup but wrong for other recording geometries. Returns sx1 complex
    (freq, time)."""
    stream = _xethru_read_frames(dat_path)         # (182, n_frames)
    half = stream.shape[0] // 2                    # 91 range bins
    I = stream[:half, :]
    Q = stream[half:half * 2, :]
    iq = (I + 1j * Q).astype(np.complex128)        # (91, n_frames)

    # Slightly relaxed MTI cutoff (was 0.01, now 0.005) — XeThru's slow PRF
    # means low-Doppler activity content can be filtered out too aggressively.
    b, a = butter(4, 0.005, btype="high")
    iq_mti = np.zeros_like(iq)
    for k in range(iq.shape[0]):
        iq_mti[k, :] = lfilter(b, a, iq[k, :])

    # Adaptive: pick the most-active 5 range bins and sum them.
    range_signal = _select_active_range_bins(iq_mti, search_range=(3, 60), k=5)

    PRF = 512
    _, _, sx = stft(range_signal, fs=PRF,
                    nperseg=150, noverlap=128, nfft=4096,
                    return_onesided=False, boundary=None, padded=False)
    sx1 = np.fft.fftshift(sx, axes=0)
    return sx1.astype(np.complex64)


# =====================================================================
# Preview rendering (matches 77 GHz Panel A look)
# =====================================================================
def save_preview(sx1: np.ndarray, label: str, out_png: Path):
    mag = np.abs(sx1)
    db = 20 * np.log10(mag + 1e-9)
    db = np.clip(db, db.max() - 50, db.max())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(db, aspect="auto", cmap="jet", origin="lower")
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("time"); ax.set_ylabel("Doppler / freq")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# =====================================================================
# Debug-one: detailed diagnosis of a single problematic .dat file
# =====================================================================
def debug_one_file(path: Path):
    """Print a verbose layout report for one .dat file. Use this when the
    converter fails — paste the output to identify the actual file format."""
    if not path.exists():
        print(f"ERROR: file not found: {path}"); return
    size = path.stat().st_size
    print(f"=== Debugging {path} ===")
    print(f"file size: {size:,} bytes")

    with open(path, "rb") as f:
        head = f.read(256)
    # ASCII-ness
    printable = sum(1 for b in head if b in b"\r\n\t " or 33 <= b <= 126)
    pct = 100.0 * printable / max(len(head), 1)
    print(f"first 256 bytes: {pct:.1f}% printable ASCII -> "
          f"{'likely TEXT' if pct > 90 else 'likely BINARY'}")
    print(f"first bytes (hex): {head[:64].hex(' ')}")
    try:
        print(f"first bytes (ascii): {head[:200].decode('ascii', errors='replace')!r}")
    except Exception:
        pass

    # Try ASCII parse
    print("\n-- attempt: ASCII regex parse --")
    try:
        with open(path, "rb") as f:
            blob = f.read()
        toks = _FLOAT_RE.findall(blob)
        print(f"  found {len(toks)} float tokens")
        if toks:
            vals = np.fromiter((float(t) for t in toks[:20]), dtype=np.float64,
                               count=min(20, len(toks)))
            print(f"  first 20 values: {vals}")
            if len(toks) >= 5:
                all_vals = np.fromiter((float(t) for t in toks), dtype=np.float64,
                                       count=len(toks))
                print(f"  total {len(all_vals)} values; "
                      f"range [{all_vals.min():.3g}, {all_vals.max():.3g}]")
                print(f"  interpreted header: fc={all_vals[0]:.3g}  "
                      f"Tsweep_ms={all_vals[1]:.3g}  NTS={int(all_vals[2])}  "
                      f"Bw={all_vals[3]:.3g}")
                data_len = len(all_vals) - 4
                nts = int(all_vals[2])
                if nts > 0:
                    nc = data_len // nts
                    print(f"  -> {nc} chirps would be produced from {data_len} samples")
    except Exception as e:
        print(f"  ASCII attempt failed: {type(e).__name__}: {e}")

    # Try binary parses
    print("\n-- attempt: binary parses --")
    with np.errstate(over="ignore", invalid="ignore"):
        for dtype in (np.int16, np.int32, np.float32, np.float64):
            try:
                arr = np.fromfile(str(path), dtype=dtype)
                if arr.size == 0:
                    continue
                finite = arr[np.isfinite(arr)] if dtype in (np.float32, np.float64) else arr
                if finite.size == 0:
                    print(f"  as {np.dtype(dtype).name:8s}: all non-finite (not this dtype)")
                    continue
                print(f"  as {np.dtype(dtype).name:8s}: {arr.size:,} elements,  "
                      f"range [{float(finite.min()):.3g}, {float(finite.max()):.3g}],  "
                      f"std={float(finite.std()):.3g}")
            except Exception as e:
                print(f"  as {np.dtype(dtype).name}: failed ({e})")

    print("\nIf ASCII parsing found a sensible header (fc ~ 24e9, "
          "Tsweep_ms in 0.1-10, NTS in 50-2000) the file is ASCII Ancortek and "
          "the converter should now read it. Otherwise paste this output back.")


# =====================================================================
# Orchestration
# =====================================================================
DEFAULT_ROOT = r"C:\Users\atikm\Downloads\CSI RADAR\Open Radar Datasets (OpenRadarInitiative)\Raw Data_raw_data\Cross-frequency\Data"
DEFAULT_OUT  = r"C:\Users\atikm\Downloads\CSI RADAR\Data_converted"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Both optional now: bare `python dat2mat.py` uses the defaults below.
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"Original Data folder.  (default: {DEFAULT_ROOT})")
    ap.add_argument("--out",  default=DEFAULT_OUT,
                    help=f"Output folder, created if missing.  (default: {DEFAULT_OUT})")
    ap.add_argument("--copy-77", action="store_true",
                    help="Also copy existing 77 GHz .mat files into --out.")
    ap.add_argument("--preview", type=int, default=1,
                    help="Save N preview PNGs per folder (0 = none).")
    ap.add_argument("--debug-one", default=None,
                    help="Path to a single .dat file. Prints a verbose layout "
                         "diagnosis and exits. Use this when conversion fails.")
    args = ap.parse_args()

    if args.debug_one:
        debug_one_file(Path(args.debug_one)); return

    print(f"[info] root = {args.root}")
    print(f"[info] out  = {args.out}")
    print(f"[info] copy-77 = {args.copy_77}   preview = {args.preview}\n")

    root = Path(args.root); out_root = Path(args.out)
    if not root.exists():
        print(f"ERROR: --root does not exist: {root}"); sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)
    prev_dir = out_root / "_previews"
    if args.preview > 0:
        prev_dir.mkdir(exist_ok=True)

    totals = {"10ghz": [0, 0], "24ghz": [0, 0], "77ghz": [0, 0]}
    errors_by_band = {"10ghz": {}, "24ghz": {}, "77ghz": {}}
    failed_examples = {"10ghz": [], "24ghz": [], "77ghz": []}

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        nlow = folder.name.lower()
        if nlow.startswith("10ghz"):
            band, processor = "10ghz", process_10ghz_xethru
        elif nlow.startswith("24ghz"):
            band, processor = "24ghz", process_24ghz_ancortek
        elif nlow.startswith("77ghz"):
            band, processor = "77ghz", None
        else:
            continue

        out_folder = out_root / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        previewed = 0

        if band == "77ghz":
            if not args.copy_77:
                continue
            for f in sorted(folder.rglob("*.mat")):
                shutil.copy2(f, out_folder / f.name)
                totals["77ghz"][0] += 1
            print(f"[77ghz] {folder.name}: copied {totals['77ghz'][0]} .mat files")
            continue

        files = sorted(folder.rglob("*.dat"))
        if not files:
            print(f"[{band}] {folder.name}: NO .dat files inside"); continue

        ok = fail = 0
        for f in files:
            try:
                sx1 = processor(f)
                mat_path = out_folder / (f.stem + ".mat")
                savemat(str(mat_path), {"sx1": sx1}, do_compression=True)
                ok += 1
                if previewed < args.preview:
                    save_preview(sx1, f"{folder.name} / {f.name}\nshape {sx1.shape}",
                                 prev_dir / f"{folder.name}__{f.stem}.png")
                    previewed += 1
            except Exception as e:
                fail += 1
                key = f"{type(e).__name__}: {e}"
                errors_by_band[band][key] = errors_by_band[band].get(key, 0) + 1
                if len(failed_examples[band]) < 3:
                    failed_examples[band].append((f, key))
        totals[band][0] += ok; totals[band][1] += fail
        print(f"[{band}] {folder.name}: converted {ok}/{ok+fail} "
              f"-> {out_folder}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    for band, (ok, fail) in totals.items():
        if ok + fail:
            print(f"  {band:6s}  ok={ok:4d}   failed={fail:4d}")
    # Per-band breakdown of error types
    for band, errs in errors_by_band.items():
        if not errs:
            continue
        print(f"\n[{band}] error types (top 5):")
        for err, n in sorted(errs.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {n:5d}x  {err}")
        print(f"[{band}] example failed files:")
        for path, err in failed_examples[band]:
            print(f"    {path.name}")
        print(f"[{band}] -> Re-run with --debug-one on one of these files to diagnose:")
        print(f'    python dat2mat.py --debug-one "{failed_examples[band][0][0]}"')
    if args.preview > 0:
        print(f"\nPreview PNGs in {prev_dir.resolve()}")
        print("OPEN A FEW. If they look like the 77 GHz Panel A (a bright "
              "center band with limb 'flares'), the conversion is correct.")
    print(f"\nNext step:  point DATA_ROOT in radar_har_single.py at:")
    print(f"    {out_root.resolve()}")
    print(f"...then run training as usual.")


if __name__ == "__main__":
    main()
