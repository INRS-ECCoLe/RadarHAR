# Radar-Based Human Motion Detection

Deep-learning classification of **human activities** from **radar micro-Doppler
spectrograms**, across three radar frequency bands and a combined setting.

This repository trains and compares many neural-network architectures — from
lightweight CNN+RNN hybrids to Vision Transformers (ViT) — on the same radar
dataset, under a controlled **ablation grid**, and reports accuracy, F1, and
confusion matrices for each. Everything is built to run on the
[Digital Research Alliance of Canada](https://alliancecan.ca/) **Narval**
cluster (SLURM + GPU).

---

## 1. What problem does this solve?

A radar sensor pointed at a person produces a **micro-Doppler signature**: as
different body parts (torso, arms, legs) move at different velocities, they
shift the radar echo's frequency by different amounts. Plotting that
frequency shift over time gives a **spectrogram** — a 2-D image that looks
different for walking vs. sitting vs. crawling, etc.

The task: **given one spectrogram, predict which of 11 activities it is.**

### The 11 activities (classes)

| ID | Code   | Activity |
|----|--------|----------|
| 0  | WLKT   | Walking towards radar |
| 1  | WALKA  | Walking away from radar |
| 2  | PICK   | Picking up an object |
| 3  | BEND   | Bending |
| 4  | SIT    | Sitting on a chair |
| 5  | KNEEL  | Kneeling |
| 6  | CRWL   | Crawling towards radar |
| 7  | WTOES  | Walking on both toes |
| 8  | LIMP   | Limping with right leg stiff |
| 9  | SHTEPS | Walking with short steps |
| 10 | SCSSR  | Scissors gait |

### The data

- Source: CI4R / OpenRadarInitiative cross-frequency micro-Doppler dataset.
- Each recording is a MATLAB `.mat` file holding a complex
  (frequency × time) micro-Doppler matrix (variable `sx1`, ~4096 × 538).
- Three radar bands are treated as separate experiments, plus a **combined**
  setting that pools all three:

  ```
  FREQ_BANDS = ["10ghz", "24ghz", "77ghz", "combined"]
  ```

- Data is **not** included in this repo. Point the scripts at your local copy
  with `--data-root` (folders are auto-discovered by name, e.g. `10ghz_05...`).

---

## 2. Repository contents

| File | What it is |
|------|-----------|
| `Main.py` | Full single-file pipeline (data loading → training → evaluation → plots) for local experimentation. |
| `Narval_RADAR_Motion_OG.py` | The original ("OG") Narval training pipeline with the full set of hybrid CNN/RNN/Transformer models. |
| `Narval_radar_motion_transformer.py` | Transformer-focused variant of the Narval pipeline. |
| `Diagnose.py` | Inspects `.mat` files (shapes, variable names) to confirm how the data should be read. |
| `dat2mat.py` | Converts raw `.dat` captures to `.mat`. |
| `LICENSE`, `README.md` | Housekeeping. |

> **Naming note.** Files prefixed `Narval_` are meant to be submitted as SLURM
> jobs on the Narval cluster. `Main.py` is the convenient all-in-one version
> for running locally or on Colab.

There are also **model-specific Vision-Transformer pipelines** (each a
self-contained `.py` plus a matching `.sh` SLURM script). These share the same
data pipeline, ablation grid, and output format as the OG pipeline, but swap in
a specific pretrained backbone and its paper's fine-tuning recipe. See
[Section 6](#6-vision-transformer-pipelines).

---

## 3. How a model gets from a `.mat` file to a prediction

```
.mat file (4096 x 538 complex)
        │
        ▼
  magnitude + dB scale + normalise          # to a clean [0,1] image
        │
        ▼
  Doppler-region selection                  # keep the signal-carrying rows
        │
        ▼
  smart downscaling to 224 x 224            # 3 methods compared (below)
        │
        ▼
  neural network (CNN / RNN / Transformer)
        │
        ▼
  softmax over 11 classes -> predicted activity
```

### Smart downscaling (3 methods, compared head-to-head)

The raw spectrogram is far larger than the network input (224 × 224), and naive
resizing smears the sharp micro-Doppler peaks. Three strategies are compared:

- **`mixed`** — blends max-pooling (keeps peaks) with average-pooling (keeps
  energy) into one channel.
- **`multichannel`** *(default)* — a 3-channel image: `Ch0` = average (energy
  context), `Ch1` = mixed (balanced), `Ch2` = max (peak detector).
- **`autoencoder`** — builds a slightly larger (280 × 280) multichannel image,
  then a small learned convolutional stem compresses it to 224 × 224, trained
  jointly with the classifier.

---

## 4. The ablation grid

Every model is trained under **every combination** of these switches, so the
effect of each choice can be measured fairly:

| Axis | Options | Meaning |
|------|---------|---------|
| **Loss** | `ce`, `edl` | Standard cross-entropy vs. Evidential Deep Learning (gives an uncertainty estimate). |
| **Physics-prior** | `off`, `on` | Optional loss term encouraging consistent predictions when the Doppler axis is flipped (cyclic-motion symmetry). |
| **Pretrain (aux)** | `none`, `denoising` | Optional self-supervised consistency loss (predict the same class from a noised input). |
| **Downscaling** | `mixed`, `multichannel`, `autoencoder` | The three methods above. |

That's **2 × 2 × 2 × 3 = 24 configurations per (model, band)**.

### How predictions are scored at test time

Two inference modes are reported **side by side** for every configuration:

- **`softmax`** — a single deterministic forward pass.
- **`TTA + MC-dropout`** — test-time augmentation (flips/shifts) combined with
  Monte-Carlo dropout (several stochastic passes averaged), which can improve
  accuracy and gives a rough confidence estimate.

---

## 5. Outputs

Each run writes, per band and per model:

- `*_result.json` — full record per configuration (true labels, predictions,
  probabilities, hyper-parameters) so results can be re-analysed without
  re-training.
- `*.pth` — trained model checkpoints.
- `comparison.md` / `comparison.json` / `comparison.csv` — ranked table of all
  24 configs.

Then `--aggregate` builds the top-level summary:

- `master_comparison.md` — best config per model, per band, both inference modes.
- `metrics_best_per_model.md` — accuracy / precision / recall / F1 (macro).
- `confusion_matrices/` — text confusion matrices (no images needed, so they
  render fine in a terminal or over SSH).
- `downscaling_comparison.md` — which downscaling method won.
- `efficiency_profile.md` — parameters, FLOPs, latency, memory per model.

---

## 6. Vision-Transformer pipelines

Alongside the CNN/RNN hybrids, several ViT backbones are provided, each with the
**fine-tuning recipe from its own paper** (so the comparison is faithful, not
one-size-fits-all):

| Model key | Backbone / weights | Paper recipe |
|-----------|--------------------|--------------|
| `vit_b16_in21k` | ViT-B/16, ImageNet-21K (MIIL) | Adam + 1-cycle LR (max 3e-4), weight decay 1e-4, Cutout — *ImageNet-21K Pretraining for the Masses*, arXiv:2104.10972 |
| `vit_s16_dino` | ViT-S/16, DINO self-supervised | SGD momentum + cosine, no weight decay — *When ViTs Outperform ResNets…*, arXiv:2106.01548 |
| `vit_s8_dino` | ViT-S/8, DINO (patch 8 → ~4× compute) | same SAM-paper recipe |
| `vit_s16plus_dinov3` | ViT-S+/16, DINOv3 (LVD-1689M) | same SAM-paper recipe |

Pretrained weights load through [`timm`](https://github.com/huggingface/pytorch-image-models)
(with a HuggingFace fallback). **SAM** (sharpness-aware minimization, the core
idea of arXiv:2106.01548) is available as an option (`--sam`) but off by
default, since that paper does not use SAM during fine-tuning.

By default these ViT scripts **train until they start to overfit** (early
stopping on the validation loss) rather than for a fixed number of epochs; pass
`--paper-schedule` to reproduce each paper's exact fixed-length schedule
instead. This changes only the *training schedule*, never the model
architecture or its pretraining.

---

## 7. Running on Narval

### One-time setup (on a **login node** — it has internet)

```bash
mkdir -p results/logs
module load python cuda
source ~/projects/def-shervinv/<user>/spectrum_sensing_env/bin/activate
pip install -r requirements.txt          # torch, timm, transformers, scikit-learn, h5py, ...

# Pre-download the pretrained weights into project storage (compute nodes are OFFLINE):
export TORCH_HOME=~/projects/def-shervinv/<user>/torch_weights
export HF_HOME=$TORCH_HOME/hf
python Narval_radar_motion_vit_b16_s16_s8_s16plus.py --download-weights --torch-home $TORCH_HOME
```

You should see an `[OK]` line per model. (DINO/DINOv3 weights via `timm` are
**not** gated — no HuggingFace token needed.)

### Submit a job

```bash
# All four ViT models, all four bands, in one job:
sbatch Narval_radar_motion_vit_b16_s16_s8_s16plus.sh

# ...or one model at a time (recommended — see the note below):
sbatch run_vit_b16_in21k.sh
sbatch run_vit_s16_dino.sh
sbatch run_vit_s8_dino.sh
sbatch run_vit_s16plus_dinov3.sh
```

### Build the final tables (after training finishes)

```bash
python Narval_radar_motion_vit_b16_s16_s8_s16plus.py --aggregate --torch-home $TORCH_HOME
```

> **Why the weight cache matters.** Narval **compute nodes have no internet**.
> If the one-time `--download-weights` step is skipped, the models silently fall
> back to *random* initialization and produce near-chance (~10–20%) accuracy.
> The scripts now **abort with a clear message** if the cache is empty (override
> with `--allow-random-init` only for a deliberate from-scratch run).

> **Walltime.** Four models × four bands × 24 configs = 384 trainings, each with
> the extra MC-dropout inference, and ViT-S/8 is ~4× heavier than the others. A
> single combined job can exceed the 7-day limit — running one model per job
> (the `run_*.sh` scripts) is safer.

---

## 8. Common command-line options

| Flag | Effect |
|------|--------|
| `--data-root PATH` | Where the `.mat` data lives. |
| `--out-dir PATH` | Where results are written. |
| `--models M [M ...]` | Train only specific model(s). |
| `--bands B [B ...]` | Train only specific band(s). |
| `--task-id N` | Run a single `(model, band)` pair — used for SLURM job arrays. |
| `--downscale-mode {mixed,multichannel,autoencoder}` | Use one downscaling method instead of comparing all three. |
| `--paper-schedule` | Use each paper's fixed-length schedule instead of early stopping. |
| `--sam` | Enable sharpness-aware minimization (~2× compute). |
| `--quick` | Smaller ablation grid (drops the pretrain axis) for a fast check. |
| `--download-weights` | Cache pretrained weights (run on a login node). |
| `--aggregate` | Build the master comparison tables from existing results. |

---

## 9. Reproducibility

- Fixed random seed (`RANDOM_SEED = 42`) for data splits and initialization.
- Stratified train / val / test split (70 / 15 / 15) so every class is
  represented in each split.
- All hyper-parameters and per-sample predictions are saved to JSON, so any
  table or plot can be regenerated without re-training.

---

## License

See [`LICENSE`](LICENSE).

## Contact

Atik Mahabub — INRS-EMT — <atik.mahabub@inrs.ca>
