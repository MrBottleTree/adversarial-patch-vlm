# IR VLM Project — Context

## What this project is

Research on the adversarial robustness of Vision-Language Models (VLMs), using
Food-101 as the evaluation benchmark. The original focus was typographic attacks
(overlaying text to trick a VLM into reading a label instead of classifying the
image); the scope has since expanded to four additional attack categories.

The experimental pattern:
1. Take images from Food-101 validation split.
2. Apply an adversarial perturbation (text overlay, pixel noise, occlusion, etc.).
3. Ask the VLM to classify the food. Measure clean accuracy, attack accuracy /
   fooled %, and accuracy drop.

## Files

| File | Purpose |
|------|---------|
| `typographic_attack_vlm.ipynb` | Original Colab notebook — 3 small VLMs, 3 typographic styles, T4 GPU. |
| `typographic_attack_vlm_large.ipynb` | Scaled-up notebook — 13 VLMs (≤ 28 GB fp16), 10 typographic styles, defense-prompt sweep. Outputs `{model}_results.xlsx`. |
| `adversarial_vlm_eval.ipynb` | **Primary notebook.** 7 large VLMs (up to 78B @ 4-bit), 32 attack variants across 5 categories, fine-grained JSON checkpointing, per-model disk purge. |
| `results/` | Excel workbooks and PNG plots from completed runs. |

## Primary notebook — `adversarial_vlm_eval.ipynb`

### Models (7)

| # | Name | HF ID | Quant | ~VRAM |
|---|------|--------|-------|-------|
| 1 | Qwen2-VL-7B | `Qwen/Qwen2-VL-7B-Instruct` | fp16 | 18 GB |
| 2 | Molmo-7B-D | `allenai/Molmo-7B-D-0924` | fp16 | 14 GB |
| 3 | LLaVA-NeXT-34B | `lmsys/llava-v1.6-34b` | 4-bit NF4 | 17 GB |
| 4 | InternVL2-26B | `OpenGVLab/InternVL2-26B` | 4-bit NF4 | 13 GB |
| 5 | InternVL2.5-38B | `OpenGVLab/InternVL2_5-38B` | 4-bit NF4 | 19 GB |
| 6 | Qwen2.5-VL-72B | `Qwen/Qwen2.5-VL-72B-Instruct` | 4-bit NF4 | 36 GB |
| 7 | InternVL2.5-78B | `OpenGVLab/InternVL2_5-78B` | 4-bit NF4 | 39 GB |

All non-gated. 4-bit config: NF4, double quant, compute dtype fp16. Model weights
are downloaded to a per-model subdirectory of `MODEL_CACHE_BASE` and deleted
immediately after results are saved (only one model on disk at a time, peak ~44 GB).

### Attack categories (32 variants total)

- **A. Typographic (10):** centered, tiled, banner, corner, caption, watermark,
  sticker, translucent, scattered, authority. Each also run with defense prompt
  if `INCLUDE_DEFENSE=True`. Metric: Fooled %.
- **B. Transfer adversarial (4):** FGSM ε=4/8/16, PGD-10 ε=8. ResNet-50
  surrogate (torchvision). Metric: accuracy drop.
- **C. Corruptions (9):** Gaussian noise σ×3, motion blur k×3, JPEG quality×3.
  Metric: accuracy drop.
- **D. Occlusion (4):** center block 25%/50%, random patches, grid mask.
  Metric: accuracy drop.
- **E. Perceptual (5):** negative, hue-90, hue-180, channel-swap, grayscale.
  Metric: accuracy drop.

### Checkpointing

Checkpoints are JSON files in `results/checkpoints/`, one per (model × attack
variant). Atomic writes via `os.replace`. On restart, each attack resumes from
the last saved image index. If `{model}_full_results.xlsx` exists the entire
model is skipped. If `{model}_results.xlsx` exists (from the large notebook) the
typographic section is loaded from it instead of rerunning.

### Cell layout (26 cells)

| Cells | Content |
|-------|---------|
| 0 | Markdown title + roster |
| 1–3 | Imports + directory setup; pip install (commented); user config |
| 4 | Dataset + subset + attack mapping |
| 5 | 10 typographic overlay functions + `TYPO_ATTACK_STYLES` |
| 6 | Transfer adversarial: surrogate load/unload, FGSM, PGD, `FOOD101_TO_IMAGENET` |
| 7 | Corruption, occlusion, perceptual functions + dicts |
| 8 | `ATTACK_REGISTRY` (unified, QUICK_TEST trimming) |
| 9 | Checkpoint utils (atomic JSON read/write) |
| 10 | `load_typo_results_from_xlsx()` — parses legacy Summary sheet |
| 11 | `safe_classify`, `unload_model`, `_vqa_prompt`, `match` |
| 12 | `run_clean_pass_checkpointed()` |
| 13 | `run_attack_checkpointed()` |
| 14 | `compute_typo_stats()`, `compute_acc_drop_stats()` |
| 15 | Excel style constants |
| 16 | fp16 loaders: Qwen2-VL-7B, Molmo-7B |
| 17 | `_internvl2_preprocess()` |
| 18 | 4-bit loaders: LLaVA-NeXT-34B, InternVL2-26B, InternVL2.5-38B, Qwen2.5-VL-72B, InternVL2.5-78B |
| 19 | `MODEL_REGISTRY` + filters |
| 20 | `_save_typo_xlsx()`, `save_model_full_excel()` |
| 21 | **MAIN LOOP** |
| 22 | `save_master_full_excel()` |
| 23 | Visualizations (heatmaps) |
| 24 | Console summary + master xlsx save |
| 25 | Output file listing |

## Legacy notebook — `typographic_attack_vlm_large.ipynb`

13 models (Kosmos-2 through LLaVA-NeXT Vicuna-13B), 10 typographic styles,
defense-prompt sweep. All fp16, target 48 GB GPU. Outputs
`{model}_results.xlsx` per model and `master_comparison.xlsx`. Still useful as a
lighter run for typographic-only results; `adversarial_vlm_eval.ipynb` can load
its output and skip the typographic section.

## Environment

- **GPU:** vast.ai 48 GB A6000 or L40 (PyTorch image).
- **Results dir resolution:** `/workspace/results` → `/root/results` → `~/results` → `./results`.
- **Model cache resolution:** same prefix order under `model_cache/`.

## Design conventions (keep when extending)

- **Imports inside loaders** — model-specific classes imported inside `load_*`
  so a missing class only breaks that one model.
- **`safe_classify` wrapper** — OOM and exceptions become `[OOM_ERROR]` /
  `[ERROR: TypeName]` in output instead of crashing the run.
- **Atomic checkpoint writes** — `.tmp` + `os.replace` prevents corrupt state.
- **Per-model disk purge** — `shutil.rmtree(model_cache, ignore_errors=True)`
  runs in a `finally` block after each model.
- **Top-level skip** — if `_full_results.xlsx` exists, skip the whole model
  including the load step.
- **No gated models** — all HF model IDs are publicly accessible.
- **Defense prompt only for typo** — the "ignore text overlays" prefix is only
  meaningful for typographic attacks; other categories use std prompt only.
