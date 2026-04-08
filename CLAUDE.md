# Project Context: Adversarial Patch — VLM Robustness Evaluation

## Project Objective
Build an evaluation pipeline that tests the robustness of pre-trained image classification models against adversarial patches. Core goal: quantify vulnerabilities across diverse architectures using a mathematically optimized, location-independent patch.

## Current Status: Phase 3 — Adversarial Patch Generation & Evaluation

### Completed:
* **Dataset:** ImageNet-1k validation set downloaded as 14 parquet shards (`data/imagenet_val/data/validation-*.parquet`, 6.3GB, ~50k images, columns: `image` (bytes), `label` (int 0–999)).
* **Refactor:** All CIFAR-10 code removed. Pipeline is fully ImageNet/224x224 native.
* **model_factory.py:** 6 models — ResNet-18, VGG-16 (torchvision, real VGG not MobileNet), ViT, Swin, ConvNeXt, CLIP (zero-shot over 1000 ImageNet classes via torchvision class names). All expect `(N, 3, 224, 224)` float32 input in `[0, 1]`.
* **patch_utility.py:** `generate_adversarial_patch()` calls ART's `.generate()` for gradient-optimized patches. `apply_adversarial_patch()` uses direct pixel stamping (NOT ART's mask system — mask boundary validation causes `ValueError` when center pixels fall on edge). Patch is 56×56 (1/16th area of 224×224), placed on a 4×4 grid (16 locations).
* **eval_utility.py:** `load_imagenet_batch()` reads from parquet. `evaluate_patch()` returns clean/patched accuracy and mean P(true class). `evaluate_all_models()` runs all 6 models.
* **playground.ipynb:** End-to-end notebook: load → generate patch → location test → transferability across 6 models → save.

### Key Design Decisions:
* **ART mask system avoided:** `apply_patch()` with a single-pixel mask raises `ValueError: a must be greater than 0` because ART zeros out boundary pixels. Direct numpy stamping is used instead.
* **Patch scale:** `scale_min=0.24, scale_max=0.25` (ART requires `scale_min < scale_max`).
* **Batch size:** Keep ≤ 8 images for GTX 1050 (2–4GB VRAM).
* **Device:** Auto-selects CUDA if available (`"cuda" if torch.cuda.is_available() else "cpu"`).
* **Metric:** Direct P(true class) from softmax — no CIFAR-to-ImageNet remapping needed.

## Hardware
* GPU: NVIDIA GTX 1050 (2–4GB VRAM)
* CPU training: ~25 min / 500 iter. GPU training: ~2–3 min / 500 iter.

## Immediate Next Steps
1. Run full patch generation on GPU and verify location-independence across all 16 locations.
2. Run transferability evaluation across all 6 models.
3. Analyze which architectures (CNNs vs Transformers) are more vulnerable.
