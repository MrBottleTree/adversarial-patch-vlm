# Project Context: Information Retrieval - Adversarial Misclassification

## Project Objective
To build an evaluation pipeline that tests the robustness of various pre-trained image classification models. The core goal is to quantify vulnerabilities to adversarial inputs (patches/noise) across diverse architectures.

## Current Status: Phase 2 - Evaluation & High-Resolution Transition
Moved from data collection to rigorous evaluation and resolution upgrading.

### Completed Progress:
* **Surgical Patching Pipeline:** Implemented `model_factory.py` and `patch_utility.py` for exact 8x8 patching (no pixel bleed).
* **Relative Subset Probability:** Created `eval_utility.py` to calculate probability relative to 10 target classes, resolving the 1000-class dilution problem.
* **Architecture Suite:** 6 models: ResNet-18, VGG-16, ViT, Swin Transformer, ConvNeXt, CLIP.
* **Insight:** Confirmed that ViTs/Transformers are more robust to low-res/blurry inputs than CNNs due to global attention vs. local kernels.

## Workflow Strategy
We are transitioning from CIFAR-10 (32x32) to **ImageNet-1k Validation (224x224)** to eliminate resolution bottlenecks and ensure high clean-image accuracy (>90%) for baseline measurements.

### 1. Automated Dataset Sourcing
* **Kaggle CLI:** Used for automated download of `ILSVRC2012_img_val.tar`.
* **Environment:** Configured with `KAGGLE_USERNAME` and `KAGGLE_KEY`.

### 2. Metric Specification
* **Relative Probability:** Calculated as P(True Class) / Sum(P(All 10 target classes)) to normalize across models.

## Immediate Action Items
1. **Infrastructure:** Finish downloading and extracting the ImageNet-1k Validation set.
2. **Refactor:** Update `eval_utility.py` and `patch_utility.py` for 224x224 native image processing.
3. **Execution:** Benchmarking baseline and patched probabilities across the 6-model suite.
