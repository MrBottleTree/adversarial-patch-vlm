import os
import numpy as np
from PIL import Image
from art.attacks.evasion import AdversarialPatch

# 224x224 image divided into a 4x4 grid → each cell is 56x56 = 1/16th of the image area
CELL_SIZE = 56
GRID_COLS = 4
PATCH_SCALE = 0.25  # linear scale → 0.25^2 = 1/16 of image area


def generate_adversarial_patch(classifier, x_batch, y_batch, max_iter=500, learning_rate=5.0):
    """
    Optimizes a 56x56 adversarial patch against a specific classifier.

    The patch is trained over a batch of images with random placement each
    iteration, making it location-independent at inference time.

    Args:
        classifier: ART PyTorchClassifier (expects 224x224 input)
        x_batch:    np.ndarray (N, 3, 224, 224), float32 in [0, 1]
        y_batch:    np.ndarray (N,), int — true ImageNet label indices
        max_iter:   number of optimization steps
        learning_rate: patch update step size

    Returns:
        patch:  np.ndarray (3, 56, 56) — the optimized patch
        attack: AdversarialPatch instance (needed for apply_patch)
    """
    attack = AdversarialPatch(
        classifier=classifier,
        rotation_max=0.0,
        scale_min=PATCH_SCALE - 0.01,
        scale_max=PATCH_SCALE,
        patch_shape=(3, CELL_SIZE, CELL_SIZE),
        max_iter=max_iter,
        learning_rate=learning_rate,
        targeted=False,
        verbose=True,
    )
    patch, _ = attack.generate(x=x_batch, y=y_batch)
    return patch, attack


def apply_adversarial_patch(patch, attack, x, patch_location=None):
    """
    Stamps an adversarial patch onto image(s).

    Args:
        patch:          np.ndarray (3, 56, 56) from generate_adversarial_patch
        attack:         AdversarialPatch instance from generate_adversarial_patch
        x:              np.ndarray (N, 3, 224, 224) — images to patch
        patch_location: int 0-15 (cell in 4x4 grid), or None for random placement via ART

    Returns:
        np.ndarray (N, 3, 224, 224) — patched images
    """
    if patch_location is None:
        return attack.apply_patch(x=x, scale=PATCH_SCALE, patch_external=patch)

    if not (0 <= patch_location <= 15):
        raise ValueError("patch_location must be between 0 and 15.")

    row = patch_location // GRID_COLS
    col = patch_location % GRID_COLS
    y0 = row * CELL_SIZE
    x0 = col * CELL_SIZE

    # Direct pixel stamping — avoids ART's mask boundary validation entirely
    x_patched = x.copy()
    x_patched[:, :, y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE] = np.clip(patch, 0.0, 1.0)
    return x_patched


def save_patched_image(x_patched, output_dir, filename):
    """Saves the first image in x_patched (N, 3, 224, 224) to disk."""
    arr = (np.transpose(x_patched[0], (1, 2, 0)) * 255).round().astype(np.uint8)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    Image.fromarray(arr).save(path)
    return path
