import glob
import io
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image


def load_imagenet_batch(n_images=32, shard=0, seed=42):
    """
    Loads a random sample of images from one validation parquet shard.

    Args:
        n_images: number of images to load
        shard:    which parquet shard to read (0–13)
        seed:     random seed for reproducible sampling

    Returns:
        x:      np.ndarray (N, 3, 224, 224), float32 in [0, 1]
        labels: np.ndarray (N,), int — ImageNet class indices (0–999)
    """
    parquet_files = sorted(glob.glob("data/imagenet_val/data/validation-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No validation parquet files found in data/imagenet_val/data/")

    df = pd.read_parquet(parquet_files[shard])
    df = df.sample(n=min(n_images, len(df)), random_state=seed).reset_index(drop=True)

    images, labels = [], []
    for _, row in df.iterrows():
        img = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB').resize((224, 224))
        x = np.array(img, dtype=np.float32) / 255.0
        images.append(np.transpose(x, (2, 0, 1)))
        labels.append(int(row['label']))

    return np.stack(images), np.array(labels)


def evaluate_patch(classifier, x_clean, x_patched, true_labels):
    """
    Compares clean vs patched classification for a batch of images.

    Args:
        classifier: ART PyTorchClassifier
        x_clean:    np.ndarray (N, 3, 224, 224)
        x_patched:  np.ndarray (N, 3, 224, 224)
        true_labels: np.ndarray (N,) — ImageNet class indices

    Returns:
        dict with clean_acc, patched_acc, clean_prob_mean, patched_prob_mean
    """
    p_clean = F.softmax(torch.from_numpy(classifier.predict(x_clean)), dim=1).numpy()
    p_patched = F.softmax(torch.from_numpy(classifier.predict(x_patched)), dim=1).numpy()

    idx = np.arange(len(true_labels))
    clean_probs = p_clean[idx, true_labels]
    patched_probs = p_patched[idx, true_labels]

    return {
        'clean_acc':         float((np.argmax(p_clean, axis=1) == true_labels).mean()),
        'patched_acc':       float((np.argmax(p_patched, axis=1) == true_labels).mean()),
        'clean_prob_mean':   float(clean_probs.mean()),
        'patched_prob_mean': float(patched_probs.mean()),
    }


def evaluate_all_models(patch, attack, x_clean, true_labels, device="cpu"):
    """
    Runs evaluate_patch across all 6 models for a given patch.

    Args:
        patch:       np.ndarray (3, 56, 56) — optimized adversarial patch
        attack:      AdversarialPatch instance used to apply the patch
        x_clean:     np.ndarray (N, 3, 224, 224)
        true_labels: np.ndarray (N,)
        device:      "cpu" or "cuda"

    Returns:
        dict mapping model_name -> eval metrics dict
    """
    from model_factory import get_art_classifier
    from patch_utility import apply_adversarial_patch

    models = ['resnet18', 'vgg16', 'vit', 'swin', 'convnext', 'clip']
    results = {}

    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        try:
            clf = get_art_classifier(model_name, device=device)
            x_patched = apply_adversarial_patch(patch, attack, x_clean)
            results[model_name] = evaluate_patch(clf, x_clean, x_patched, true_labels)
        except Exception as e:
            print(f"  Error: {e}")
            results[model_name] = None

    return results
