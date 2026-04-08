import torch
import torch.nn as nn
import clip
import torchvision.models as tv_models
from transformers import AutoModelForImageClassification
from art.estimators.classification import PyTorchClassifier


def _get_imagenet_class_names():
    return tv_models.VGG16_Weights.IMAGENET1K_V1.meta['categories']


class CLIPWrapper(nn.Module):
    """Zero-shot CLIP classifier over all 1000 ImageNet classes."""
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.device = device
        class_names = _get_imagenet_class_names()
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, x):
        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return 100.0 * image_features @ self.text_features.T


class HFWrapper(nn.Module):
    """Unwraps HuggingFace model output to raw logits."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.logits if hasattr(out, 'logits') else out


def get_art_classifier(model_name, device="cpu"):
    """
    Returns an ART PyTorchClassifier for one of 6 ImageNet models.
    All models expect (N, 3, 224, 224) float32 input in [0, 1].
    """
    print(f"Loading {model_name} on {device}...")

    if model_name == 'resnet18':
        model = HFWrapper(AutoModelForImageClassification.from_pretrained("microsoft/resnet-18"))
    elif model_name == 'vgg16':
        model = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'vit':
        model = HFWrapper(AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224"))
    elif model_name == 'swin':
        model = HFWrapper(AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224"))
    elif model_name == 'convnext':
        model = HFWrapper(AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224"))
    elif model_name == 'clip':
        model = CLIPWrapper(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.eval()

    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0.0, 1.0),
        device_type=device,
    )
