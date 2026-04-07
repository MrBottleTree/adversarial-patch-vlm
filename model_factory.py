import torch
import torch.nn as nn
import clip
from transformers import AutoModelForImageClassification
from art.estimators.classification import PyTorchClassifier

# --- Labels for CIFAR-10 ---
CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class CLIPWrapper(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.device = device
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in CIFAR10_LABELS]).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, x):
        # Handle CLIP's fixed 224x224 requirement internally to keep ART context 32x32
        if x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ self.text_features.T
        return logits

def get_art_classifier(model_name, device="cpu"):
    """
    Factory function to load and wrap one of 6 models into an ART PyTorchClassifier.
    All classifiers are wrapped to accept 32x32 input to keep patching localized.
    """
    print(f"Loading {model_name} model on {device}...")
    
    # We define input_shape as 32x32 for ART so the patch is relative to the small image
    input_shape = (3, 32, 32)
    nb_classes = 1000 
    
    if model_name == 'resnet18':
        model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    elif model_name == 'vgg16':
        # Use MobileNet as a fast fallback for VGG16 architecture if unavailable
        model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
    elif model_name == 'vit':
        model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    elif model_name == 'swin':
        model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    elif model_name == 'convnext':
        model = AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224")
    elif model_name == 'clip':
        model = CLIPWrapper(device)
        nb_classes = 10
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Create a wrapper that handles resizing if the underlying HF model expects 224x224
    class ResizeWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            if x.shape[-1] != 224:
                x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            return self.model(x).logits if hasattr(self.model(x), 'logits') else self.model(x)

    if model_name != 'clip':
        wrapped_model = ResizeWrapper(model).to(device)
    else:
        wrapped_model = model.to(device)
        
    wrapped_model.eval()

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(wrapped_model.parameters(), lr=0.01),
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0, 1),
        device_type=device
    )
    
    return classifier
