import os
import numpy as np
import torch
from PIL import Image
from art.attacks.evasion import AdversarialPatch
from IPython.display import display

def patch_image_modular(input_path, output_dir, patch_location, classifier, model_name):
    """
    Patches a 32x32 image using ART's AdversarialPatch for a specific classifier.
    Guarantees that ONLY the 8x8 region is modified.
    """
    if not (0 <= patch_location <= 15):
        raise ValueError("patch_location must be between 0 and 15 inclusive.")

    # 1. Load original image as-is
    # We do NOT resize or convert the clean base unless necessary to avoid global noise
    with Image.open(input_path) as img:
        img_rgb = img.convert("RGB")
        clean_array = np.array(img_rgb)
    
    # 2. Define the grid logic (4x4 grid on 32x32 image)
    # Cell size is exactly 8x8 pixels
    row, col = patch_location // 4, patch_location % 4
    y_start, x_start = row * 8, col * 8
    y_end, x_end = y_start + 8, x_start + 8
    
    # 3. Prepare for ART (N, C, H, W) normalized to [0, 1]
    # We only use this for the ART API call
    x = clean_array.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1)) 
    x = np.expand_dims(x, axis=0) # (1, 3, 32, 32)
    
    # 4. Create the mask for ART
    # Center of an 8x8 patch starting at (y_start, x_start) is +4,+4
    mask = np.zeros((1, 32, 32), dtype=bool)
    center_y = y_start + 4
    center_x = x_start + 4
    mask[0, center_y, center_x] = True

    # 5. Initialize ART Attack
    # We create a random patch for the generation
    # Use a fixed seed for reproducibility in testing if needed
    patch_external = np.random.rand(3, 8, 8).astype(np.float32)
    
    attack = AdversarialPatch(
        classifier=classifier,
        rotation_max=0.0,
        scale_min=0.24,
        scale_max=0.25,
        patch_shape=(3, 8, 8)
    )
    
    # 6. Apply patch using ART API
    x_patched = attack.apply_patch(x=x, scale=0.25, patch_external=patch_external, mask=mask)
    
    # 7. SURGICAL APPLICATION:
    # Convert the ART result back to uint8
    patched_full_array = (np.transpose(x_patched[0], (1, 2, 0)) * 255).round().astype(np.uint8)
    
    # Start with a COMPLETELY FRESH copy of the original data
    final_array = clean_array.copy()
    
    # ONLY overwrite the targeted 8x8 box
    # This is the "surgical" step that guarantees no other pixel changes
    final_array[y_start:y_end, x_start:x_end] = patched_full_array[y_start:y_end, x_start:x_end]
    
    # 8. Save and Display
    final_img = Image.fromarray(final_array)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"patched_{model_name}_loc{patch_location}_{base_name}")
    final_img.save(output_path)
    
    print(f"Surgically patched location {patch_location} ([{y_start}:{y_end}, {x_start}:{x_end}]) for {model_name}")
    display(final_img)
    
    return output_path

def patch_image_all_models(input_path, output_dir, patch_location, device="cpu"):
    """
    Runs the patching process for all 6 supported models and saves the results.
    """
    from model_factory import get_art_classifier
    
    models = ['resnet18', 'vgg16', 'vit', 'swin', 'convnext', 'clip']
    results = {}

    print(f"Starting all-model patching for location {patch_location} on device: {device}...")
    
    for model_name in models:
        try:
            classifier = get_art_classifier(model_name, device=device)
            model_output_dir = os.path.join(output_dir, model_name)
            
            out_path = patch_image_modular(
                input_path=input_path,
                output_dir=model_output_dir,
                patch_location=patch_location,
                classifier=classifier,
                model_name=model_name
            )
            results[model_name] = out_path
            
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            
    print("\nAll-model patching complete!")
    return results
