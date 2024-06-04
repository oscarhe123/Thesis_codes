import numpy as np


def normalize_images(images):     # Normalizing images for 
    # images should be a B x N x M tensor
    min_vals = images.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
    max_vals = images.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]

# Normalize to [0, 1]
    normalized_images = (images - min_vals) / (max_vals - min_vals)

# Scale to [0, 255]
    #normalized_images = normalized_images * 255
    
    #normalized_images = normalized_images.int()

# Optional: convert to uint8
    #normalized_images = normalized_images.byte()

    return normalized_images
