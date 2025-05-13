# utils.py
import numpy as np
from config import IMAGE_SIZE, PATCH_SIZE

def reconstruct_image(patches, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    """
    Reconstructs an image from a flat list of patches.
    Assumes the patches are ordered row-wise.

    Args:
        patches: (num_patches, patch_dim) as a NumPy array
        image_size: full image width/height (e.g., 28)
        patch_size: patch width/height (e.g., 4)

    Returns:
        Reconstructed image as a NumPy 2D array.
    """
    patches = patches.reshape(-1, patch_size, patch_size)
    num_patches = patches.shape[0]
    grid_size = int(np.sqrt(num_patches))

    if grid_size * patch_size != image_size:
        raise ValueError("Patch layout doesn't match image size.")

    rows = []
    for i in range(grid_size):
        row = np.hstack(patches[i * grid_size:(i + 1) * grid_size])
        rows.append(row)

    return np.vstack(rows)