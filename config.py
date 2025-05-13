import torch

# Image and patch configuration
IMAGE_SIZE = 28
PATCH_SIZE = 4
PATCH_DIM = PATCH_SIZE * PATCH_SIZE
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# Model configuration
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 5

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 