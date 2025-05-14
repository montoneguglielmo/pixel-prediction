# Pixel Prediction Transformer

This project implements an autoregressive transformer model for generating MNIST-like handwritten digits. The model learns to predict and generate images one patch at a time, using a transformer architecture. The model is trained on the MNIST dataset and learns to generate new handwritten digits that follow similar patterns.

## Project Structure

- `model.py`: Contains the transformer model architecture
- `data.py`: Handles data loading and preprocessing
- `train.py`: Main training script
- `utils.py`: Utility functions for visualization and processing
- `config.py`: Configuration parameters for the model and training
- `build_codebook.py`: Script for building the image token codebook
- `generated_images/`: Directory containing generated images
- `data/`: Directory containing training data
- Model checkpoints:
  - `best_pixel_transformer.pth`: Best performing model checkpoint
  - `final_pixel_transformer.pth`: Final model checkpoint
  - `pixel_transformer.pth`: Latest model checkpoint
- `codebook.pkl`: Pre-computed codebook for image tokenization

## Features

- Image tokenization using patch-based approach with K-means clustering for codebook generation
- Autoregressive transformer architecture
- Training on MNIST handwritten digits dataset
- Generation of new MNIST-like handwritten digits
- Configurable model parameters
- Progress tracking and visualization

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build the codebook (if not using pre-computed one):
```bash
python build_codebook.py
```

4. Run training:
```bash
python train.py
```

## How it Works

The model works by:
1. Breaking down MNIST images into patches
2. Converting patches into tokens using a K-means clustering based codebook
3. Using a transformer to predict the next patch in the sequence
4. Generating new MNIST-like digits autoregressively

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- torchvision 0.15.0+
- numpy 1.21.0+
- matplotlib 3.4.0+
- Pillow 8.0.0+
- tqdm 4.65.0+
- einops 0.6.0+

See `requirements.txt` for full list of dependencies. 