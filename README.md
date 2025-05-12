# Pixel Prediction Transformer

This project implements an autoregressive transformer model for generating simple sketches and drawings. The model learns to predict and generate images one patch at a time, using a transformer architecture.

## Project Structure

- `model.py`: Contains the transformer model architecture
- `data.py`: Handles data loading and preprocessing
- `train.py`: Main training script
- `utils.py`: Utility functions for visualization and processing

## Features

- Image tokenization using patch-based approach
- Autoregressive transformer architecture
- Training on simple shapes and sketches
- Generation of new drawings

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

## How it Works

The model works by:
1. Breaking down images into patches
2. Converting patches into tokens
3. Using a transformer to predict the next patch in the sequence
4. Generating new images autoregressively

## Requirements

See `requirements.txt` for full list of dependencies. 