import torch
from torch.utils.data import DataLoader
from model import PatchTransformer
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from data import PatchSequenceDataset
from utils import reconstruct_image
import joblib
from tqdm import tqdm
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WARMUP_EPOCHS,
    NUM_PATCHES, DEVICE
)

kmeans = joblib.load("codebook.pkl")

def generate_sample(model, device, seq_len=NUM_PATCHES, epoch=None):
    model.eval()
    generated = torch.zeros(1, 1, dtype=torch.long).to(device)

    for _ in range(seq_len-1):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        generated = torch.cat([generated, next_token], dim=1)

    tokens = generated[0].cpu().numpy()  # shape (seq_len,)
    patches = kmeans.cluster_centers_[tokens]  # shape (seq_len, patch_dim)
    image = reconstruct_image(patches)  # should return a 2D or 3D array

    # Create directory for generated images if it doesn't exist
    os.makedirs('generated_images', exist_ok=True)
    
    # Save the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"Generated Image - Epoch {epoch+1}")
    plt.axis("off")
    plt.savefig(f'generated_images/epoch_{epoch+1}.png')
    plt.close()

def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
    return LEARNING_RATE

def train():
    device = torch.device(DEVICE)
    dataset = PatchSequenceDataset(train=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PatchTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Update learning rate with warmup
        lr = get_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        for inp, target in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            inp, target = inp.to(device), target.to(device)
            optimizer.zero_grad()

            out = model(inp)
            loss = criterion(out.view(-1, out.size(-1)), target.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader) 
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
        
        # Generate and save images every 5 epochs
        if (epoch + 1) % 5 == 0:
            generate_sample(model, device, seq_len=NUM_PATCHES, epoch=epoch)

if __name__ == '__main__':
    train()
