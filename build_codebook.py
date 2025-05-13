import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import numpy as np
import pickle

def extract_all_patches(patch_size=4):
    dataset = datasets.MNIST(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor()
    )

    all_patches = []
    for img, _ in dataset:
        img = img.squeeze(0)  # (28, 28)
        patches = img.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size * patch_size)  # (49, 16)
        all_patches.append(patches)

    all_patches = torch.cat(all_patches, dim=0)  # (N_patches_total, patch_dim)
    return all_patches

def build_kmeans_codebook(n_clusters=256, patch_size=4):
    print("Extracting patches from MNIST...")
    all_patches = extract_all_patches(patch_size)
    print(f"Total patches: {all_patches.shape[0]}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    print("Fitting KMeans...")
    kmeans.fit(all_patches.numpy())

    with open("codebook.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    print("Codebook saved to codebook.pkl")

if __name__ == '__main__':
    build_kmeans_codebook()