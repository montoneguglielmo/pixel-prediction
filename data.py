import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class MNISTPatches(Dataset):
    def __init__(self, train=True, patch_size=4):
        self.data = datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        img = img.squeeze(0)  # (28, 28)
        patches = self._extract_patches(img)
        return patches

    def _extract_patches(self, img):
        ps = self.patch_size
        patches = img.unfold(0, ps, ps).unfold(1, ps, ps)  # (7, 7, ps, ps)
        patches = patches.contiguous().view(-1, ps * ps)   # (49, 16)
        return patches