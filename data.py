import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import pickle
from config import PATCH_SIZE, IMAGE_SIZE

class PatchSequenceDataset(Dataset):
    def __init__(self, train=True, patch_size=PATCH_SIZE):
        self.data = datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )
        self.patch_size = patch_size
        # Load KMeans codebook
        with open("codebook.pkl", "rb") as f:
            self.kmeans = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        img = img.squeeze(0)  # (28, 28)

        patches = img.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, self.patch_size * self.patch_size)  # (49, 16)

        with torch.no_grad():
            # Assign each patch to its nearest centroid (i.e., token id)
            patch_tokens = torch.tensor(self.kmeans.predict(patches.numpy()), dtype=torch.long)  # (49,)

        return patch_tokens[:-1], patch_tokens[1:]  # input sequence, target sequence