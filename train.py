import torch
from torch.utils.data import DataLoader
from data import MNISTPatches
from model import PatchTransformer
import torch.nn.functional as F


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MNISTPatches(train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PatchTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for patches in loader:
            patches = patches.to(device)  # (B, 49, 16)
            inp = patches[:, :-1, :]
            target = patches[:, 1:, :]

            out = model(inp)
            #print('Out shape:', out.shape)
            #print('Target shape:', target.shape)
            loss = F.mse_loss(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")


if __name__ == '__main__':
    train()
