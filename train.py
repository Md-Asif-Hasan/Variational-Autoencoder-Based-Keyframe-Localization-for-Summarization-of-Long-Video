"""Training loop for the VAE model."""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LR
from vae_model import VAE, vae_loss_function
from utils import set_seed

class DummyImageDataset(Dataset):
    """Placeholder dataset. Replace with real frame dataset."""
    def __init__(self, n=1000, image_size=(3,224,224)):
        import numpy as np
        self.data = np.random.rand(n, *image_size).astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.data[idx])

def train():
    set_seed()
    model = VAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    ds = DummyImageDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch in dl:
            batch = batch.to(DEVICE)
            recon, mu, logvar = model(batch)
            loss, recon_l, kld = vae_loss_function(recon, batch, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss: {total_loss/len(dl):.4f}")

if __name__ == '__main__':
    train()
