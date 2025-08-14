"""Training utilities for the VAE."""
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def train_vae(vae, data_loader, device, epochs=10, lr=1e-3, weight_kl=0.001, save_path=None):
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.to(device)
    vae.train()
    epoch_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs}", unit='batch') as pbar:
            for frames in data_loader:
                frames = frames.to(device)
                reconstructed, mu, log_var = vae(frames)
                recon_loss = nn.MSELoss()(reconstructed, frames)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + weight_kl * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        avg = epoch_loss / max(1, len(data_loader))
        epoch_losses.append(avg)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")
        if save_path:
            torch.save(vae.state_dict(), save_path)
    return epoch_losses

def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(model, filename, device='cpu'):
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    print(f"Model loaded from {filename}")
    return model
