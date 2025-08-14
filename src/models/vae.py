"""VAE model definition (simple convolutional VAE)."""
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=224, latent_dim=128):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (input_dim // 4) * (input_dim // 4), 512),
            nn.ReLU()
        )
        self.mu = nn.Linear(512, latent_dim)
        self.log_var = nn.Linear(512, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * (input_dim // 4) * (input_dim // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (64, input_dim // 4, input_dim // 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        reconstructed = self.decoder(z)
        return reconstructed, mu, log_var
