import torch.nn as nn
from src.config import AE_INPUT_DIM, AE_LATENT_DIM

class Autoencoder(nn.Module):
    def __init__(self, input_dim=AE_INPUT_DIM, latent_dim=AE_LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

