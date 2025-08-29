from __future__ import annotations
import numpy as np
import pandas as pd

# Make sure to import torch and nn
import torch
import torch.nn as nn

# Assume build_tabular_matrix and TORCH_OK are defined elsewhere

class AEModel(nn.Module):
    def __init__(self, d_in: int, latent: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(),
            nn.Linear(64, d_in)
        )

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, z

class AEArtifacts:
    def __init__(self, model: AEModel | None, d_in: int):
        self.model = model
        self.d_in = d_in

def train_autoencoder(tx: pd.DataFrame) -> AEArtifacts:
    X = build_tabular_matrix(tx)
    d_in = X.shape[1]

    if not TORCH_OK:
        return AEArtifacts(None, d_in)

    model = AEModel(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    model.train()
    for _ in range(5):  # few epochs for demo
        opt.zero_grad()
        xrec, _ = model(X_t)
        loss = loss_fn(xrec, X_t)
        loss.backward()
        opt.step()
    model.eval()
    return AEArtifacts(model, d_in)

def predict_autoencoder(art: AEArtifacts, tx_batch: pd.DataFrame) -> np.ndarray:
    X = build_tabular_matrix(tx_batch).astype(np.float32)
    if not TORCH_OK or art.model is None:
        # Fallback: scaled z-score anomaly proxy
        z = (X - X.mean(0)) / (X.std(0) + 1e-6)
        return np.clip(np.mean(np.abs(z), axis=1) / 10.0, 0, 1)

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        xrec, _ = art.model(X_t)
        rec_err = ((X_t - xrec) ** 2).mean(dim=1).cpu().numpy()
        # Normalize to 0..1
        rec_err = (rec_err - rec_err.min()) / (rec_err.ptp() + 1e-9)
        return rec_err