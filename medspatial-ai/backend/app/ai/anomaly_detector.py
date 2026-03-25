"""
MedSpatial AI — Anomaly Detector
Detects anomalies in 3D medical volumes using reconstruction-based detection
combined with learned density estimation. Generates 3D anomaly heatmaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class AnomalyEncoder(nn.Module):
    """3D convolutional encoder that compresses volumes to a latent space."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(4),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        flat = features.flatten(1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar


class AnomalyDecoder(nn.Module):
    """3D convolutional decoder that reconstructs volumes from latent space."""

    def __init__(self, latent_dim: int = 256, output_size: int = 128):
        super().__init__()
        self.output_size = output_size
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.ConvTranspose3d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.reshape(-1, 256, 4, 4, 4)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.output_size, mode="trilinear", align_corners=False)
        return x


class DensityEstimator(nn.Module):
    """
    Learned density estimator using a Gaussian Mixture in the latent space.
    Estimates how likely a given latent vector is under the learned distribution.
    """

    def __init__(self, latent_dim: int = 256, num_components: int = 10):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim

        self.means = nn.Parameter(torch.randn(num_components, latent_dim) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(num_components, latent_dim))
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log-likelihood under the mixture model."""
        B = z.shape[0]
        weights = F.softmax(self.weights, dim=0)

        log_probs = []
        for k in range(self.num_components):
            mean = self.means[k]
            log_var = self.log_vars[k]
            var = torch.exp(log_var) + 1e-6

            diff = z - mean.unsqueeze(0)
            log_p = -0.5 * (
                self.latent_dim * np.log(2 * np.pi)
                + log_var.sum()
                + (diff ** 2 / var).sum(dim=1)
            )
            log_probs.append(log_p + torch.log(weights[k] + 1e-10))

        log_probs = torch.stack(log_probs, dim=1)
        log_likelihood = torch.logsumexp(log_probs, dim=1)
        return log_likelihood


class AnomalyDetector3D(nn.Module):
    """
    3D Anomaly Detection Model combining:
    1. Variational Autoencoder for reconstruction-based anomaly detection
    2. Learned density estimation in latent space
    3. Spatial anomaly localization via reconstruction error maps

    The model learns the distribution of "normal" anatomy. Anomalies produce:
    - High reconstruction error (the decoder can't properly reconstruct unusual patterns)
    - Low density scores (the latent vector falls outside the learned normal distribution)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        input_size: int = 128,
        num_gmm_components: int = 10,
        transformer_dim: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.encoder = AnomalyEncoder(in_channels=1, latent_dim=latent_dim)
        self.decoder = AnomalyDecoder(latent_dim=latent_dim, output_size=input_size)
        self.density = DensityEstimator(latent_dim=latent_dim, num_components=num_gmm_components)

        # Optional transformer feature fusion
        if transformer_dim > 0:
            self.transformer_fusion = nn.Sequential(
                nn.Linear(transformer_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        else:
            self.transformer_fusion = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        transformer_features: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, D, H, W) input volume
            transformer_features: optional (B, D) global features from SpatialTransformer3D

        Returns:
            dict with:
                reconstruction: (B, 1, D, H, W) reconstructed volume
                anomaly_map: (B, 1, D, H, W) per-voxel anomaly scores
                anomaly_score: (B,) overall anomaly score
                mu, logvar: latent distribution parameters
        """
        mu, logvar = self.encoder(x)

        # Fuse transformer features if available
        if self.transformer_fusion is not None and transformer_features is not None:
            tf = self.transformer_fusion(transformer_features)
            mu = mu + tf
            logvar = logvar + tf * 0.1  # mild influence on variance

        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        # Reconstruction error map (per-voxel anomaly scores)
        recon_error = (x - reconstruction) ** 2
        anomaly_map = recon_error

        # Density-based anomaly score
        density_score = -self.density(mu)  # negative log-likelihood = anomaly

        # Overall anomaly score combines reconstruction error and density
        recon_score = recon_error.flatten(1).mean(dim=1)
        anomaly_score = 0.7 * recon_score + 0.3 * torch.sigmoid(density_score)

        return {
            "reconstruction": reconstruction,
            "anomaly_map": anomaly_map,
            "anomaly_score": anomaly_score,
            "density_score": density_score,
            "mu": mu,
            "logvar": logvar,
        }

    def compute_loss(
        self, outputs: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute VAE + density estimation loss."""
        recon_loss = F.mse_loss(outputs["reconstruction"], targets)

        # KL divergence
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + 0.001 * kl_loss

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl_divergence": kl_loss,
        }


def create_anomaly_detector(
    latent_dim: int = 256,
    input_size: int = 128,
    transformer_dim: int = 512,
    pretrained_path: str = None,
) -> AnomalyDetector3D:
    """Factory function for AnomalyDetector3D."""
    model = AnomalyDetector3D(
        latent_dim=latent_dim,
        input_size=input_size,
        transformer_dim=transformer_dim,
    )
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model
