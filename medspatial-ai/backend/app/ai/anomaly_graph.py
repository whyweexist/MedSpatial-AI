"""
MedSpatial AI — AnomalyGraph: Graph-Augmented Anomaly Transformer (GAAT)
Novel architecture combining 3D PatchCore + GraphSAGE + classification head.

This implements the EXACT architecture from program.md Section 1 / Subsystem C:
1. Patch-level anomaly scoring (PatchCore-3D)
2. GraphSAGE for anatomical context propagation
3. Multi-label disease classification + per-voxel severity
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Disease Labels ───────────────────────────────────────────────
DISEASE_LABELS = [
    "Pneumonia", "Pneumothorax", "Pleural Effusion",
    "Lung Nodule/Mass", "Cardiomegaly", "Atelectasis",
    "Consolidation", "Emphysema", "Fibrosis", "Fracture",
    "Tuberculosis", "COVID-19 patterns", "Normal/No Finding",
]
NUM_DISEASES = len(DISEASE_LABELS)


# ── Patch Embedding MLP ──────────────────────────────────────────
class PatchEmbedMLP(nn.Module):
    """Maps a flattened 8³=512 patch to a 64-dim normality embedding."""

    def __init__(self, patch_dim: int = 512, hidden_dim: int = 256,
                 embed_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


# ── Memory Bank ──────────────────────────────────────────────────
class NormalMemoryBank(nn.Module):
    """
    Stored codebook of 1024 normal patch embeddings.
    Loaded from disk or initialized from atlas on first run.
    """

    def __init__(self, bank_size: int = 1024, embed_dim: int = 64) -> None:
        super().__init__()
        self.register_buffer(
            "bank",
            F.normalize(torch.randn(bank_size, embed_dim), dim=-1),
        )
        self.bank_size = bank_size
        self.embed_dim = embed_dim

    def load_from_npz(self, path: str) -> None:
        """Load centroid embeddings computed offline from atlas patches."""
        data = np.load(path)
        centroids = torch.from_numpy(data["centroids"]).float()
        # If centroid dim != embed_dim, project
        if centroids.shape[-1] != self.embed_dim:
            # Simple PCA-like projection
            U, S, Vh = torch.linalg.svd(centroids, full_matrices=False)
            centroids = (U[:, :self.embed_dim] * S[:self.embed_dim].unsqueeze(0))
        self.bank = F.normalize(centroids[:self.bank_size], dim=-1)

    def anomaly_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (N, embed_dim)
        Returns: (N,) anomaly score in [0, 1]
        """
        # Cosine similarity between each embedding and all bank entries
        sim = torch.mm(embeddings, self.bank.T)  # (N, bank_size)
        max_sim, _ = sim.max(dim=-1)              # (N,)
        return 1.0 - max_sim.clamp(0, 1)


# ── GraphSAGE Layer ──────────────────────────────────────────────
class GraphSAGELayer(nn.Module):
    """GraphSAGE with mean aggregation (Hamilton et al.)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, N, F) node features
        adj: (N, N) adjacency matrix (sparse or dense)
        Returns: (B, N, out_features)
        """
        # Mean neighbor aggregation
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh_mean = torch.bmm(
            adj.unsqueeze(0).expand(x.size(0), -1, -1), x
        ) / deg.unsqueeze(0)
        out = F.gelu(self.norm(self.linear_self(x) + self.linear_neigh(neigh_mean)))
        return out


# ── Graph Builder ────────────────────────────────────────────────
def build_patch_adjacency(grid_d: int = 16, grid_h: int = 16,
                           grid_w: int = 16) -> torch.Tensor:
    """
    Build 6-connectivity adjacency matrix for a 3D patch grid.
    Returns (N, N) float tensor where N = grid_d * grid_h * grid_w.
    """
    N = grid_d * grid_h * grid_w
    adj = torch.zeros(N, N)

    def idx(d: int, h: int, w: int) -> int:
        return d * (grid_h * grid_w) + h * grid_w + w

    for d in range(grid_d):
        for h in range(grid_h):
            for w in range(grid_w):
                i = idx(d, h, w)
                for nd, nh, nw in [
                    (d-1, h, w), (d+1, h, w),
                    (d, h-1, w), (d, h+1, w),
                    (d, h, w-1), (d, h, w+1),
                ]:
                    if 0 <= nd < grid_d and 0 <= nh < grid_h and 0 <= nw < grid_w:
                        j = idx(nd, nh, nw)
                        adj[i, j] = 1.0
    return adj


# ── Classification Head ──────────────────────────────────────────
class DiseaseClassificationHead(nn.Module):
    """Multi-label global classification + per-node anomaly scoring."""

    def __init__(self, node_dim: int = 128, num_diseases: int = NUM_DISEASES,
                 temperature: float = 1.5) -> None:
        super().__init__()
        self.global_clf = nn.Sequential(
            nn.Linear(node_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_diseases),
        )
        self.local_severity = nn.Linear(node_dim, 1)
        self.local_binary = nn.Linear(node_dim, 1)
        # Temperature scaling for calibration
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, node_features: torch.Tensor) -> dict:
        """
        node_features: (B, N, node_dim)
        Returns dict with disease_logits, severity_map, local_binary
        """
        # Global classification via mean pooling
        global_feat = node_features.mean(dim=1)  # (B, node_dim)
        disease_logits = self.global_clf(global_feat)
        temperature = self.log_temperature.exp().clamp(0.5, 5.0)
        disease_logits = disease_logits / temperature

        # Per-node outputs
        severity = torch.sigmoid(self.local_severity(node_features)).squeeze(-1)  # (B, N)
        binary = torch.sigmoid(self.local_binary(node_features)).squeeze(-1)       # (B, N)

        return {
            "disease_logits": disease_logits,
            "severity_map": severity,
            "anomaly_binary": binary,
        }


# ── GAAT: Main Model ─────────────────────────────────────────────
class AnomalyGraph(nn.Module):
    """
    Graph-Augmented Anomaly Transformer (GAAT).
    Full implementation from program.md Section 1 / Subsystem C.

    Args:
        patch_size: 8 (gives 16³ = 4096 patches for 128³ volume)
        embed_dim: 64-dim patch embeddings
        graph_dim: 128-dim GraphSAGE hidden features
        num_sage_layers: 3
        num_diseases: 13
        memory_bank_path: pre-built .npz centroid file
    """

    PATCH_SIZE = 8

    def __init__(
        self,
        embed_dim: int = 64,
        graph_dim: int = 128,
        num_sage_layers: int = 3,
        num_diseases: int = NUM_DISEASES,
        memory_bank_size: int = 1024,
        memory_bank_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = self.PATCH_SIZE
        patch_flat = self.patch_size ** 3  # 512

        # Node feature = [patch_embed(64) | anomaly_score(1) | layer_class_distribution(12)]
        # = 64 + 1 + 12 = 77 → project to graph_dim
        node_feature_dim = embed_dim + 1 + 12

        self.patch_embed = PatchEmbedMLP(patch_flat, 256, embed_dim)
        self.memory_bank = NormalMemoryBank(memory_bank_size, embed_dim)

        if memory_bank_path and Path(memory_bank_path).exists():
            self.memory_bank.load_from_npz(memory_bank_path)

        self.node_proj = nn.Linear(node_feature_dim, graph_dim)

        self.sage_layers = nn.ModuleList([
            GraphSAGELayer(graph_dim if i > 0 else graph_dim, graph_dim)
            for i in range(num_sage_layers)
        ])

        self.clf_head = DiseaseClassificationHead(graph_dim, num_diseases)

        # Pre-build adjacency for 128³ volume (16³ patches)
        self.register_buffer("adj_128", build_patch_adjacency(16, 16, 16))

    def extract_patches(self, volume: torch.Tensor) -> torch.Tensor:
        """
        volume: (B, 1, D, H, W)
        Returns: (B, N, patch_flat) where N = num_patches
        """
        B, C, D, H, W = volume.shape
        ps = self.patch_size
        patches = volume.unfold(2, ps, ps).unfold(3, ps, ps).unfold(4, ps, ps)
        # patches: (B, C, nd, nh, nw, ps, ps, ps)
        nd, nh, nw = patches.shape[2], patches.shape[3], patches.shape[4]
        patches = patches.contiguous().view(B, nd * nh * nw, ps * ps * ps)
        return patches  # (B, N, 512)

    def forward(
        self,
        volume: torch.Tensor,
        seg_probs: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        volume:    (B, 1, D, H, W) normalized float
        seg_probs: (B, 12, D, H, W) optional segmentation probabilities

        Returns dict:
            anomaly_map_3d:    (B, D, H, W) voxel-level anomaly scores
            disease_probs:     (B, 13) multi-label disease probabilities
            severity_patches:  (B, N) per-patch severity scores
            disease_labels:    list[str]
        """
        B, C, D, H, W = volume.shape
        ps = self.patch_size
        nd, nh, nw = D // ps, H // ps, W // ps
        N = nd * nh * nw  # number of patches

        # ── 1. Extract and embed patches ───────────────────────
        patches = self.extract_patches(volume)  # (B, N, 512)
        embeddings = self.patch_embed(patches)   # (B, N, embed_dim) — normalized

        # ── 2. Anomaly score from memory bank ──────────────────
        B_flat = embeddings.view(B * N, self.embed_dim)
        anomaly_scores = self.memory_bank.anomaly_score(B_flat)  # (B*N,)
        anomaly_scores = anomaly_scores.view(B, N, 1)             # (B, N, 1)

        # ── 3. Layer class distribution per patch ──────────────
        if seg_probs is not None:
            # Average pool segmentation probs over each patch
            scale = D // seg_probs.shape[2] if seg_probs.shape[2] != D else 1
            seg_p = F.interpolate(seg_probs, size=(D, H, W), mode="trilinear",
                                  align_corners=False)
            seg_patches = seg_p.unfold(2, ps, ps).unfold(3, ps, ps).unfold(4, ps, ps)
            seg_patches = seg_patches.contiguous().view(B, 12, nd * nh * nw, ps**3)
            layer_dist = seg_patches.mean(dim=-1).permute(0, 2, 1)  # (B, N, 12)
        else:
            # Uniform distribution
            layer_dist = torch.ones(B, N, 12, device=volume.device) / 12.0

        # ── 4. Build node features ─────────────────────────────
        node_features = torch.cat([embeddings, anomaly_scores, layer_dist], dim=-1)  # (B, N, 77)
        node_features = F.gelu(self.node_proj(node_features))  # (B, N, graph_dim)

        # ── 5. GraphSAGE message passing ───────────────────────
        adj = self.adj_128[:N, :N] if N <= 4096 else self.adj_128
        adj = adj.to(volume.device)
        for sage_layer in self.sage_layers:
            node_features = sage_layer(node_features, adj)

        # ── 6. Classification ──────────────────────────────────
        clf_out = self.clf_head(node_features)

        # ── 7. Reshape anomaly map back to volume ──────────────
        severity = clf_out["severity_map"].view(B, nd, nh, nw)  # (B, nd, nh, nw)
        anomaly_map_3d = F.interpolate(
            severity.unsqueeze(1).float(), size=(D, H, W), mode="trilinear",
            align_corners=False
        ).squeeze(1)  # (B, D, H, W)

        return {
            "anomaly_map_3d": anomaly_map_3d,
            "disease_logits": clf_out["disease_logits"],
            "disease_probs": torch.sigmoid(clf_out["disease_logits"]),
            "severity_patches": clf_out["severity_map"],
            "anomaly_binary": clf_out["anomaly_binary"],
            "patch_anomaly_scores": anomaly_scores.squeeze(-1),
            "disease_labels": DISEASE_LABELS,
        }


def create_anomaly_graph(
    memory_bank_path: Optional[str] = None,
    embed_dim: int = 64,
    graph_dim: int = 128,
) -> AnomalyGraph:
    """Factory function for AnomalyGraph with optional pre-built memory bank."""
    return AnomalyGraph(
        embed_dim=embed_dim,
        graph_dim=graph_dim,
        num_sage_layers=3,
        num_diseases=NUM_DISEASES,
        memory_bank_size=1024,
        memory_bank_path=memory_bank_path,
    )
