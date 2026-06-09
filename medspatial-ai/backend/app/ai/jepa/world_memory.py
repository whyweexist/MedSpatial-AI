"""Object-centric latent memory spanning the whole body."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ANATOMICAL_SLOTS = (
    "brain", "cerebellum", "brainstem", "eyes", "sinuses", "neck",
    "thyroid", "left_lung", "right_lung", "lung_lobes", "heart",
    "aorta", "vessels", "bronchi", "trachea", "pleura", "diaphragm",
    "liver", "gallbladder", "spleen", "pancreas", "stomach",
    "small_bowel", "large_bowel", "left_kidney", "right_kidney",
    "bladder", "pelvic_organs", "cervical_spine", "thoracic_spine",
    "lumbar_spine", "sacrum", "spinal_cord", "bone_marrow", "ribs",
    "pelvis", "left_shoulder", "right_shoulder", "left_arm", "right_arm",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand",
    "right_hand", "left_hip", "right_hip", "left_femur", "right_femur",
    "left_knee", "right_knee", "left_leg", "right_leg", "left_ankle",
    "right_ankle", "left_foot", "right_foot", "skin", "muscle",
    "lymph_nodes", "lesions", "uncertain_regions", "current_viewport",
)


class WorldStateMemory(nn.Module):
    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.slot_names = ANATOMICAL_SLOTS
        self.slot_queries = nn.Parameter(torch.randn(len(self.slot_names), embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        queries = self.slot_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        slots, weights = self.attention(queries, tokens, tokens, need_weights=True)
        slots = F.normalize(self.norm(slots + queries), dim=-1)
        return {
            name: slots[:, index] for index, name in enumerate(self.slot_names)
        } | {"attention_weights": weights}
