"""Optional MONAI provider adapter."""

from app.ai.segmentation.providers.medsam2_adapter import MedSAM2Adapter


class MONAISegmenter(MedSAM2Adapter):
    name = "monai"
