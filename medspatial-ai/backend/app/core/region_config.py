"""
MedSpatial AI — Region Configuration
Per-region configuration for atlas selection, segmentation classes, disease
screening, window presets, and layer names/colors.
"""

from dataclasses import dataclass, field
from typing import Optional

from app.ai.body_region_classifier import BodyRegion


@dataclass
class WindowPreset:
    """CT window preset."""
    name: str
    center: float
    width: float


@dataclass
class RegionConfiguration:
    """Region-specific pipeline configuration."""
    region: BodyRegion
    display_name: str
    icon: str

    # Segmentation
    layer_names: list[str]
    layer_colors: list[tuple[float, float, float]]
    layer_descriptions: list[str]

    # Disease screening
    disease_screening: list[str]

    # Window presets (first is default)
    window_presets: list[WindowPreset]

    # Atlas
    atlas_filename: Optional[str] = None

    # Reconstruction defaults
    default_iso_level: float = 300.0
    default_step_size: int = 2


# ── Region configurations ────────────────────────────────────────

CHEST_CONFIG = RegionConfiguration(
    region=BodyRegion.CHEST,
    display_name="Chest",
    icon="🫁",
    layer_names=[
        "skin", "bone", "left_lung", "right_lung",
        "heart", "vessels", "soft_tissue", "pathology",
    ],
    layer_colors=[
        (0.90, 0.75, 0.65),  # skin
        (0.95, 0.92, 0.80),  # bone
        (0.40, 0.65, 0.85),  # left_lung
        (0.30, 0.55, 0.80),  # right_lung
        (0.90, 0.40, 0.40),  # heart
        (0.85, 0.20, 0.20),  # vessels
        (0.90, 0.70, 0.60),  # soft_tissue
        (1.00, 0.15, 0.00),  # pathology
    ],
    layer_descriptions=[
        "Skin and subcutaneous tissue",
        "Ribs, sternum, clavicles, scapulae, vertebrae",
        "Left lung parenchyma",
        "Right lung parenchyma",
        "Heart and cardiac silhouette",
        "Pulmonary and great vessels",
        "Musculature and soft tissue",
        "Abnormality / pathology regions",
    ],
    disease_screening=[
        "Pneumonia", "Pneumothorax", "Pleural Effusion",
        "Lung Nodule/Mass", "Cardiomegaly", "Atelectasis",
        "Consolidation", "Emphysema", "Fibrosis", "Fracture",
        "Tuberculosis", "COVID-19 patterns",
    ],
    window_presets=[
        WindowPreset("Lung", -600, 1500),
        WindowPreset("Mediastinum", 40, 400),
        WindowPreset("Bone", 400, 1800),
        WindowPreset("Soft Tissue", 50, 350),
    ],
    atlas_filename="atlas_chest_128.npz",
    default_iso_level=300.0,
)

HEAD_CONFIG = RegionConfiguration(
    region=BodyRegion.HEAD,
    display_name="Head",
    icon="🧠",
    layer_names=[
        "skin", "bone", "brain", "vessels", "soft_tissue", "pathology",
    ],
    layer_colors=[
        (0.90, 0.75, 0.65),
        (0.95, 0.92, 0.80),
        (0.80, 0.75, 0.85),
        (0.85, 0.20, 0.20),
        (0.90, 0.70, 0.60),
        (1.00, 0.15, 0.00),
    ],
    layer_descriptions=[
        "Scalp and skin",
        "Skull and facial bones",
        "Brain parenchyma",
        "Cerebral vessels",
        "Soft tissue",
        "Pathology regions",
    ],
    disease_screening=[
        "Hemorrhage", "Ischemic stroke", "Fracture", "Mass/Tumor",
        "Hydrocephalus", "Midline shift",
    ],
    window_presets=[
        WindowPreset("Brain", 40, 80),
        WindowPreset("Bone", 400, 1800),
        WindowPreset("Subdural", 75, 215),
        WindowPreset("Stroke", 32, 8),
    ],
    default_iso_level=500.0,
)

ABDOMEN_CONFIG = RegionConfiguration(
    region=BodyRegion.ABDOMEN,
    display_name="Abdomen",
    icon="🫁",
    layer_names=[
        "skin", "bone", "liver", "kidneys", "vessels", "soft_tissue", "pathology",
    ],
    layer_colors=[
        (0.90, 0.75, 0.65),
        (0.95, 0.92, 0.80),
        (0.70, 0.45, 0.35),
        (0.65, 0.50, 0.40),
        (0.85, 0.20, 0.20),
        (0.90, 0.70, 0.60),
        (1.00, 0.15, 0.00),
    ],
    layer_descriptions=[
        "Skin", "Spine and pelvis", "Liver",
        "Kidneys", "Vessels", "Soft tissue", "Pathology",
    ],
    disease_screening=[
        "Liver lesion", "Kidney stone", "Bowel obstruction",
        "Aneurysm", "Lymphadenopathy", "Fracture",
    ],
    window_presets=[
        WindowPreset("Abdomen", 60, 400),
        WindowPreset("Liver", 60, 150),
        WindowPreset("Bone", 400, 1800),
        WindowPreset("Soft Tissue", 50, 350),
    ],
    default_iso_level=200.0,
)

# Default config for any region not explicitly defined
DEFAULT_CONFIG = RegionConfiguration(
    region=BodyRegion.UNKNOWN,
    display_name="General",
    icon="📦",
    layer_names=["skin", "bone", "soft_tissue", "vessels", "pathology"],
    layer_colors=[
        (0.90, 0.75, 0.65),
        (0.95, 0.92, 0.80),
        (0.90, 0.70, 0.60),
        (0.85, 0.20, 0.20),
        (1.00, 0.15, 0.00),
    ],
    layer_descriptions=[
        "Skin", "Bone", "Soft tissue", "Vessels", "Pathology",
    ],
    disease_screening=[
        "Fracture", "Mass/Tumor", "Inflammation",
    ],
    window_presets=[
        WindowPreset("Soft Tissue", 50, 350),
        WindowPreset("Bone", 400, 1800),
    ],
    default_iso_level=300.0,
)


_REGION_CONFIGS: dict[BodyRegion, RegionConfiguration] = {
    BodyRegion.CHEST: CHEST_CONFIG,
    BodyRegion.HEAD: HEAD_CONFIG,
    BodyRegion.ABDOMEN: ABDOMEN_CONFIG,
    BodyRegion.NECK: HEAD_CONFIG,  # neck uses head config
    BodyRegion.PELVIS: ABDOMEN_CONFIG,  # pelvis uses abdomen config
    BodyRegion.SPINE: DEFAULT_CONFIG,
    BodyRegion.EXTREMITY: DEFAULT_CONFIG,
    BodyRegion.WHOLE_BODY: CHEST_CONFIG,
    BodyRegion.UNKNOWN: DEFAULT_CONFIG,
}


def get_region_config(region: BodyRegion) -> RegionConfiguration:
    """Get the pipeline configuration for a body region."""
    return _REGION_CONFIGS.get(region, DEFAULT_CONFIG)
