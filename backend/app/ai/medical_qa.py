"""
MedSpatial AI — Medical Q&A Module
Multimodal medical Q&A system that combines 3D volumetric context with natural language
understanding to answer questions about medical scans.

Architecture:
    - Context Encoder: processes scan metadata, analysis results, and volume features
    - Question Encoder: transforms text queries into semantic embeddings
    - Cross-Attention Decoder: attends over spatial features to generate answers
    - Medical Knowledge Base: built-in anatomical and pathological knowledge
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  Medical Knowledge Base (built-in anatomical knowledge)
# ──────────────────────────────────────────────────────────────

MEDICAL_KNOWLEDGE = {
    "anatomy": {
        "lung": {
            "description": "Paired organs for gas exchange in the thoracic cavity",
            "normal_hu": {"min": -900, "max": -500},
            "common_findings": ["nodule", "consolidation", "ground glass opacity", "pneumothorax", "pleural effusion"],
            "adjacent_structures": ["heart", "mediastinum", "diaphragm", "ribs", "trachea"],
        },
        "heart": {
            "description": "Muscular organ pumping blood through the circulatory system",
            "normal_hu": {"min": 30, "max": 60},
            "common_findings": ["cardiomegaly", "pericardial effusion", "coronary calcification"],
            "adjacent_structures": ["lung", "aorta", "esophagus", "sternum"],
        },
        "bone": {
            "description": "Dense connective tissue forming the skeletal system",
            "normal_hu": {"min": 200, "max": 1500},
            "common_findings": ["fracture", "osteoporosis", "lytic lesion", "sclerotic lesion"],
        },
        "liver": {
            "description": "Largest solid organ, located in the right upper abdomen",
            "normal_hu": {"min": 40, "max": 70},
            "common_findings": ["hepatic mass", "cirrhosis", "fatty liver", "hepatomegaly"],
        },
        "spine": {
            "description": "Vertebral column providing structural support",
            "normal_hu": {"min": 200, "max": 1200},
            "common_findings": ["compression fracture", "disc herniation", "spondylosis", "metastasis"],
        },
    },
    "pathology": {
        "nodule": {
            "description": "A small round or oval lesion, commonly found in lungs",
            "significance": "May be benign (granuloma) or malignant (cancer). Size > 8mm warrants follow-up.",
            "hu_characteristics": "Usually 20-150 HU depending on composition",
        },
        "consolidation": {
            "description": "Region where air in alveoli is replaced by fluid, pus, or cells",
            "significance": "Common in pneumonia, indicates active infection or inflammation",
            "hu_characteristics": "Higher density than normal lung, 0-50 HU",
        },
        "fracture": {
            "description": "A break in bone continuity",
            "significance": "Acute fractures require clinical correlation for management",
            "hu_characteristics": "Disruption in cortical bone with possible displacement",
        },
        "pleural_effusion": {
            "description": "Fluid accumulation in the pleural space",
            "significance": "Can be transudative or exudative, may require drainage",
            "hu_characteristics": "0-20 HU for simple effusions, higher for complex",
        },
        "ground_glass_opacity": {
            "description": "Hazy area of increased opacity in the lung that does not obscure underlying structures",
            "significance": "Seen in COVID-19, pulmonary edema, alveolar hemorrhage",
            "hu_characteristics": "-600 to -300 HU",
        },
    },
}


# ──────────────────────────────────────────────────────────────
#  Neural Components
# ──────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Basic word-level tokenizer for medical text."""

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self._build_medical_vocab()

    def _build_medical_vocab(self):
        """Build vocabulary from medical knowledge base."""
        idx = len(self.word2idx)
        words = set()

        # Extract words from knowledge base
        for category in MEDICAL_KNOWLEDGE.values():
            for item_data in category.values():
                if isinstance(item_data, dict):
                    for val in item_data.values():
                        if isinstance(val, str):
                            words.update(val.lower().split())
                        elif isinstance(val, list):
                            for v in val:
                                words.update(str(v).lower().split())

        # Common medical and question words
        common_words = [
            "what", "is", "the", "are", "there", "any", "show", "me", "can", "you",
            "find", "detect", "abnormal", "normal", "where", "how", "large", "small",
            "left", "right", "upper", "lower", "anterior", "posterior", "lateral",
            "medial", "bilateral", "unilateral", "acute", "chronic", "mild", "moderate",
            "severe", "density", "opacity", "lesion", "mass", "tumor", "cyst", "fluid",
            "air", "tissue", "organ", "structure", "scan", "image", "volume", "slice",
            "axial", "coronal", "sagittal", "ct", "xray", "mri", "dicom", "patient",
            "findings", "diagnosis", "report", "analysis", "anomaly", "heatmap",
            "layer", "bone", "muscle", "fat", "blood", "vessel", "artery", "vein",
            "lung", "heart", "liver", "kidney", "brain", "spine", "rib", "chest",
            "abdomen", "pelvis", "head", "neck", "thorax", "fracture", "nodule",
            "infection", "inflammation", "cancer", "benign", "malignant", "metastasis",
            "at", "in", "on", "of", "to", "with", "from", "about", "this", "that",
            "a", "an", "and", "or", "not", "no", "yes", "do", "does", "did", "have",
            "has", "had", "will", "would", "could", "should", "may", "might",
            "size", "shape", "location", "position", "region", "area", "part",
            "dissect", "separate", "isolate", "highlight", "compare",
        ]
        words.update(common_words)

        for word in sorted(words):
            if word not in self.word2idx and idx < self.vocab_size:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text: str, max_length: int = 64) -> list[int]:
        """Tokenize text to indices."""
        tokens = [self.word2idx.get("<bos>", 2)]
        for word in text.lower().split():
            word = word.strip(".,?!;:()")
            tokens.append(self.word2idx.get(word, self.word2idx["<unk>"]))
        tokens.append(self.word2idx.get("<eos>", 3))

        # Pad or truncate
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, indices: list[int]) -> str:
        """Decode indices to text."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, "<unk>")
            if word == "<eos>":
                break
            if word not in ("<pad>", "<bos>"):
                words.append(word)
        return " ".join(words)


class ContextEncoder(nn.Module):
    """Encodes scan context (metadata, analysis results, volume stats) into embeddings."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        # Encode numerical features
        self.numeric_encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        # Encode categorical features via embeddings
        self.modality_embed = nn.Embedding(8, embed_dim // 4)
        self.body_part_embed = nn.Embedding(16, embed_dim // 4)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, numeric_features: torch.Tensor, modality_idx: torch.Tensor, body_part_idx: torch.Tensor) -> torch.Tensor:
        num_enc = self.numeric_encoder(numeric_features)
        mod_enc = self.modality_embed(modality_idx)
        body_enc = self.body_part_embed(body_part_idx)
        combined = torch.cat([num_enc, mod_enc, body_enc], dim=-1)
        return self.fusion(combined)


class QuestionEncoder(nn.Module):
    """Encodes text questions into semantic embeddings."""

    def __init__(self, vocab_size: int = 8000, embed_dim: int = 256, max_length: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=512,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(token_ids) + self.pos_embed(positions)

        padding_mask = (token_ids == 0)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        # Pool: use first non-padding token (BOS)
        return x[:, 0]


class CrossAttentionDecoder(nn.Module):
    """
    Decoder that generates answers by cross-attending over spatial features and context.
    """

    def __init__(self, embed_dim: int = 256, vocab_size: int = 8000, max_length: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=512,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        target_ids: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        B, L = target_ids.shape
        positions = torch.arange(L, device=target_ids.device).unsqueeze(0).expand(B, -1)

        tgt = self.token_embed(target_ids) + self.pos_embed(positions)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=target_ids.device)

        x = self.transformer_decoder(tgt, memory, tgt_mask=causal_mask)
        x = self.norm(x)
        logits = self.output_proj(x)
        return logits


class MedicalQAModel(nn.Module):
    """
    Full Medical Q&A model combining context encoding, question understanding,
    cross-attention decoding, and medical knowledge integration.
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        embed_dim: int = 256,
        spatial_dim: int = 512,
        max_length: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.context_encoder = ContextEncoder(embed_dim)
        self.question_encoder = QuestionEncoder(vocab_size, embed_dim)
        self.decoder = CrossAttentionDecoder(embed_dim, vocab_size, max_length)

        # Project spatial features to decoder dimension
        self.spatial_proj = nn.Linear(spatial_dim, embed_dim)

        # Combine question + context into memory for decoder
        self.memory_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        context_numeric: torch.Tensor,
        modality_idx: torch.Tensor,
        body_part_idx: torch.Tensor,
        spatial_features: torch.Tensor = None,
        target_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        # Encode question and context
        q_emb = self.question_encoder(question_ids)  # (B, D)
        c_emb = self.context_encoder(context_numeric, modality_idx, body_part_idx)  # (B, D)

        # Fuse into memory
        fused = self.memory_fusion(torch.cat([q_emb, c_emb], dim=-1))  # (B, D)
        memory = fused.unsqueeze(1)  # (B, 1, D)

        # Add spatial features if available
        if spatial_features is not None:
            sf = self.spatial_proj(spatial_features)  # (B, N, D)
            memory = torch.cat([memory, sf], dim=1)  # (B, 1+N, D)

        # Decode
        if target_ids is not None:
            logits = self.decoder(target_ids, memory)
            return logits

        return memory


def create_medical_qa_model(
    vocab_size: int = 8000,
    embed_dim: int = 256,
    spatial_dim: int = 512,
    pretrained_path: str = None,
) -> MedicalQAModel:
    """Factory function for MedicalQAModel."""
    model = MedicalQAModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        spatial_dim=spatial_dim,
    )
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    return model
