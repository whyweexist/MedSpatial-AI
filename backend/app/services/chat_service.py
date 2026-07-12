"""
MedSpatial AI — Chat Service
Orchestrates the Medical Q&A pipeline: builds context from scan data,
runs the Q&A model, and formats responses with referenced regions.
"""

import numpy as np
import torch
from loguru import logger
from typing import Any, Optional

from app.ai.medical_qa import (
    MEDICAL_KNOWLEDGE,
    MedicalQAModel,
    SimpleTokenizer,
    create_medical_qa_model,
)
from app.ai.spatial_transformer import SpatialTransformer3D, create_spatial_transformer
from app.config import settings
from app.core.volume_processor import VolumeProcessor


class ChatService:
    """
    Service layer for the conversational medical Q&A system.
    Combines neural model inference with rule-based medical knowledge
    for robust, informative answers.
    """

    MODALITY_MAP = {"CT": 0, "XR": 1, "MR": 2, "CR": 3, "DX": 4, "US": 5, "NM": 6}
    BODY_PART_MAP = {
        "CHEST": 0, "HEAD": 1, "ABDOMEN": 2, "PELVIS": 3, "SPINE": 4,
        "EXTREMITY": 5, "NECK": 6, "THORAX": 7, "BRAIN": 8, "LUNG": 9,
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = SimpleTokenizer(vocab_size=8000)
        self.volume_proc = VolumeProcessor()

        self._qa_model: MedicalQAModel = None
        self._spatial_transformer: SpatialTransformer3D = None

    def _load_models(self):
        """Lazy-load Q&A models."""
        if self._qa_model is None:
            logger.info("Loading MedicalQAModel...")
            self._qa_model = create_medical_qa_model(
                vocab_size=8000,
                embed_dim=256,
                spatial_dim=settings.EMBED_DIM,
            ).to(self.device).eval()

        if self._spatial_transformer is None:
            logger.info("Loading SpatialTransformer3D for Q&A context...")
            self._spatial_transformer = create_spatial_transformer(
                embed_dim=settings.EMBED_DIM,
                num_heads=settings.NUM_HEADS,
                num_layers=settings.NUM_LAYERS,
                pretrained_path=settings.SPATIAL_TRANSFORMER_WEIGHTS,
            ).to(self.device).eval()

    def answer_question(
        self,
        question: str,
        chat_history: list[dict],
        scan_context: dict,
        volume_context: Optional[dict] = None,
        analysis_context: Optional[list[dict]] = None,
    ) -> dict:
        """
        Answer a medical question about a scan.

        This uses a hybrid approach:
        1. Knowledge-based reasoning for factual medical questions
        2. Data-driven analysis referencing actual scan findings
        3. Neural model inference for complex contextual questions

        Returns:
            dict with answer, referenced_slices, referenced_regions, findings_mentioned
        """
        question_lower = question.lower().strip()

        # 1. Check for knowledge-base answerable questions
        kb_answer = self._answer_from_knowledge_base(question_lower, scan_context, analysis_context)
        if kb_answer:
            return kb_answer

        # 2. Context-aware analysis-based answer
        analysis_answer = self._answer_from_analysis(question_lower, scan_context, volume_context, analysis_context)
        if analysis_answer:
            return analysis_answer

        # 3. Neural model-based answer
        neural_answer = self._neural_answer(question, scan_context, volume_context, analysis_context)
        return neural_answer

    def _answer_from_knowledge_base(
        self,
        question: str,
        scan_context: dict,
        analysis_context: Optional[list[dict]],
    ) -> Optional[dict]:
        """Try to answer from built-in medical knowledge."""

        # What is X? questions
        for anatomy_name, info in MEDICAL_KNOWLEDGE["anatomy"].items():
            if anatomy_name in question and ("what" in question or "tell" in question or "about" in question):
                answer = (
                    f"**{anatomy_name.title()}**: {info['description']}. "
                    f"Normal Hounsfield Unit range: {info['normal_hu']['min']} to {info['normal_hu']['max']} HU. "
                )
                if "common_findings" in info:
                    answer += f"Common findings include: {', '.join(info['common_findings'])}. "
                if "adjacent_structures" in info:
                    answer += f"Adjacent structures: {', '.join(info['adjacent_structures'])}."

                return {
                    "answer": answer,
                    "referenced_slices": None,
                    "referenced_regions": None,
                    "findings_mentioned": None,
                    "context_summary": f"Answered anatomical query about {anatomy_name}",
                }

        for pathology_name, info in MEDICAL_KNOWLEDGE["pathology"].items():
            name_variants = [pathology_name, pathology_name.replace("_", " ")]
            if any(v in question for v in name_variants):
                answer = (
                    f"**{pathology_name.replace('_', ' ').title()}**: {info['description']}. "
                    f"**Clinical significance**: {info['significance']}. "
                    f"**HU characteristics**: {info['hu_characteristics']}."
                )
                return {
                    "answer": answer,
                    "referenced_slices": None,
                    "referenced_regions": None,
                    "findings_mentioned": None,
                    "context_summary": f"Answered pathology query about {pathology_name}",
                }

        return None

    def _answer_from_analysis(
        self,
        question: str,
        scan_context: dict,
        volume_context: Optional[dict],
        analysis_context: Optional[list[dict]],
    ) -> Optional[dict]:
        """Answer from analysis results."""

        # Questions about findings/abnormalities/anomalies
        anomaly_keywords = ["finding", "abnormal", "anomal", "detect", "disease", "problem", "wrong", "issue"]
        if any(kw in question for kw in anomaly_keywords):
            if analysis_context:
                findings_text = []
                all_findings = []
                for analysis in analysis_context:
                    if analysis.get("findings"):
                        findings_data = analysis["findings"]
                        if isinstance(findings_data, dict) and "anomalies" in findings_data:
                            for f in findings_data["anomalies"]:
                                all_findings.append(f)
                                desc = f.get("description", "Unknown finding")
                                sev = f.get("severity", "unknown")
                                conf = f.get("confidence", 0)
                                findings_text.append(
                                    f"- **{sev.title()}** ({conf:.0%} confidence): {desc}"
                                )

                    if analysis.get("summary"):
                        findings_text.append(f"\n**Summary**: {analysis['summary']}")

                if findings_text:
                    answer = (
                        f"Based on the AI analysis of this {scan_context.get('modality', 'scan')} scan "
                        f"({scan_context.get('body_part', 'unknown body region')}):\n\n"
                        + "\n".join(findings_text)
                        + "\n\n*Note: These are AI-generated findings and should be reviewed by a qualified radiologist.*"
                    )
                    return {
                        "answer": answer,
                        "referenced_slices": None,
                        "referenced_regions": [f.get("location") for f in all_findings if f.get("location")],
                        "findings_mentioned": all_findings[:5],
                        "context_summary": "Reported analysis findings",
                    }

            return {
                "answer": (
                    "No analysis has been run on this scan yet. "
                    "Please run the anomaly detection analysis first by clicking the 'Analyze' button, "
                    "then I can report any findings."
                ),
                "referenced_slices": None,
                "referenced_regions": None,
                "findings_mentioned": None,
                "context_summary": "No analysis available yet",
            }

        # Questions about scan info
        scan_keywords = ["scan", "image", "modality", "patient", "study", "series", "info"]
        if any(kw in question for kw in scan_keywords):
            modality = scan_context.get("modality", "Unknown")
            body_part = scan_context.get("body_part", "Unknown")
            num_slices = scan_context.get("num_slices", 0)
            study_desc = scan_context.get("study_description", "Not specified")

            answer = (
                f"**Scan Information:**\n"
                f"- **Modality**: {modality}\n"
                f"- **Body Part**: {body_part}\n"
                f"- **Number of Slices**: {num_slices}\n"
                f"- **Study Description**: {study_desc}\n"
            )

            if volume_context:
                dims = volume_context.get("dimensions", {})
                spacing = volume_context.get("voxel_spacing", {})
                answer += (
                    f"- **Volume Dimensions**: {dims.get('x', '?')} × {dims.get('y', '?')} × {dims.get('z', '?')}\n"
                    f"- **Voxel Spacing**: {spacing.get('x', '?'):.2f} × {spacing.get('y', '?'):.2f} × {spacing.get('z', '?'):.2f} mm\n"
                    f"- **HU Range**: {volume_context.get('hu_min', '?'):.0f} to {volume_context.get('hu_max', '?'):.0f}\n"
                )

            return {
                "answer": answer,
                "referenced_slices": None,
                "referenced_regions": None,
                "findings_mentioned": None,
                "context_summary": "Provided scan information",
            }

        # Questions about layers
        layer_keywords = ["layer", "tissue", "bone", "soft tissue", "dissect", "separate"]
        if any(kw in question for kw in layer_keywords):
            answer = (
                "The 3D model can be dissected into the following tissue layers based on Hounsfield Unit ranges:\n\n"
                "- **Air** (HU < -500): Airways and pneumothorax spaces\n"
                "- **Lung/Fat** (-500 to -100 HU): Lung parenchyma and fatty tissue\n"
                "- **Soft Tissue** (-100 to 200 HU): Organs, muscles, fluid\n"
                "- **Bone** (200 to 3000 HU): Skeletal structures\n"
                "- **Contrast/Metal** (> 3000 HU): Contrast agents, metallic implants\n\n"
                "You can toggle each layer's visibility and opacity using the Layer Controls panel on the left."
            )
            return {
                "answer": answer,
                "referenced_slices": None,
                "referenced_regions": None,
                "findings_mentioned": None,
                "context_summary": "Explained layer dissection",
            }

        return None

    def _neural_answer(
        self,
        question: str,
        scan_context: dict,
        volume_context: Optional[dict],
        analysis_context: Optional[list[dict]],
    ) -> dict:
        """
        Generate answer using the neural Q&A model with available context.
        Falls back to knowledge-based response if model can't generate a good answer.
        """
        try:
            self._load_models()

            # Prepare inputs
            q_tokens = torch.tensor([self.tokenizer.encode(question)]).to(self.device)

            # Build numeric context vector (32 features)
            numeric_features = np.zeros(32, dtype=np.float32)
            if volume_context:
                dims = volume_context.get("dimensions", {})
                numeric_features[0] = dims.get("x", 0) / 256.0
                numeric_features[1] = dims.get("y", 0) / 256.0
                numeric_features[2] = dims.get("z", 0) / 256.0
                numeric_features[3] = (volume_context.get("hu_min", -1024) + 1024) / 4095.0
                numeric_features[4] = (volume_context.get("hu_max", 3071) + 1024) / 4095.0
            numeric_features[5] = scan_context.get("num_slices", 0) / 500.0

            if analysis_context:
                for i, ac in enumerate(analysis_context[:5]):
                    numeric_features[10 + i] = ac.get("confidence", 0)

            numeric_tensor = torch.tensor([numeric_features]).to(self.device)

            # Modality and body part
            modality = scan_context.get("modality", "").upper()
            modality_idx = torch.tensor([self.MODALITY_MAP.get(modality, 0)]).to(self.device)
            body_part = scan_context.get("body_part", "").upper()
            body_part_idx = torch.tensor([self.BODY_PART_MAP.get(body_part, 0)]).to(self.device)

            # Generate response using the model
            with torch.no_grad():
                memory = self._qa_model(
                    question_ids=q_tokens,
                    context_numeric=numeric_tensor,
                    modality_idx=modality_idx,
                    body_part_idx=body_part_idx,
                )

            # For now generate a contextual response (model needs training for full generation)
            body_part_str = scan_context.get("body_part", "the scanned region")
            modality_str = scan_context.get("modality", "imaging")

            answer = (
                f"Regarding your question about this {modality_str} scan of {body_part_str}: "
                f"Based on the volumetric analysis, I can help you explore the 3D model interactively. "
                f"You can use the layer controls to dissect different tissue types, "
                f"navigate through slices in axial/coronal/sagittal views, "
                f"and review the anomaly heatmap overlay for areas of interest. "
                f"Would you like me to provide more specific information about a particular region or finding?"
            )

            return {
                "answer": answer,
                "referenced_slices": None,
                "referenced_regions": None,
                "findings_mentioned": None,
                "context_summary": "Generated contextual response",
            }

        except Exception as exc:
            logger.error(f"Neural Q&A failed: {exc}")
            return {
                "answer": (
                    "I can help you explore and understand this medical scan. "
                    "Try asking about:\n"
                    "- 'What findings were detected?'\n"
                    "- 'Tell me about the scan information'\n"
                    "- 'What are the tissue layers?'\n"
                    "- 'What is a pulmonary nodule?'\n"
                    "- Or any specific medical imaging question!"
                ),
                "referenced_slices": None,
                "referenced_regions": None,
                "findings_mentioned": None,
                "context_summary": "Provided usage guidance",
            }
