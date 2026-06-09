"""Focused contract tests for the AWM-JGEM core."""

import torch

from app.ai.energy.energy_terms import EnergyTerms
from app.ai.energy.scorer import EnergyScorer
from app.ai.graph.graph_builder import build_scene_graph
from app.ai.jepa.inference import JEPAInferenceEngine
from app.awm.schema import (
    AnatomicalStructure,
    AnatomicalSystem,
    AnatomicalWorldModel,
    BodyRegion,
    EvidenceAnchor,
    FindingCandidate,
    GroundingType,
    Modality,
    ModelIdentity,
    StudyManifest,
)
from app.policy.intent_schema import Action, Intent
from app.policy.policy_engine import PolicyEngine
from app.reasoning.grounded_answer import answer_question


def base_awm() -> AnatomicalWorldModel:
    evidence = EvidenceAnchor(
        evidence_type="source_slice",
        reference_id="slice-12",
        label="Axial source slice 12",
        slice_index=12,
        confidence=0.8,
    )
    structure = AnatomicalStructure(
        id="left-knee",
        name="Left knee",
        canonical_name="knee joint",
        body_region=BodyRegion.KNEE,
        system=AnatomicalSystem.MUSCULOSKELETAL,
        laterality="left",
        confidence=0.8,
        uncertainty=0.2,
        evidence_ids=[evidence.id],
    )
    finding = FindingCandidate(
        label="Focal signal deviation",
        description="A localized candidate region is visible on the referenced slice.",
        structure_ids=[structure.id],
        confidence=0.7,
        uncertainty=0.3,
        evidence_ids=[evidence.id],
        model=ModelIdentity(name="test-model", version="1"),
    )
    return AnatomicalWorldModel(
        study=StudyManifest(
            study_id="study-1",
            deidentified_study_id="study-1",
            modality=Modality.MR,
            body_regions=[BodyRegion.KNEE],
            source_format="dicom",
            file_count=20,
            quality_score=0.9,
        ),
        structures=[structure],
        findings=[finding],
        evidence=[evidence],
    )


def test_awm_supports_whole_body_structure() -> None:
    awm = base_awm()
    assert awm.architecture == "AWM-JGEM"
    assert awm.structures[0].body_region == BodyRegion.KNEE


def test_xray_volume_must_be_estimated() -> None:
    from app.awm.schema import VolumeReference
    import pytest

    with pytest.raises(ValueError, match="must be marked estimated"):
        AnatomicalWorldModel(
            study=StudyManifest(
                study_id="xray", deidentified_study_id="xray",
                modality=Modality.XR, body_regions=[BodyRegion.CHEST],
                source_format="image", file_count=1, quality_score=1,
            ),
            volumes=[VolumeReference(
                uri="xray.npy", shape_zyx=(64, 64, 64),
                voxel_spacing_zyx_mm=(1, 1, 1),
                grounding=GroundingType.SCAN_DERIVED,
            )],
        )


def test_jepa_output_shape() -> None:
    output = JEPAInferenceEngine(embed_dim=32).infer(torch.rand(1, 1, 32, 32, 32))
    assert output.latent_tokens.shape[0] == 1
    assert output.latent_tokens.shape[-1] == 32
    assert "left_knee" in output.slot_embeddings
    assert 0 <= output.uncertainty <= 1


def test_scene_graph_creation() -> None:
    graph = build_scene_graph(base_awm())
    assert len(graph.nodes) == 2
    assert graph.edges[0].relation == "suspicious_for"


def test_energy_score_is_explainable() -> None:
    result = EnergyScorer().score(
        EnergyTerms(jepa_residual=0.2, graph_inconsistency=0.1, uncertainty_penalty=0.3),
        ["evidence-1"],
    )
    assert result.energy_score > 0
    assert "jepa_residual" in result.explanation_terms
    assert result.evidence_ids == ["evidence-1"]


def test_policy_allow_and_deny() -> None:
    engine = PolicyEngine()
    allowed = Intent(
        actor_id="clinician-1", actor_role="clinician",
        deidentified_study_id="study-1", study_id="study-1", modality="MR",
        requested_action=Action.ASK_CLINICAL_QUESTION,
    )
    blocked = Intent(
        actor_id="clinician-1", actor_role="clinician",
        deidentified_study_id="study-1", study_id="study-1", modality="MR",
        requested_action=Action.EXPORT_RAW_STUDY,
        sensitivity_level="phi",
    )
    assert engine.evaluate(allowed).allowed is True
    assert engine.evaluate(blocked).allowed is False


def test_grounded_answer_contains_evidence_and_uncertainty() -> None:
    result = answer_question(base_awm(), "What is visible in the left knee?")
    assert result.evidence_anchors
    assert "not a confirmed diagnosis" in result.answer
    assert result.uncertainty == 0.3


def test_spine_reconstruction_uses_spine_layers_and_labels() -> None:
    from app.ai.body_part_labeler import BodyPartLabeler
    from app.core.surface_processor import get_tissue_configs

    configs = get_tissue_configs("spine")
    names = {config.name for config in configs}
    assert "left_lung" not in names
    assert "heart" not in names
    assert {"bone", "bone_marrow", "spinal_canal"} <= names
    labeler = BodyPartLabeler()
    assert labeler._format_tissue_name("bone", "spine") == "Spinal Column"
    assert labeler._format_tissue_name("spinal_canal", "spine") == "Estimated Spinal Canal"


def test_cervical_spine_metadata_routes_to_spine() -> None:
    from app.ai.body_region_classifier import BodyRegion, BodyRegionClassifier

    classifier = BodyRegionClassifier()
    direct = classifier.classify_from_metadata({"modality": "CT", "body_part": "CSPINE"})
    described = classifier.classify_from_metadata({
        "modality": "MR",
        "study_description": "MRI cervical spine without contrast",
    })
    assert direct is not None and direct.region == BodyRegion.SPINE
    assert described is not None and described.region == BodyRegion.SPINE


def test_explainability_never_fabricates_fallback_heatmaps() -> None:
    import numpy as np

    from app.ai.explainability import ExplainabilityEngine

    engine = ExplainabilityEngine()
    result = engine.compute_full_xai(
        model=None,
        volume_tensor=None,
        findings=[],
        disease_probs=None,
        anomaly_map=None,
        volume=np.zeros((8, 8, 8), dtype=np.float32),
        modality="MR",
        body_region="spine",
    )
    assert result.grad_cam_heatmaps == {}
    assert result.limitations


def test_mri_explanation_does_not_claim_hounsfield_units() -> None:
    import numpy as np

    from app.ai.explainability import ExplainabilityEngine

    chains = ExplainabilityEngine().generate_reasoning_chain(
        findings=[{
            "description": "Candidate signal deviation",
            "confidence": 0.6,
            "severity": "mild",
            "region": "central region",
            "location": {"x": 4, "y": 4, "z": 4},
        }],
        disease_probs=None,
        anomaly_map=np.ones((8, 8, 8), dtype=np.float32) * 0.5,
        volume=np.ones((8, 8, 8), dtype=np.float32),
        modality="MR",
        body_region="spine",
    )
    descriptions = " ".join(step.description for step in chains[0].steps)
    assert "relative intensity" in descriptions
    assert "mean=1 HU" not in descriptions
