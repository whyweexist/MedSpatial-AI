# AWM-JGEM Architecture

AWM-JGEM is the internal architecture name for the Anatomical World Model with
JEPA prediction, graph reasoning, energy modeling, and lightweight
mixture-of-experts routing.

## Migration Summary

The legacy pipeline remains available:

`image -> volume -> segmentation/anomaly -> mesh -> viewer`

The new control plane is:

`study -> AWM -> JEPA latent state -> scene graph -> energy score -> grounded reasoning -> policy gateway -> deterministic executor`

Preserved modules include DICOM parsing, slice sorting, HU conversion, hardware
adaptation, volume processing, segmentation networks, anomaly models,
marching-cubes surfaces, GLB export, and the React/Three.js viewer.

## Source Of Truth

`backend/app/awm/schema.py` defines study manifests, volumes, images,
segmentations, structures, findings, measurements, evidence, provenance,
scene graph state, latent state, processing events, and view state.

The schema covers the full body through body-region and anatomical-system
taxonomies plus extensible structure identifiers. JEPA object memory includes
the major organs, spine levels, bone marrow, bilateral upper and lower
extremities, knees, ankles, feet, uncertain regions, and the viewport.

## Grounding Rules

- CT/MR volumetric assets may be scan-derived.
- Single-view X-ray 3D is always estimated, probabilistic, atlas-aligned, and
  non-ground-truth.
- No-image atlas mode is synthetic, educational, non-diagnostic, and
  non-patient-specific.
- Findings are candidates, never certain diagnoses.
- Findings require evidence anchors and uncertainty.

## Model Layers

- JEPA: compact 3D context encoder, EMA target encoder, latent predictor,
  object-slot world memory, composable losses, and ONNX export.
- Graph: scene graph builder, pure-PyTorch graph attention, deterministic graph
  queries, and evidence-preserving explanations.
- Energy: independently weighted residual, graph, topology, boundary, density,
  and uncertainty terms.
- MoE: deterministic selection of small experts based on modality, region,
  task, weights, and memory budget.
- Segmentation: provider abstraction with local, MONAI, MedSAM2, and HU
  fallback implementations.

## Security Boundary

The reasoning layer only produces `Intent` objects. `PolicyEngine` evaluates
every privileged operation, `ExecutorGateway` is the sole execution bridge,
and `AuditLogger` writes PHI-minimized append-only events.

Clinical deployment requires independent model validation, regulatory review,
quality management, cybersecurity review, and clinical evaluation. This
repository does not claim regulatory approval.
