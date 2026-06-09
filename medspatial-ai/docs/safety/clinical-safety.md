# Clinical Safety

AWM-JGEM is a research and engineering platform, not an autonomous diagnostic
device.

1. Candidate findings must include evidence, confidence, uncertainty,
   provenance, model version, modality, and grounding type.
2. Answers without evidence return an explicit insufficient-evidence response.
3. The language safety layer rejects unsupported diagnostic certainty.
4. X-ray-derived 3D and synthetic anatomy are visibly distinguished from
   scan-derived anatomy.
5. Remote transfer, raw export, deletion, and sharing are policy-controlled and
   audited.
6. Model weights bundled for fallback or structural testing are not clinically
   calibrated.

Do not use outputs for patient care without appropriate validation,
human oversight, local governance, and applicable regulatory authorization.
