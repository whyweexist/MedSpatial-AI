# Runtime Profiles

## Lightweight

For 4-8 GB RAM and CPU-only systems. Uses 64-cubed inference, lazy loading,
INT8/ONNX hooks, a 256 MB tensor budget, deterministic fallbacks, and mesh LOD.

## Codespaces

Pins inference to CPU, limits volumes to 64-cubed, disables remote providers,
and reduces the tensor budget to 128 MB.

## Enhanced

Allows 128-cubed default inference, larger working volumes, and optional
configured MONAI/MedSAM2 providers. Core API and policy contracts do not
change.

Select a profile through `PROFILE`; provider weights and remote access remain
explicit configuration. Training is never required for application startup.
