"""
MedSpatial AI — Anomaly Service Shim
=====================================
This module is placed at app/services/anomaly_service.py inside the
Nuitka build so that the existing API routes (analysis.py) import
anomaly_svc from here unchanged.

The shim simply exposes the ONNX engine singleton under the same
name (anomaly_svc) that the rest of the codebase already uses.
No API routes need to be modified.

How it gets wired up:
  main_prod.py registers this module in sys.modules["app.services.anomaly_service"]
  BEFORE any app.api.* imports happen, so the monkey-patch is transparent.
"""

from onnx_inference import get_engine

# The engine singleton is already initialised by main_prod.py before this
# module is first imported.  get_engine() with no args returns it.
anomaly_svc = get_engine()
