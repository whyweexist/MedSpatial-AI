"""
MedSpatial AI — XAI Explain API
Endpoints for explainable AI: Grad-CAM heatmaps, reasoning chains, attributions.
"""

import json
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Analysis, Scan, Volume
from app.models.database import get_db

router = APIRouter(prefix="/api/analysis/explain", tags=["Explainability"])


@router.post("/{scan_id}")
async def compute_explanations(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Compute all XAI outputs for a scan's analysis results."""
    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()
    if not volume:
        raise HTTPException(status_code=409, detail="Scan not reconstructed")

    result = await db.execute(
        select(Analysis).where(Analysis.scan_id == scan_id)
        .order_by(Analysis.created_at.desc())
    )
    analysis = result.scalar_one_or_none()

    try:
        from app.ai.explainability import ExplainabilityEngine

        engine = ExplainabilityEngine(device="cpu")

        # Load volume
        vol_data = None
        try:
            vol_data = np.load(volume.volume_path)
        except Exception:
            pass

        # Get findings
        findings = []
        disease_probs = None
        anomaly_map = None

        if analysis and analysis.findings:
            raw_findings = analysis.findings
            if isinstance(raw_findings, dict):
                findings = raw_findings.get("anomalies", [])
            elif isinstance(raw_findings, list):
                findings = raw_findings

        if analysis and analysis.heatmap_path:
            try:
                anomaly_map = np.load(analysis.heatmap_path)
            except Exception:
                pass

        # Compute XAI
        xai_result = engine.compute_full_xai(
            model=None,
            volume_tensor=None,
            findings=findings,
            disease_probs=disease_probs,
            anomaly_map=anomaly_map,
            volume=vol_data,
        )

        # Save heatmaps
        xai_dir = Path(settings.ANALYSIS_DIR) / "xai"
        xai_dir.mkdir(parents=True, exist_ok=True)

        saved_heatmaps = {}
        for disease_class, heatmap in xai_result.grad_cam_heatmaps.items():
            safe_name = disease_class.replace("/", "_").replace(" ", "_").lower()
            hmap_path = str(xai_dir / f"{scan_id}_{safe_name}_gradcam.npy")
            np.save(hmap_path, heatmap)
            saved_heatmaps[disease_class] = f"/api/analysis/explain/{scan_id}/heatmap/{safe_name}"

        # Save reasoning
        reasoning_data = []
        for chain in xai_result.reasoning_chains:
            chain_dict = {
                "finding": chain.finding,
                "confidence": chain.confidence,
                "anatomical_context": chain.anatomical_context,
                "differential": chain.differential,
                "bbox_3d": chain.bbox_3d,
                "representative_slice_idx": chain.representative_slice_idx,
                "steps": [
                    {
                        "category": s.category,
                        "description": s.description,
                        "confidence": s.confidence,
                        "evidence_type": s.evidence_type,
                    }
                    for s in chain.steps
                ],
            }
            reasoning_data.append(chain_dict)

        reasoning_path = str(xai_dir / f"{scan_id}_reasoning.json")
        with open(reasoning_path, "w") as f:
            json.dump(reasoning_data, f, indent=2)

        return JSONResponse({
            "scan_id": scan_id,
            "status": "completed",
            "heatmaps": saved_heatmaps,
            "reasoning_url": f"/api/analysis/explain/{scan_id}/reasoning",
            "num_reasoning_chains": len(reasoning_data),
        })

    except Exception as exc:
        logger.error(f"XAI computation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"XAI computation failed: {str(exc)}")


@router.get("/{scan_id}/heatmap/{disease_class}")
async def get_xai_heatmap(scan_id: str, disease_class: str):
    """Download a Grad-CAM heatmap for a specific disease class."""
    xai_dir = Path(settings.ANALYSIS_DIR) / "xai"
    hmap_path = xai_dir / f"{scan_id}_{disease_class}_gradcam.npy"

    if not hmap_path.exists():
        raise HTTPException(status_code=404, detail=f"Heatmap for '{disease_class}' not found")

    return FileResponse(
        str(hmap_path),
        media_type="application/octet-stream",
        filename=f"{scan_id}_{disease_class}_gradcam.npy",
    )


@router.get("/{scan_id}/reasoning")
async def get_reasoning_chains(scan_id: str):
    """Get structured reasoning chains for a scan."""
    xai_dir = Path(settings.ANALYSIS_DIR) / "xai"
    reasoning_path = xai_dir / f"{scan_id}_reasoning.json"

    if not reasoning_path.exists():
        raise HTTPException(status_code=404, detail="Reasoning chains not computed yet")

    with open(reasoning_path) as f:
        data = json.load(f)

    return JSONResponse({"scan_id": scan_id, "reasoning_chains": data})
