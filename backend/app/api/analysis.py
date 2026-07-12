"""
MedSpatial AI — Analysis API
Endpoints for AI-powered anomaly detection, segmentation, and layer dissection.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Analysis, Scan, ScanStatus, Volume
from app.models.database import get_db
from app.schemas import AnalysisRequest, AnalysisResponse, AnomalyFinding
from app.services.anomaly_service import AnomalyService

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])
anomaly_svc = AnomalyService()


async def _run_analysis(scan_id: str, analysis_id: str, analysis_type: str):
    """Background task: run AI analysis on reconstructed volume."""
    from app.models.database import async_session_factory

    async with async_session_factory() as db:
        try:
            result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result.scalar_one_or_none()
            if not analysis:
                return

            result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
            volume = result.scalar_one_or_none()
            if not volume:
                analysis.status = "failed"
                await db.commit()
                return

            analysis.status = "running"
            await db.commit()

            import numpy as np
            volume_data = np.load(volume.volume_path)

            if analysis_type in ("anomaly", "full"):
                anomaly_result = anomaly_svc.detect_anomalies(volume_data)
                heatmap_path = Path(settings.ANALYSIS_DIR) / f"{analysis_id}_heatmap.npy"
                np.save(str(heatmap_path), anomaly_result["heatmap"])
                analysis.heatmap_path = str(heatmap_path)
                analysis.findings = {
                    "anomalies": [f.dict() if hasattr(f, "dict") else f for f in anomaly_result["findings"]]
                }
                analysis.confidence = anomaly_result.get("overall_confidence", 0.0)
                analysis.summary = anomaly_result.get("summary", "Analysis complete.")

            if analysis_type in ("segmentation", "full"):
                seg_result = anomaly_svc.segment_organs(volume_data)
                seg_path = Path(settings.ANALYSIS_DIR) / f"{analysis_id}_segmentation.npy"
                np.save(str(seg_path), seg_result["mask"])
                analysis.segmentation_mask_path = str(seg_path)

            analysis.status = "completed"
            await db.commit()
            logger.info(f"Analysis {analysis_id} completed for scan {scan_id}")

        except Exception as exc:
            logger.error(f"Analysis failed: {exc}")
            analysis.status = "failed"
            await db.commit()


@router.post("/run", response_model=AnalysisResponse)
async def run_analysis(
    req: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger AI analysis on a reconstructed scan."""
    result = await db.execute(select(Scan).where(Scan.id == req.scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    result = await db.execute(select(Volume).where(Volume.scan_id == req.scan_id))
    volume = result.scalar_one_or_none()
    if not volume:
        raise HTTPException(status_code=409, detail="Scan must be reconstructed before analysis.")

    analysis = Analysis(
        id=str(uuid.uuid4()),
        scan_id=req.scan_id,
        analysis_type=req.analysis_type,
        status="pending",
    )
    db.add(analysis)
    await db.flush()

    background_tasks.add_task(_run_analysis, req.scan_id, analysis.id, req.analysis_type)

    return AnalysisResponse(
        analysis_id=analysis.id,
        scan_id=req.scan_id,
        status="pending",
        analysis_type=req.analysis_type,
    )


@router.get("/results/{scan_id}", response_model=list[AnalysisResponse])
async def get_analysis_results(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Get all analysis results for a scan."""
    result = await db.execute(
        select(Analysis).where(Analysis.scan_id == scan_id).order_by(Analysis.created_at.desc())
    )
    analyses = result.scalars().all()

    responses = []
    for a in analyses:
        findings = None
        if a.findings and "anomalies" in a.findings:
            findings = [AnomalyFinding(**f) for f in a.findings["anomalies"]]

        responses.append(AnalysisResponse(
            analysis_id=a.id,
            scan_id=a.scan_id,
            status=a.status,
            analysis_type=a.analysis_type,
            findings=findings,
            heatmap_url=f"/api/analysis/heatmap/{a.id}" if a.heatmap_path else None,
            segmentation_url=f"/api/analysis/segmentation/{a.id}" if a.segmentation_mask_path else None,
            summary=a.summary,
            confidence=a.confidence,
        ))

    return responses


@router.get("/heatmap/{analysis_id}")
async def get_heatmap(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Download the 3D anomaly heatmap as a numpy array."""
    from fastapi.responses import FileResponse

    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis or not analysis.heatmap_path:
        raise HTTPException(status_code=404, detail="Heatmap not found.")

    return FileResponse(analysis.heatmap_path, media_type="application/octet-stream")


@router.get("/segmentation/{analysis_id}")
async def get_segmentation(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Download the segmentation mask."""
    from fastapi.responses import FileResponse

    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis or not analysis.segmentation_mask_path:
        raise HTTPException(status_code=404, detail="Segmentation mask not found.")

    return FileResponse(analysis.segmentation_mask_path, media_type="application/octet-stream")
