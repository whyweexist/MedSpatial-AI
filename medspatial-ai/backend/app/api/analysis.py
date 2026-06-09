"""
MedSpatial AI — Analysis API
Endpoints for AI-powered anomaly detection, segmentation, and layer dissection.
"""

import asyncio
import datetime
import time
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

def _estimate_duration_seconds(analysis_type: str) -> int:
    device = anomaly_svc.device.type
    if not anomaly_svc._has_anomaly_weights and not anomaly_svc._has_segmentation_weights:
        return 20 if analysis_type == "full" else 12
    if device == "cuda":
        return 45 if analysis_type == "full" else 30
    return 180 if analysis_type == "full" else 120


async def _set_progress(
    db: AsyncSession,
    analysis: Analysis,
    progress: float,
    stage: str,
    started_at: float,
    estimated_total: int,
    methodology: str | None = None,
) -> None:
    elapsed = max(0.0, time.time() - started_at)
    eta = max(0, int(estimated_total - elapsed))
    analysis.reasoning_json = {
        **(analysis.reasoning_json or {}),
        "progress": progress,
        "stage": stage,
        "started_at": datetime.datetime.fromtimestamp(
            started_at, tz=datetime.timezone.utc
        ).isoformat(),
        "estimated_total_seconds": estimated_total,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "methodology": methodology,
    }
    await db.commit()


async def _run_analysis(scan_id: str, analysis_id: str, analysis_type: str):
    """Background task: run AI analysis on reconstructed volume."""
    from app.models.database import async_session_factory

    started_at = time.time()
    estimated_total = _estimate_duration_seconds(analysis_type)
    analysis = None
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

            scan_result = await db.execute(select(Scan).where(Scan.id == scan_id))
            scan = scan_result.scalar_one_or_none()
            body_region = (scan.body_region or scan.body_part or "unknown").lower() if scan else "unknown"
            modality = (scan.modality or "CT").upper() if scan else "CT"

            analysis.status = "running"
            await _set_progress(db, analysis, 5, "loading_volume", started_at, estimated_total)

            import numpy as np
            volume_data = np.load(volume.volume_path, mmap_mode="r")
            await _set_progress(db, analysis, 12, "preparing_inference", started_at, estimated_total)

            methodology = None
            if analysis_type == "full":
                await _set_progress(db, analysis, 20, "analyzing_volume", started_at, estimated_total)
                full_result = await asyncio.to_thread(
                    anomaly_svc.analyze_full, volume_data, body_region, modality
                )
                anomaly_result = full_result["anomaly"]
                seg_result = full_result["segmentation"]
                methodology = full_result["methodology"]
            else:
                anomaly_result = None
                seg_result = None

            if analysis_type == "anomaly":
                await _set_progress(db, analysis, 20, "detecting_candidates", started_at, estimated_total)
                anomaly_result = await asyncio.to_thread(
                    anomaly_svc.detect_anomalies, volume_data, body_region, modality
                )
                methodology = (
                    "trained_neural"
                    if anomaly_svc._has_anomaly_weights
                    else "deterministic_fallback"
                )

            if anomaly_result is not None:
                await _set_progress(
                    db, analysis, 72, "saving_anomaly_evidence",
                    started_at, estimated_total, methodology,
                )
                heatmap_path = Path(settings.ANALYSIS_DIR) / f"{analysis_id}_heatmap.npy"
                np.save(str(heatmap_path), anomaly_result["heatmap"])
                analysis.heatmap_path = str(heatmap_path)
                analysis.findings = {
                    "anomalies": [f.dict() if hasattr(f, "dict") else f for f in anomaly_result["findings"]]
                }
                analysis.confidence = anomaly_result.get("overall_confidence", 0.0)
                analysis.summary = anomaly_result.get("summary", "Analysis complete.")

            if analysis_type == "segmentation":
                await _set_progress(db, analysis, 25, "segmenting_anatomy", started_at, estimated_total)
                seg_result = await asyncio.to_thread(
                    anomaly_svc.segment_organs, volume_data, modality
                )
                methodology = (
                    "trained_neural"
                    if anomaly_svc._has_segmentation_weights
                    else "deterministic_fallback"
                )

            if seg_result is not None:
                await _set_progress(
                    db, analysis, 86, "saving_segmentation",
                    started_at, estimated_total, methodology,
                )
                seg_path = Path(settings.ANALYSIS_DIR) / f"{analysis_id}_segmentation.npy"
                np.save(str(seg_path), seg_result["mask"])
                analysis.segmentation_mask_path = str(seg_path)

            analysis.status = "completed"
            if scan:
                scan.status = ScanStatus.ANALYZED
            await _set_progress(
                db, analysis, 100, "completed", started_at,
                max(1, int(time.time() - started_at)), methodology,
            )
            logger.info(f"Analysis {analysis_id} completed for scan {scan_id}")

        except Exception as exc:
            logger.error(f"Analysis failed: {exc}")
            if analysis is not None:
                analysis.status = "failed"
                analysis.reasoning_json = {
                    **(analysis.reasoning_json or {}),
                    "stage": "failed",
                    "error": str(exc),
                    "eta_seconds": 0,
                }
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
        reasoning_json={
            "progress": 0,
            "stage": "queued",
            "estimated_total_seconds": _estimate_duration_seconds(req.analysis_type),
            "eta_seconds": _estimate_duration_seconds(req.analysis_type),
        },
    )
    db.add(analysis)
    await db.flush()

    background_tasks.add_task(_run_analysis, req.scan_id, analysis.id, req.analysis_type)

    return AnalysisResponse(
        analysis_id=analysis.id,
        scan_id=req.scan_id,
        status="pending",
        analysis_type=req.analysis_type,
        progress=0,
        stage="queued",
        eta_seconds=_estimate_duration_seconds(req.analysis_type),
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
        progress_state = a.reasoning_json or {}
        elapsed = progress_state.get("elapsed_seconds")
        eta = progress_state.get("eta_seconds")
        started_at = progress_state.get("started_at")
        if a.status == "running" and started_at:
            try:
                started = datetime.datetime.fromisoformat(started_at)
                elapsed = max(
                    0.0,
                    (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds(),
                )
                estimated = progress_state.get("estimated_total_seconds", 0)
                eta = max(0, int(estimated - elapsed))
            except (TypeError, ValueError):
                pass
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
            progress=float(progress_state.get("progress", 100 if a.status == "completed" else 0)),
            stage=progress_state.get("stage", a.status),
            eta_seconds=eta,
            elapsed_seconds=elapsed,
            methodology=progress_state.get("methodology"),
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
