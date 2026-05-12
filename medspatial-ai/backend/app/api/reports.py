"""
MedSpatial AI — Reports API
Endpoints for generating downloadable PDF and DOCX analysis reports.
"""

from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Analysis, Scan, Volume
from app.models.database import get_db
from app.services.report_service import ReportService

router = APIRouter(prefix="/api/reports", tags=["Reports"])
report_svc = ReportService()


@router.get("/generate/{scan_id}")
async def generate_report(
    scan_id: str,
    format: str = Query("pdf", regex="^(pdf|docx)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate and download a comprehensive analysis report.
    Supports PDF and DOCX formats.
    """
    # Load scan
    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Load volume
    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()

    # Load analysis
    result = await db.execute(
        select(Analysis).where(Analysis.scan_id == scan_id)
        .order_by(Analysis.created_at.desc())
    )
    analysis = result.scalar_one_or_none()

    # Prepare data
    scan_metadata = {
        "modality": scan.modality or "Unknown",
        "body_part": scan.body_part or "Unknown",
        "body_region": getattr(scan, "body_region", scan.body_part) or "Unknown",
        "study_description": scan.study_description or "N/A",
        "series_description": scan.series_description or "N/A",
        "num_slices": scan.num_slices,
        "pixel_spacing_x": scan.pixel_spacing_x,
        "pixel_spacing_y": scan.pixel_spacing_y,
        "slice_thickness": scan.slice_thickness,
        "patient_id": "ANONYMIZED",
    }

    # Load findings
    findings = []
    if analysis and analysis.findings:
        raw = analysis.findings
        if isinstance(raw, dict):
            findings = raw.get("anomalies", [])
        elif isinstance(raw, list):
            findings = raw

    # Load volume data for slice rendering
    vol_data = None
    if volume and volume.volume_path:
        try:
            vol_data = np.load(volume.volume_path)
        except Exception:
            pass

    # Reconstruction summary
    reconstruction_summary = None
    if volume:
        layer_data = getattr(volume, "layer_mesh_paths", None)
        if layer_data and isinstance(layer_data, dict):
            tissues = []
            for name, info in layer_data.items():
                if isinstance(info, dict):
                    tissues.append(info)
                else:
                    tissues.append({"name": name, "volume_mm3": 0, "vertex_count": 0, "mean_hu": 0})
            reconstruction_summary = {"tissues": tissues}

    # Load reasoning chains
    reasoning_chains = None
    xai_reasoning_path = Path(settings.ANALYSIS_DIR) / "xai" / f"{scan_id}_reasoning.json"
    if xai_reasoning_path.exists():
        import json
        with open(xai_reasoning_path) as f:
            reasoning_chains = json.load(f)

    # Generate report
    reports_dir = Path(settings.ANALYSIS_DIR) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        if format == "pdf":
            output_path = str(reports_dir / f"{scan_id}_report.pdf")
            report_svc.generate_pdf(
                scan_metadata=scan_metadata,
                reconstruction_summary=reconstruction_summary,
                findings=findings,
                tissue_results=None,
                reasoning_chains=reasoning_chains,
                volume=vol_data,
                output_path=output_path,
            )
            media_type = "application/pdf"
            filename = f"MedSpatial_Report_{scan_id[:8]}.pdf"
        else:
            output_path = str(reports_dir / f"{scan_id}_report.docx")
            report_svc.generate_docx(
                scan_metadata=scan_metadata,
                reconstruction_summary=reconstruction_summary,
                findings=findings,
                tissue_results=None,
                reasoning_chains=reasoning_chains,
                volume=vol_data,
                output_path=output_path,
            )
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            filename = f"MedSpatial_Report_{scan_id[:8]}.docx"

        return FileResponse(
            output_path,
            media_type=media_type,
            filename=filename,
        )

    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"Report generation dependency missing: {exc}. Install with: pip install reportlab python-docx matplotlib",
        )
    except Exception as exc:
        logger.error(f"Report generation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(exc)}")
