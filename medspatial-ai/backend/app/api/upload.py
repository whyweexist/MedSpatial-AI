"""
MedSpatial AI — DICOM Upload API
Handles multi-file DICOM uploads, metadata extraction, and scan creation.
"""

import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Scan, ScanStatus
from app.models.database import get_db
from app.schemas import ScanListResponse, ScanMetadata, ScanUploadResponse
from app.services.dicom_service import DicomService

router = APIRouter(prefix="/api/scans", tags=["Scans"])
dicom_svc = DicomService()


@router.post("/upload", response_model=ScanUploadResponse)
async def upload_dicom_files(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload one or more DICOM files to create a new scan."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    scan_id = str(uuid.uuid4())
    upload_dir = Path(settings.UPLOAD_DIR) / scan_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    try:
        for f in files:
            dest = upload_dir / f.filename
            with open(dest, "wb") as buf:
                content = await f.read()
                buf.write(content)
            saved_paths.append(dest)
            logger.info(f"Saved DICOM file: {dest}")
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")

    # Parse DICOM metadata from the first file (representative)
    try:
        meta = dicom_svc.extract_metadata(saved_paths)
    except Exception as exc:
        logger.error(f"DICOM parse error: {exc}")
        meta = {}

    scan = Scan(
        id=scan_id,
        patient_id=meta.get("patient_id"),
        patient_name=meta.get("patient_name"),
        study_description=meta.get("study_description"),
        series_description=meta.get("series_description"),
        modality=meta.get("modality"),
        body_part=meta.get("body_part"),
        num_slices=len(saved_paths),
        slice_thickness=meta.get("slice_thickness"),
        pixel_spacing_x=meta.get("pixel_spacing_x"),
        pixel_spacing_y=meta.get("pixel_spacing_y"),
        rows=meta.get("rows"),
        columns=meta.get("columns"),
        upload_path=str(upload_dir),
        status=ScanStatus.UPLOADED,
        metadata_json=meta,
    )
    db.add(scan)
    await db.flush()

    logger.info(f"Created scan {scan_id} with {len(saved_paths)} slices")

    return ScanUploadResponse(
        scan_id=scan_id,
        status=ScanStatus.UPLOADED.value,
        message=f"Successfully uploaded {len(saved_paths)} DICOM files.",
        num_slices=len(saved_paths),
    )


@router.get("/", response_model=ScanListResponse)
async def list_scans(db: AsyncSession = Depends(get_db)):
    """List all uploaded scans."""
    result = await db.execute(select(Scan).order_by(Scan.created_at.desc()))
    scans = result.scalars().all()
    return ScanListResponse(
        scans=[ScanMetadata.model_validate(s) for s in scans],
        total=len(scans),
    )


@router.get("/{scan_id}", response_model=ScanMetadata)
async def get_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Get metadata for a specific scan."""
    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return ScanMetadata.model_validate(scan)


@router.delete("/{scan_id}")
async def delete_scan(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a scan and all associated data."""
    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    # Remove files
    upload_dir = Path(scan.upload_path)
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)

    await db.delete(scan)
    return {"message": f"Scan {scan_id} deleted."}
