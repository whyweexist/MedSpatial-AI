"""
MedSpatial AI — 3D Reconstruction API (Enhanced)
Endpoints for DICOM → 3D meshes with tissue-specific layer separation,
body region detection, segments/dissection data, and anatomy labels.
"""

import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Scan, ScanStatus, Volume
from app.models.database import get_db
from app.schemas import ReconstructionRequest, ReconstructionResponse, SliceRequest, SliceResponse
from app.services.reconstruction_service import ReconstructionService

router = APIRouter(prefix="/api/reconstruction", tags=["Reconstruction"])
recon_svc = ReconstructionService()


async def _run_reconstruction(scan_id: str, generate_layers: bool, iso_level: float, step_size: int):
    """Background task: build volume + meshes using the enhanced pipeline."""
    from app.models.database import async_session_factory
    from app.services.dicom_service import DicomService

    dicom_svc = DicomService()

    async with async_session_factory() as db:
        try:
            result = await db.execute(select(Scan).where(Scan.id == scan_id))
            scan = result.scalar_one_or_none()
            if not scan:
                return

            scan.status = ScanStatus.PROCESSING
            await db.commit()

            # 1. Load DICOM volume
            volume_data = dicom_svc.load_dicom_series(scan.upload_path)

            if isinstance(volume_data, dict):
                volume_array = volume_data["volume"]
                voxel_spacing = np.array(volume_data.get("voxel_spacing", [1.0, 1.0, 1.0]))
                metadata = volume_data.get("metadata", {})
            elif isinstance(volume_data, tuple):
                if len(volume_data) == 2:
                    volume_array, voxel_spacing = volume_data
                    metadata = {}
                elif len(volume_data) == 3:
                    volume_array, voxel_spacing, metadata = volume_data
                else:
                    raise ValueError("load_dicom_series returned an unsupported tuple shape")
            else:
                raise TypeError("Expected load_dicom_series to return dict or tuple")

            # Add scan fields to metadata
            metadata["modality"] = scan.modality or metadata.get("modality", "CT")
            metadata["body_part"] = scan.body_part or metadata.get("body_part", "")
            metadata["study_description"] = scan.study_description or metadata.get("study_description", "")
            metadata["series_description"] = scan.series_description or metadata.get("series_description", "")

            # Check if single image (X-ray)
            is_xray = volume_array.ndim == 2 or (volume_array.ndim == 3 and volume_array.shape[0] == 1)

            if is_xray:
                img = volume_array.squeeze() if volume_array.ndim == 3 else volume_array
                recon_result = await recon_svc.reconstruct_from_xray(
                    scan_id=scan_id,
                    image_array=img,
                    metadata=metadata,
                )
            else:
                recon_result = await recon_svc.build_reconstruction(
                    scan_id=scan_id,
                    volume=volume_array,
                    voxel_spacing=voxel_spacing,
                    metadata=metadata,
                    iso_level=iso_level,
                    step_size=step_size,
                    generate_layers=generate_layers,
                )

            # 2. Save volume record
            vol = Volume(
                id=str(uuid.uuid4()),
                scan_id=scan_id,
                volume_path=recon_result["volume_path"],
                mesh_path=recon_result.get("primary_mesh_path"),
                dimensions=recon_result["volume_dimensions"],
                voxel_spacing=recon_result["voxel_spacing"],
                hu_min=recon_result["hu_range"]["min"],
                hu_max=recon_result["hu_range"]["max"],
                layer_mesh_paths=recon_result.get("layer_mesh_paths"),
                reconstruction_summary=recon_result.get("summary"),
            )

            # Also set legacy columns if available
            layer_paths = recon_result.get("layer_mesh_paths", {})
            if "bone" in layer_paths:
                vol.bone_mesh_path = layer_paths["bone"].get("mesh_path")
            if "soft_tissue" in layer_paths:
                vol.soft_tissue_mesh_path = layer_paths["soft_tissue"].get("mesh_path")

            db.add(vol)

            # Update scan with body region
            body_region = recon_result.get("body_region", {})
            scan.body_region = body_region.get("region", "chest")
            scan.region_confidence = body_region.get("confidence", 0.0)
            scan.status = ScanStatus.RECONSTRUCTED
            await db.commit()
            logger.info(f"Reconstruction complete for scan {scan_id}")

        except Exception as exc:
            logger.error(f"Reconstruction failed for {scan_id}: {exc}")
            import traceback
            traceback.print_exc()
            scan.status = ScanStatus.FAILED
            await db.commit()


@router.post("/build", response_model=ReconstructionResponse)
async def start_reconstruction(
    req: ReconstructionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger 3D reconstruction from uploaded DICOM slices."""
    result = await db.execute(select(Scan).where(Scan.id == req.scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    if scan.status not in (ScanStatus.UPLOADED, ScanStatus.RECONSTRUCTED):
        raise HTTPException(status_code=409, detail=f"Scan is {scan.status.value}, cannot reconstruct.")

    iso_level = req.iso_level if req.iso_level is not None else 300.0
    step_size = req.step_size if req.step_size is not None else settings.MARCHING_CUBES_STEP_SIZE

    background_tasks.add_task(
        _run_reconstruction, scan.id, req.generate_layers, iso_level, step_size
    )

    return ReconstructionResponse(
        scan_id=scan.id,
        volume_id="pending",
        status="processing",
        mesh_url=None,
        layer_urls=None,
        dimensions=None,
    )


@router.get("/status/{scan_id}")
async def get_reconstruction_status(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Check reconstruction status and get mesh URLs + summary."""
    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()

    layer_urls = {}
    body_region = None
    summary = None
    labels = []

    if volume:
        # Build layer URLs from new JSON field
        if volume.layer_mesh_paths:
            for tissue_name, info in volume.layer_mesh_paths.items():
                if isinstance(info, dict) and info.get("mesh_path"):
                    layer_urls[tissue_name] = f"/api/reconstruction/mesh/{scan_id}/{tissue_name}"

        # Legacy fallback
        if not layer_urls:
            if volume.bone_mesh_path:
                layer_urls["bone"] = f"/api/reconstruction/mesh/{scan_id}/bone"
            if volume.soft_tissue_mesh_path:
                layer_urls["soft_tissue"] = f"/api/reconstruction/mesh/{scan_id}/soft_tissue"
            if volume.air_mesh_path:
                layer_urls["air"] = f"/api/reconstruction/mesh/{scan_id}/air"
            if volume.vessel_mesh_path:
                layer_urls["vessel"] = f"/api/reconstruction/mesh/{scan_id}/vessel"

        # Extract summary
        if volume.reconstruction_summary:
            summary = volume.reconstruction_summary
            body_region = summary.get("body_region")
            labels = summary.get("labels", [])

    return JSONResponse({
        "scan_id": scan_id,
        "volume_id": volume.id if volume else "pending",
        "status": scan.status.value,
        "mesh_url": f"/api/reconstruction/mesh/{scan_id}/primary" if volume and volume.mesh_path else None,
        "layer_urls": layer_urls if layer_urls else None,
        "dimensions": volume.dimensions if volume else None,
        "body_region": body_region,
        "summary": summary,
        "labels": labels,
    })


@router.get("/segments/{scan_id}")
async def get_segments(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Get all segment data for the dissection module."""
    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()
    if not volume:
        raise HTTPException(status_code=404, detail="Volume not reconstructed.")

    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()

    segments = []
    body_region = None

    if volume.reconstruction_summary:
        summary = volume.reconstruction_summary
        body_region = summary.get("body_region")

        for tissue in summary.get("tissues", []):
            if not tissue.get("has_mesh", False):
                continue

            r, g, b = tissue.get("color_rgb", [0.5, 0.5, 0.5])
            color_hex = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

            centroid = None
            if tissue.get("centroid_mm"):
                cm = tissue["centroid_mm"]
                centroid = {"x": cm[2] if len(cm) > 2 else 0, "y": cm[1] if len(cm) > 1 else 0, "z": cm[0]}

            segments.append({
                "name": tissue["name"],
                "label_index": tissue.get("label_index", 0),
                "mesh_url": f"/api/reconstruction/mesh/{scan_id}/{tissue['name']}",
                "visible": True,
                "opacity": tissue.get("opacity", 0.8),
                "color": color_hex,
                "color_rgb": tissue.get("color_rgb", [0.5, 0.5, 0.5]),
                "volume_cm3": tissue.get("volume_cm3", tissue.get("volume_mm3", 0) / 1000.0),
                "mean_hu": tissue.get("mean_hu", 0),
                "voxel_count": tissue.get("voxel_count", 0),
                "centroid": centroid,
                "description": tissue.get("description", ""),
                "dissection_order": tissue.get("dissection_order", 5),
            })

    return JSONResponse({
        "scan_id": scan_id,
        "segments": segments,
        "body_region": body_region,
    })


@router.get("/labels/{scan_id}")
async def get_anatomy_labels(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Get anatomy labels for the 3D viewer."""
    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()
    if not volume:
        raise HTTPException(status_code=404, detail="Volume not reconstructed.")

    labels = []
    if volume.reconstruction_summary:
        labels = volume.reconstruction_summary.get("labels", [])

    return JSONResponse({"scan_id": scan_id, "labels": labels})


@router.get("/mesh/{scan_id}/{layer}")
async def get_mesh_file(scan_id: str, layer: str, db: AsyncSession = Depends(get_db)):
    """Serve a generated GLB mesh file."""
    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()
    if not volume:
        raise HTTPException(status_code=404, detail="Volume not found.")

    mesh_path = None

    # Check primary
    if layer == "primary":
        mesh_path = volume.mesh_path

    # Check new layer_mesh_paths JSON
    elif volume.layer_mesh_paths and layer in volume.layer_mesh_paths:
        info = volume.layer_mesh_paths[layer]
        if isinstance(info, dict):
            mesh_path = info.get("mesh_path")
        elif isinstance(info, str):
            mesh_path = info

    # Legacy fallback
    elif layer == "bone":
        mesh_path = volume.bone_mesh_path
    elif layer == "soft_tissue":
        mesh_path = volume.soft_tissue_mesh_path
    elif layer == "air":
        mesh_path = volume.air_mesh_path
    elif layer == "vessel":
        mesh_path = volume.vessel_mesh_path

    if not mesh_path or not Path(mesh_path).exists():
        raise HTTPException(status_code=404, detail=f"Mesh '{layer}' not found.")

    return FileResponse(mesh_path, media_type="model/gltf-binary", filename=f"{scan_id}_{layer}.glb")


@router.post("/slice", response_model=SliceResponse)
async def get_slice(req: SliceRequest, db: AsyncSession = Depends(get_db)):
    """Get a 2D slice from the reconstructed volume."""
    result = await db.execute(select(Volume).where(Volume.scan_id == req.scan_id))
    volume = result.scalar_one_or_none()
    if not volume:
        raise HTTPException(status_code=404, detail="Volume not reconstructed yet.")

    import base64
    from io import BytesIO
    from PIL import Image

    vol_data = np.load(volume.volume_path)
    slice_2d = recon_svc.extract_slice(vol_data, req.axis, req.index)

    # Normalize to 0-255
    windowed = np.clip(slice_2d, -200, 400)
    normalized = ((windowed - windowed.min()) / (windowed.max() - windowed.min() + 1e-8) * 255).astype(np.uint8)

    img = Image.fromarray(normalized, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    total_slices = {
        "axial": vol_data.shape[0],
        "coronal": vol_data.shape[1],
        "sagittal": vol_data.shape[2],
    }.get(req.axis, vol_data.shape[0])

    return SliceResponse(
        image_data=img_b64,
        axis=req.axis,
        index=req.index,
        total_slices=total_slices,
    )
