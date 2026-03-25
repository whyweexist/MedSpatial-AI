"""
MedSpatial AI — 3D Reconstruction API
Endpoints for converting DICOM volumes into 3D meshes with layer separation.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
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
    """Background task: build volume + meshes."""
    from app.models.database import async_session_factory

    async with async_session_factory() as db:
        try:
            result = await db.execute(select(Scan).where(Scan.id == scan_id))
            scan = result.scalar_one_or_none()
            if not scan:
                return

            scan.status = ScanStatus.PROCESSING
            await db.commit()

            # 1. Build 3D volume from DICOM slices
            volume_array, voxel_spacing = recon_svc.build_volume(scan.upload_path)
            volume_path = Path(settings.VOLUME_DIR) / f"{scan_id}.npy"
            volume_path.parent.mkdir(parents=True, exist_ok=True)
            recon_svc.save_volume(volume_array, str(volume_path))

            # 2. Generate primary mesh via marching cubes
            mesh_path = Path(settings.MESH_DIR) / f"{scan_id}.glb"
            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            recon_svc.generate_mesh(
                volume_array, str(mesh_path),
                iso_level=iso_level,
                step_size=step_size,
                voxel_spacing=voxel_spacing,
            )

            # 3. Generate layer meshes
            layer_paths = {}
            if generate_layers:
                layer_paths = recon_svc.generate_layer_meshes(
                    volume_array, scan_id, str(Path(settings.MESH_DIR)),
                    voxel_spacing=voxel_spacing,
                )

            # 4. Save volume record
            vol = Volume(
                id=str(uuid.uuid4()),
                scan_id=scan_id,
                volume_path=str(volume_path),
                mesh_path=str(mesh_path),
                dimensions={
                    "x": int(volume_array.shape[0]),
                    "y": int(volume_array.shape[1]),
                    "z": int(volume_array.shape[2]),
                },
                voxel_spacing={
                    "x": float(voxel_spacing[0]),
                    "y": float(voxel_spacing[1]),
                    "z": float(voxel_spacing[2]),
                },
                hu_min=float(volume_array.min()),
                hu_max=float(volume_array.max()),
                bone_mesh_path=layer_paths.get("bone"),
                soft_tissue_mesh_path=layer_paths.get("soft_tissue"),
                air_mesh_path=layer_paths.get("air"),
                vessel_mesh_path=layer_paths.get("vessel"),
            )
            db.add(vol)
            scan.status = ScanStatus.RECONSTRUCTED
            await db.commit()
            logger.info(f"Reconstruction complete for scan {scan_id}")

        except Exception as exc:
            logger.error(f"Reconstruction failed for {scan_id}: {exc}")
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
        message="Reconstruction started in background.",
    )


@router.get("/status/{scan_id}", response_model=ReconstructionResponse)
async def get_reconstruction_status(scan_id: str, db: AsyncSession = Depends(get_db)):
    """Check reconstruction status and get mesh URLs."""
    result = await db.execute(select(Scan).where(Scan.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()

    layer_urls = None
    if volume:
        layer_urls = {}
        if volume.bone_mesh_path:
            layer_urls["bone"] = f"/api/reconstruction/mesh/{scan_id}/bone"
        if volume.soft_tissue_mesh_path:
            layer_urls["soft_tissue"] = f"/api/reconstruction/mesh/{scan_id}/soft_tissue"
        if volume.air_mesh_path:
            layer_urls["air"] = f"/api/reconstruction/mesh/{scan_id}/air"
        if volume.vessel_mesh_path:
            layer_urls["vessel"] = f"/api/reconstruction/mesh/{scan_id}/vessel"

    return ReconstructionResponse(
        scan_id=scan_id,
        volume_id=volume.id if volume else "pending",
        status=scan.status.value,
        mesh_url=f"/api/reconstruction/mesh/{scan_id}/primary" if volume else None,
        layer_urls=layer_urls if layer_urls else None,
        dimensions=volume.dimensions if volume else None,
    )


@router.get("/mesh/{scan_id}/{layer}")
async def get_mesh_file(scan_id: str, layer: str, db: AsyncSession = Depends(get_db)):
    """Serve a generated GLB mesh file."""
    from fastapi.responses import FileResponse

    result = await db.execute(select(Volume).where(Volume.scan_id == scan_id))
    volume = result.scalar_one_or_none()
    if not volume:
        raise HTTPException(status_code=404, detail="Volume not found.")

    path_map = {
        "primary": volume.mesh_path,
        "bone": volume.bone_mesh_path,
        "soft_tissue": volume.soft_tissue_mesh_path,
        "air": volume.air_mesh_path,
        "vessel": volume.vessel_mesh_path,
    }
    mesh_path = path_map.get(layer)
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

    slice_data = recon_svc.extract_slice(volume.volume_path, req.axis, req.index)
    return slice_data
