"""
MedSpatial AI — Backend Test Suite
Covers: DICOM parser, model forward passes, reconstruction service, and API integration.
"""

import asyncio
import io
import struct
from pathlib import Path

import numpy as np
import pytest
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import torch

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_dicom_bytes():
    """Create a valid minimal DICOM file in memory for testing."""
    ds = Dataset()
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9.0"
    ds.file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian
    ds.file_meta.ImplementationVersionName  = "MEDSPATIAL"

    ds.is_implicit_VR   = False
    ds.is_little_endian = True
    ds.SOPClassUID      = "1.2.840.10008.5.1.4.1.1.2"
    ds.SOPInstanceUID   = "1.2.3.4.5.6.7.8.9.0"
    ds.Modality         = "CT"
    ds.Rows             = 64
    ds.Columns          = 64
    ds.BitsAllocated    = 16
    ds.BitsStored       = 16
    ds.HighBit          = 15
    ds.PixelRepresentation = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope     = 1.0
    ds.RescaleIntercept = -1024.0
    ds.WindowCenter     = 40.0
    ds.WindowWidth      = 400.0
    ds.PixelSpacing     = [1.0, 1.0]
    ds.SliceThickness   = 1.0
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.PatientPosition  = "HFS"
    ds.BodyPartExamined = "CHEST"

    # Synthetic pixel data: simple gradient
    pixels = np.arange(64 * 64, dtype=np.int16).reshape(64, 64)
    pixels -= 1024  # Shift to HU range
    ds.PixelData = pixels.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def random_volume():
    """Small 32³ random volume in HU range."""
    return np.random.uniform(-1000, 400, (32, 32, 32)).astype(np.float32)


@pytest.fixture
def random_tensor_3d():
    """Batch of 1, single-channel, 32³ normalized tensor."""
    return torch.rand(1, 1, 32, 32, 32)


# ─────────────────────────────────────────────────────────────────
# 1. DICOM Parser Tests
# ─────────────────────────────────────────────────────────────────

class TestDicomParser:

    def test_parse_valid_dicom(self, synthetic_dicom_bytes, tmp_path):
        """DICOM parser should extract pixel data and metadata."""
        from app.services.dicom_service import DicomService

        dcm_path = tmp_path / "test.dcm"
        dcm_path.write_bytes(synthetic_dicom_bytes)

        svc = DicomService()
        result = svc.parse_single_file(str(dcm_path))

        assert result is not None
        assert result["modality"] == "CT"
        assert result["rows"] == 64
        assert result["columns"] == 64
        assert result["pixel_array"] is not None
        assert result["pixel_array"].shape == (64, 64)
        print("✅ DICOM parse: passed")

    def test_hu_conversion(self, synthetic_dicom_bytes, tmp_path):
        """HU conversion should apply rescale slope/intercept."""
        from app.services.dicom_service import DicomService

        dcm_path = tmp_path / "test.dcm"
        dcm_path.write_bytes(synthetic_dicom_bytes)

        svc = DicomService()
        result = svc.parse_single_file(str(dcm_path))
        pixels = result["pixel_array"]

        # With RescaleSlope=1, RescaleIntercept=-1024, HU = raw - 1024
        assert pixels.min() >= -1100
        assert pixels.max() <= 1000
        print("✅ HU conversion: passed")

    def test_invalid_dicom_raises(self, tmp_path):
        """Non-DICOM file should raise or return None gracefully."""
        from app.services.dicom_service import DicomService

        bad_file = tmp_path / "notadicom.dcm"
        bad_file.write_bytes(b"this is not a DICOM file at all 12345")

        svc = DicomService()
        try:
            result = svc.parse_single_file(str(bad_file))
            assert result is None or "error" in (result or {})
        except Exception:
            pass  # Acceptable to raise
        print("✅ Invalid DICOM handling: passed")

    def test_load_dicom_series_multipframe_default(self, tmp_path):
        """Multi-frame DICOM must be flattened to a 3D volume by load_dicom_series."""
        from app.services.dicom_service import DicomService

        ds = Dataset()
        ds.file_meta = Dataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9.1"
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.ImplementationVersionName = "MEDSPATIAL"

        ds.is_implicit_VR = False
        ds.is_little_endian = True
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9.1"
        ds.Modality = "CT"
        ds.Rows = 64
        ds.Columns = 64
        ds.NumberOfFrames = 3
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [0.0, 0.0, 0.0]

        frames = np.stack([np.full((64, 64), i, dtype=np.int16) for i in range(3)], axis=0)
        ds.PixelData = frames.tobytes()

        dcm_path = tmp_path / "multi_frame.dcm"
        pydicom.dcmwrite(dcm_path, ds, write_like_original=False)

        svc = DicomService()
        volume, spacing = svc.load_dicom_series(str(tmp_path))

        assert volume.shape == (3, 64, 64)
        assert np.all(volume[0] == (0 - 1024))
        assert np.all(volume[1] == (1 - 1024))
        assert np.all(volume[2] == (2 - 1024))
        print("✅ Multi-frame DICOM flattening: passed")


# ─────────────────────────────────────────────────────────────────
# 2. AI Model Forward Pass Tests
# ─────────────────────────────────────────────────────────────────

class TestModelForwardPasses:

    def test_spatial_transformer_output_shape(self, random_tensor_3d):
        """SpatialTransformer3D should return cls token and spatial features."""
        from app.ai.spatial_transformer import create_spatial_transformer

        model = create_spatial_transformer(embed_dim=64, num_heads=4, num_layers=2)
        model.eval()
        with torch.no_grad():
            cls_feat, spatial_feat = model.extract_features(random_tensor_3d)
        assert cls_feat.shape[0] == 1
        assert cls_feat.shape[-1] == 64
        print(f"✅ SpatialTransformer3D: cls={cls_feat.shape}, spatial={spatial_feat.shape}")

    def test_spatial_transformer_position_encoding_with_cls(self):
        """SpatialTransformer3D pos encoding should handle cls token and large patch count."""
        from app.ai.spatial_transformer import create_spatial_transformer

        x = torch.rand(1, 1, 129, 129, 129)
        model = create_spatial_transformer(embed_dim=64, num_heads=4, num_layers=2)
        model.eval()

        with torch.no_grad():
            cls_feat, spatial_feat = model.extract_features(x)

        assert cls_feat.shape == (1, 64)
        assert spatial_feat.shape[1] >= 4096
        assert not torch.isnan(spatial_feat).any()
        print(f"✅ SpatialTransformer3D cls+patch pos encoding works shape {spatial_feat.shape}")

    def test_depth_lifter_output_shape(self):
        """DepthLifter should convert 2D X-ray to pseudo-3D volume."""
        from app.ai.depth_lifter import create_depth_lifter

        model = create_depth_lifter(out_depth=32)
        model.eval()
        xray = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            volume = model(xray)
        assert volume.shape == (1, 1, 32, 64, 64), f"Got {volume.shape}"
        print(f"✅ DepthLifter: output shape {volume.shape}")

    def test_anomaly_graph_output_shape(self, random_tensor_3d):
        """AnomalyGraph should return anomaly map and disease probs."""
        from app.ai.anomaly_graph import create_anomaly_graph

        model = create_anomaly_graph(embed_dim=64, graph_dim=64)
        model.eval()
        # Use smaller 32³ volume → 4³ = 64 patches
        x = torch.rand(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "anomaly_map_3d" in out
        assert out["anomaly_map_3d"].shape == (1, 32, 32, 32)
        assert out["disease_probs"].shape == (1, 13)
        print(f"✅ AnomalyGraph: map={out['anomaly_map_3d'].shape}, diseases={out['disease_probs'].shape}")

    def test_segmentation_net_output_shape(self, random_tensor_3d):
        """SegmentationNet3D should output per-voxel class logits."""
        from app.ai.segmentation_net import create_segmentation_net

        model = create_segmentation_net(num_classes=12)
        model.eval()
        x = torch.rand(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        seg = out["segmentation"]
        assert seg.shape == (1, 12, 32, 32, 32), f"Got {seg.shape}"
        print(f"✅ SegmentationNet3D: {seg.shape}")

    def test_anomaly_detector_output(self, random_tensor_3d):
        """AnomalyDetector3D should return anomaly map and score."""
        from app.ai.anomaly_detector import create_anomaly_detector

        model = create_anomaly_detector(latent_dim=64, input_size=32)
        model.eval()
        x = torch.rand(1, 1, 32, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert "anomaly_map" in out
        assert "anomaly_score" in out
        print(f"✅ AnomalyDetector3D: map={out['anomaly_map'].shape}, score={float(out['anomaly_score']):.3f}")


# ─────────────────────────────────────────────────────────────────
# 3. Atlas Generation Tests
# ─────────────────────────────────────────────────────────────────

class TestAtlasGeneration:

    def test_atlas_output_shapes(self):
        """Atlas generator should produce correct shapes and label ranges."""
        from app.atlas.generate_atlas import generate_atlas

        atlas = generate_atlas(size=32)
        assert atlas["labels"].shape == (32, 32, 32)
        assert atlas["probabilities"].shape == (12, 32, 32, 32)
        assert atlas["volume_hu"].shape == (32, 32, 32)
        
        # Labels should be in [0, 11]
        assert atlas["labels"].min() >= 0
        assert atlas["labels"].max() <= 11

        # Probabilities should sum to ~1 per voxel
        prob_sum = atlas["probabilities"].sum(axis=0)
        assert np.allclose(prob_sum, 1.0, atol=0.01)
        print(f"✅ Atlas generation: {atlas['labels'].shape}, labels {atlas['labels'].min()}-{atlas['labels'].max()}")


# ─────────────────────────────────────────────────────────────────
# 4. Reconstruction Pipeline Tests
# ─────────────────────────────────────────────────────────────────

class TestReconstruction:

    def test_volume_normalization(self, random_volume):
        """Volume processor should normalize to [0, 1]."""
        from app.core.volume_processor import VolumeProcessor

        proc = VolumeProcessor()
        normalized = proc.normalize(random_volume)
        assert normalized.min() >= 0.0 - 1e-6
        assert normalized.max() <= 1.0 + 1e-6
        print("✅ Volume normalization: passed")

    def test_windowing(self, random_volume):
        """Windowing should clip HU values correctly."""
        from app.core.volume_processor import VolumeProcessor

        proc = VolumeProcessor()
        windowed = proc.window_level(random_volume, window_center=40, window_width=400)
        assert windowed.min() >= -200 - 1
        assert windowed.max() <= 200 + 1
        print("✅ Windowing: passed")


# ─────────────────────────────────────────────────────────────────
# 5. Knowledge Base Tests
# ─────────────────────────────────────────────────────────────────

class TestKnowledgeBase:

    def test_build_and_query(self, tmp_path):
        """Knowledge base should build and support FTS5 queries."""
        from app.knowledge.build_kb import build_knowledge_base, MedicalKnowledgeBase

        db_path = str(tmp_path / "test_kb.sqlite")
        build_knowledge_base(db_path)

        kb = MedicalKnowledgeBase(db_path)
        
        results = kb.search_diseases("pneumonia consolidation")
        assert len(results) > 0
        assert any("Pneumonia" in r["name"] for r in results)

        anat = kb.search_anatomy("lung")
        assert len(anat) > 0

        kb.close()
        print(f"✅ Knowledge base: {len(results)} disease hits, {len(anat)} anatomy hits")


# ─────────────────────────────────────────────────────────────────
# 6. Hardware Manager Tests
# ─────────────────────────────────────────────────────────────────

class TestHardwareManager:

    def test_hardware_detection(self):
        """Hardware manager should detect and return a valid profile."""
        from app.core.hardware_manager import get_hardware_manager

        hw = get_hardware_manager()
        profile = hw.profile
        
        assert profile.device in ("cuda", "mps", "cpu")
        assert profile.ram_total_gb > 0
        assert profile.volume_size in (32, 64, 128)
        assert hw.check_memory_pressure() in ("ok", "low", "critical")
        print(f"✅ Hardware: device={profile.device}, volume={profile.volume_size}³")
