from pathlib import Path

from app.services.dicom_service import DicomService
from app.training.datasets.base import DatasetAdapter, TrainingSample


class DicomFolderDataset(DatasetAdapter):
    def __init__(self, root: str) -> None:
        self.studies = sorted(path for path in Path(root).iterdir() if path.is_dir())
        self.service = DicomService()

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, index: int) -> TrainingSample:
        study = self.studies[index]
        volume, spacing = self.service.load_dicom_series(str(study))
        return TrainingSample(
            volume=volume,
            metadata={"spacing": spacing.tolist()},
            provenance={"source": str(study), "authorization_required": True},
        )
