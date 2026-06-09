from pathlib import Path

import numpy as np

from app.training.datasets.base import DatasetAdapter, TrainingSample


class NiftiFolderDataset(DatasetAdapter):
    def __init__(self, root: str) -> None:
        self.files = sorted(Path(root).glob("*.nii*"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> TrainingSample:
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError("Install nibabel to use the NIfTI training adapter") from exc
        path = self.files[index]
        image = nib.load(str(path))
        return TrainingSample(
            volume=np.asarray(image.get_fdata(), dtype=np.float32),
            metadata={"affine": image.affine.tolist()},
            provenance={"source": str(path), "authorization_required": True},
        )
