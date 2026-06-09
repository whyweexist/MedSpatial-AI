"""User-in-the-loop mask refinement."""

import numpy as np
from scipy import ndimage


def refine_mask(mask: np.ndarray, additions: np.ndarray | None = None, removals: np.ndarray | None = None) -> np.ndarray:
    refined = mask.astype(bool)
    if additions is not None:
        refined |= additions.astype(bool)
    if removals is not None:
        refined &= ~removals.astype(bool)
    return ndimage.binary_closing(refined, iterations=1).astype(np.uint8)
