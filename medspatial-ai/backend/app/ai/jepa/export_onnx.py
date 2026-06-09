"""ONNX export utility for the lightweight context encoder."""

from pathlib import Path

import torch

from app.ai.jepa.context_encoder import ContextEncoder3D


def export_context_encoder(path: str | Path, volume_size: int = 64) -> None:
    model = ContextEncoder3D().eval()
    sample = torch.zeros(1, 1, volume_size, volume_size, volume_size)
    torch.onnx.export(
        model, sample, str(path), input_names=["volume"],
        output_names=["global_token", "tokens"], opset_version=17,
        dynamic_axes={"volume": {2: "depth", 3: "height", 4: "width"}, "tokens": {1: "tokens"}},
    )
