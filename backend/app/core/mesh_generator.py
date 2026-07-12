"""
MedSpatial AI — Mesh Generator
Converts numpy vertex/face arrays to GLB (glTF Binary) format for Three.js consumption.
"""

import struct
import json
import numpy as np
from loguru import logger


class MeshGenerator:
    """Generates GLB (glTF Binary) mesh files from vertex/face data."""

    def save_glb(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray,
        output_path: str,
        vertex_colors: np.ndarray = None,
    ) -> None:
        """
        Save mesh as GLB (binary glTF 2.0) file.

        Args:
            vertices: (N, 3) float32 array of vertex positions
            faces: (M, 3) int32 array of triangle face indices
            normals: (N, 3) float32 array of vertex normals
            output_path: path to save .glb file
            vertex_colors: optional (N, 4) float32 array of RGBA colors
        """
        vertices = vertices.astype(np.float32)
        normals = normals.astype(np.float32)
        faces = faces.astype(np.uint32)

        # Normalize normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        normals = normals / norms

        # ── Build binary buffer ───────────────────────────────
        position_data = vertices.tobytes()
        normal_data = normals.tobytes()
        index_data = faces.flatten().astype(np.uint32).tobytes()

        buffer_parts = [position_data, normal_data, index_data]
        buffer_offsets = []
        offset = 0
        for part in buffer_parts:
            buffer_offsets.append(offset)
            offset += len(part)

        color_data = b""
        if vertex_colors is not None:
            vertex_colors = vertex_colors.astype(np.float32)
            if vertex_colors.shape[1] == 3:
                alpha = np.ones((vertex_colors.shape[0], 1), dtype=np.float32)
                vertex_colors = np.hstack([vertex_colors, alpha])
            color_data = vertex_colors.tobytes()
            buffer_parts.append(color_data)
            buffer_offsets.append(offset)
            offset += len(color_data)

        binary_buffer = b"".join(buffer_parts)

        # Pad to 4-byte alignment
        padding = (4 - len(binary_buffer) % 4) % 4
        binary_buffer += b"\x00" * padding

        # ── Build glTF JSON ───────────────────────────────────
        v_min = vertices.min(axis=0).tolist()
        v_max = vertices.max(axis=0).tolist()

        accessors = [
            {  # 0: positions
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "min": v_min,
                "max": v_max,
            },
            {  # 1: normals
                "bufferView": 1,
                "componentType": 5126,
                "count": len(normals),
                "type": "VEC3",
            },
            {  # 2: indices
                "bufferView": 2,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(faces) * 3,
                "type": "SCALAR",
            },
        ]

        buffer_views = [
            {  # 0: positions
                "buffer": 0,
                "byteOffset": buffer_offsets[0],
                "byteLength": len(position_data),
                "target": 34962,  # ARRAY_BUFFER
            },
            {  # 1: normals
                "buffer": 0,
                "byteOffset": buffer_offsets[1],
                "byteLength": len(normal_data),
                "target": 34962,
            },
            {  # 2: indices
                "buffer": 0,
                "byteOffset": buffer_offsets[2],
                "byteLength": len(index_data),
                "target": 34963,  # ELEMENT_ARRAY_BUFFER
            },
        ]

        attributes = {
            "POSITION": 0,
            "NORMAL": 1,
        }

        if color_data:
            buffer_views.append({
                "buffer": 0,
                "byteOffset": buffer_offsets[3],
                "byteLength": len(color_data),
                "target": 34962,
            })
            accessors.append({
                "bufferView": 3,
                "componentType": 5126,
                "count": len(vertex_colors),
                "type": "VEC4",
            })
            attributes["COLOR_0"] = 3

        gltf = {
            "asset": {"version": "2.0", "generator": "MedSpatial AI"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": attributes,
                            "indices": 2,
                            "material": 0,
                        }
                    ]
                }
            ],
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.8, 0.85, 0.9, 1.0],
                        "metallicFactor": 0.1,
                        "roughnessFactor": 0.7,
                    },
                    "doubleSided": True,
                }
            ],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(binary_buffer)}],
        }

        gltf_json = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        # Pad JSON to 4-byte alignment
        json_padding = (4 - len(gltf_json) % 4) % 4
        gltf_json += b" " * json_padding

        # ── Assemble GLB ──────────────────────────────────────
        # GLB Header: magic(4) + version(4) + length(4)
        # Chunk 0 (JSON): length(4) + type(4) + data
        # Chunk 1 (BIN):  length(4) + type(4) + data
        total_length = (
            12  # header
            + 8 + len(gltf_json)  # JSON chunk
            + 8 + len(binary_buffer)  # BIN chunk
        )

        with open(output_path, "wb") as f:
            # Header
            f.write(struct.pack("<I", 0x46546C67))  # magic: "glTF"
            f.write(struct.pack("<I", 2))  # version
            f.write(struct.pack("<I", total_length))

            # JSON chunk
            f.write(struct.pack("<I", len(gltf_json)))
            f.write(struct.pack("<I", 0x4E4F534A))  # "JSON"
            f.write(gltf_json)

            # BIN chunk
            f.write(struct.pack("<I", len(binary_buffer)))
            f.write(struct.pack("<I", 0x004E4942))  # "BIN\0"
            f.write(binary_buffer)

        logger.info(
            f"GLB saved: {output_path} ({len(vertices)} verts, {len(faces)} faces, "
            f"{total_length / 1024:.1f} KB)"
        )

    def create_colored_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray,
        scalar_values: np.ndarray,
        colormap: str = "hot",
        output_path: str = "",
    ) -> None:
        """
        Create a mesh with per-vertex colors based on scalar values (e.g., anomaly scores).

        Args:
            vertices, faces, normals: mesh geometry
            scalar_values: (N,) array of scalar values per vertex
            colormap: matplotlib colormap name
            output_path: path to save .glb
        """
        # Normalize scalars to [0, 1]
        s_min, s_max = scalar_values.min(), scalar_values.max()
        if s_max - s_min < 1e-8:
            normalized = np.zeros_like(scalar_values)
        else:
            normalized = (scalar_values - s_min) / (s_max - s_min)

        # Generate colors using a simple hot colormap
        colors = np.zeros((len(normalized), 4), dtype=np.float32)
        colors[:, 0] = np.clip(normalized * 3.0, 0, 1)  # R
        colors[:, 1] = np.clip(normalized * 3.0 - 1.0, 0, 1)  # G
        colors[:, 2] = np.clip(normalized * 3.0 - 2.0, 0, 1)  # B
        colors[:, 3] = 1.0  # A

        self.save_glb(vertices, faces, normals, output_path, vertex_colors=colors)
