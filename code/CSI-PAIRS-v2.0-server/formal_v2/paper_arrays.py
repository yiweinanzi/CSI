"""Read-only NPY storage for an NPZ, with bounded extraction and dtype conversion."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np

from .paper_suite import atomic_json, digest, suite_lock

FLOATS = {"csi_repeat", "csi_clean", "maps", "positions", "radio_config", "bs_pose", "noop_maps", "path_power", "noop_path_power"}
INTS = {"repeat_seeds", "path_ids", "path_surface_ids", "noop_path_ids", "noop_path_surface_ids", "primitive_surface_ids", "world_bits", "primitive_ids", "anchor_bits", "natural_world_index"}


class MappedNPZ:
    def __init__(self, source, cache):
        self.sha256 = digest(source)
        self.root = Path(cache) / self.sha256
        self.manifest = {}
        with suite_lock(self.root):
            receipt = self.root / "arrays.json"
            if receipt.exists():
                self.manifest = json.loads(receipt.read_text(encoding="utf-8"))
                for name, record in self.manifest.items():
                    path = self.root / (name + ".npy")
                    if digest(path) != record["sha256"]:
                        raise ValueError("NPZ array cache changed: " + name)
            else:
                with zipfile.ZipFile(source) as archive:
                    for info in archive.infolist():
                        name = info.filename.removesuffix(".npy")
                        if info.filename != name + ".npy" or Path(name).name != name or "/" in name or "\\" in name or name in self.manifest:
                            raise ValueError("Invalid NPZ member")
                        path = self.root / (name + ".npy")
                        temp = self.root / (name + ".extract.npy")
                        with archive.open(info) as inp, temp.open("wb") as out:
                            shutil.copyfileobj(inp, out, length=1024 * 1024)
                        values = np.load(temp, mmap_mode="r", allow_pickle=False)
                        desired = np.dtype("float64" if name in FLOATS else "int64" if name in INTS else "complex128" if name == "phase_reference_values" else values.dtype)
                        if desired != values.dtype:
                            converted = self.root / (name + ".convert.npy")
                            target = np.lib.format.open_memmap(converted, mode="w+", dtype=desired, shape=values.shape)
                            if values.ndim == 0:
                                target[...] = values
                            else:
                                rows = max(1, (8 * 1024**2) // max(1, int(np.prod(values.shape[1:])) * desired.itemsize))
                                for start in range(0, len(values), rows):
                                    target[start:start + rows] = values[start:start + rows]
                            target.flush()
                            del target, values
                            os.replace(converted, path)
                            temp.unlink()
                        else:
                            del values
                            os.replace(temp, path)
                        self.manifest[name] = {"sha256": digest(path)}
                if digest(source) != self.sha256:
                    raise ValueError("NPZ changed while extracting its arrays")
                atomic_json(receipt, self.manifest)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getitem__(self, name):
        if name not in self.manifest:
            raise KeyError(name)
        return np.load(self.root / (name + ".npy"), mmap_mode="r", allow_pickle=False)


def concatenate_disk(parts, output, *, dtype=np.float32):
    """Concatenate per-scene feature files without a full RAM-sized concatenate."""
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in parts]
    if not arrays or any(a.shape[1:] != arrays[0].shape[1:] for a in arrays):
        raise ValueError("No source features or inconsistent feature dimensions")
    shape = (sum(len(a) for a in arrays), *arrays[0].shape[1:])
    target = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=shape)
    offset = 0
    for array in arrays:
        for start in range(0, len(array), 1024):
            stop = min(start + 1024, len(array))
            target[offset + start:offset + stop] = array[start:stop]
        offset += len(array)
    target.flush()
    del target, arrays
    return np.load(output, mmap_mode="c", allow_pickle=False)
