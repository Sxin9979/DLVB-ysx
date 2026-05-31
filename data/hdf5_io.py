"""Optional HDF5 helpers for packed single-molecule chunk storage."""

from __future__ import annotations

import pickle
from pathlib import Path
from urllib.parse import quote

import numpy as np

from data.schema import PackedMoleculeChunk

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None


PACKED_CHUNK_ARRAY_FIELDS = (
    "atom_numbers",
    "atom_positions",
    "local_frame_e1",
    "local_frame_e2",
    "local_frame_e3",
    "atom_node_features",
    "atom_senders",
    "atom_receivers",
    "atom_pair_features",
    "lap_evals",
    "lap_evecs",
    "atom_n_node",
    "atom_n_edge",
    "orbital_atom_index",
    "orbital_role",
    "active_slot_index",
    "rumer_senders",
    "rumer_receivers",
    "rumer_edge_type",
    "rumer_n_node",
    "rumer_n_edge",
    "active_orbital_index",
    "active_rumer_senders",
    "active_rumer_receivers",
    "active_rumer_edge_type",
    "active_rumer_n_node",
    "active_rumer_n_edge",
    "targets",
    "top_mass_focus_mask",
)


def require_h5py():
    """Return the optional h5py module or raise one actionable error."""

    if h5py is None:
        raise ImportError(
            "HDF5 support requires the optional dependency 'h5py'. "
            "Install h5py in the training environment before using a .h5 packed cache."
        )
    return h5py


def isHdf5Path(path: str | Path | None) -> bool:
    """Return whether one configured cache path should use HDF5 storage."""

    if path is None:
        return False
    suffix = str(path).lower()
    return suffix.endswith(".h5") or suffix.endswith(".hdf5")


def safeHdf5Name(value: str) -> str:
    """Return one deterministic HDF5-safe path component."""

    return quote(str(value), safe="")


def chunkDatasetPath(
    split_name: str,
    molecule_id: str,
    chunk_index: int,
    view_name: str = "full",
) -> str:
    """Build one deterministic group path for a packed chunk inside HDF5."""

    safe_molecule_id = safeHdf5Name(molecule_id)
    safe_view_name = safeHdf5Name(view_name)
    return f"/packed_chunks/{safe_view_name}/{split_name}/{safe_molecule_id}/chunk_{int(chunk_index):04d}"


def _write_array(group, name: str, value):
    """Write one array dataset, preserving optional None fields."""

    if value is None:
        group.attrs[f"{name}__is_none"] = True
        return
    array = np.asarray(value)
    if name in group:
        del group[name]
    group.create_dataset(name, data=array)


def writePackedChunk(path: str | Path, dataset_path: str, chunk: PackedMoleculeChunk) -> None:
    """Write one packed chunk into one HDF5 file at the given group path."""

    h5py_module = require_h5py()
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py_module.File(resolved_path, "a") as handle:
        if dataset_path in handle:
            del handle[dataset_path]
        group = handle.create_group(dataset_path)
        group.attrs["molecule_id"] = str(chunk.molecule_id)
        group.attrs["chunk_index"] = int(chunk.chunk_index)
        group.attrs["num_chunks"] = int(chunk.num_chunks)
        group.attrs["dataset_id"] = str(getattr(chunk, "dataset_id", "default"))
        for field_name in PACKED_CHUNK_ARRAY_FIELDS:
            _write_array(group, field_name, getattr(chunk, field_name, None))


def _read_array(group, name: str):
    """Read one array dataset, honoring optional None markers."""

    if bool(group.attrs.get(f"{name}__is_none", False)):
        return None
    return np.asarray(group[name])


def readPackedChunk(path: str | Path, dataset_path: str) -> PackedMoleculeChunk:
    """Load one packed chunk from one HDF5 file group."""

    h5py_module = require_h5py()
    with h5py_module.File(path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"Packed HDF5 chunk group was not found: {dataset_path}")
        group = handle[dataset_path]
        payload = {
            "molecule_id": str(group.attrs["molecule_id"]),
            "chunk_index": int(group.attrs["chunk_index"]),
            "num_chunks": int(group.attrs["num_chunks"]),
            "dataset_id": str(group.attrs.get("dataset_id", "default")),
        }
        for field_name in PACKED_CHUNK_ARRAY_FIELDS:
            payload[field_name] = _read_array(group, field_name)
    return PackedMoleculeChunk(**payload)


def writePackedCacheMetadata(path: str | Path, payload) -> None:
    """Persist one PackedDatasetCache payload as a pickled blob inside HDF5."""

    h5py_module = require_h5py()
    encoded = np.frombuffer(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8)
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py_module.File(resolved_path, "a") as handle:
        if "/metadata" not in handle:
            handle.create_group("/metadata")
        if "/metadata/packed_cache_pickle" in handle:
            del handle["/metadata/packed_cache_pickle"]
        handle.create_dataset("/metadata/packed_cache_pickle", data=encoded)


def readPackedCacheMetadata(path: str | Path):
    """Load one PackedDatasetCache payload from one HDF5 metadata blob."""

    h5py_module = require_h5py()
    with h5py_module.File(path, "r") as handle:
        if "/metadata/packed_cache_pickle" not in handle:
            raise KeyError(f"HDF5 packed cache metadata is missing in {path}.")
        encoded = np.asarray(handle["/metadata/packed_cache_pickle"], dtype=np.uint8)
    return pickle.loads(encoded.tobytes())


def clearPackedChunkStore(path: str | Path) -> None:
    """Drop any existing packed chunk groups before a rebuild."""

    h5py_module = require_h5py()
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py_module.File(resolved_path, "a") as handle:
        if "/packed_chunks" in handle:
            del handle["/packed_chunks"]
