#!/usr/bin/env python3
"""Filter one packed dataset cache by excluding selected molecule ids.

This helper intentionally uses only the Python standard library so it can run
even on hosts where the full training environment is unavailable.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path


class SafeDummy:
    """Fallback object used while unpickling cache records without project deps."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __setstate__(self, state):
        self.state = state


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that tolerates missing project/runtime dependencies."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return SafeDummy


@dataclass
class PackedChunkReference:
    molecule_id: str
    chunk_index: int
    num_chunks: int
    chunk_path: str
    num_structures: int
    num_focus_structures: int
    static_num_atoms: int
    total_atoms: int
    total_atom_edges: int
    total_orbitals: int
    total_rumer_edges: int
    total_active_orbitals: int
    total_active_rumer_edges: int
    dataset_id: str = "default"
    storage_format: str = "pickle"
    hdf5_dataset_path: str | None = None


@dataclass
class PackedDatasetCache:
    split_ids: dict
    split_structure_counts: dict
    packed_chunks: dict | None = None
    packed_chunk_refs: dict | None = None
    dataset_ids: list | None = None
    split_ids_by_dataset: dict | None = None
    split_structure_counts_by_dataset: dict | None = None


def ensure_pickle_schema() -> tuple[type, type]:
    """Create minimal pickle-visible schema classes under module ``data.schema``."""

    if "data" not in sys.modules:
        sys.modules["data"] = types.ModuleType("data")
    schema_module = sys.modules.get("data.schema")
    if schema_module is None:
        schema_module = types.ModuleType("data.schema")
        sys.modules["data.schema"] = schema_module
        setattr(sys.modules["data"], "schema", schema_module)

    PackedChunkReference.__module__ = "data.schema"
    PackedDatasetCache.__module__ = "data.schema"
    schema_module.PackedChunkReference = PackedChunkReference
    schema_module.PackedDatasetCache = PackedDatasetCache
    return PackedDatasetCache, PackedChunkReference


def object_state(value):
    """Return a dict-like state payload from a dataclass or SafeDummy object."""

    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        if "state" in value.__dict__ and isinstance(value.__dict__["state"], dict):
            return value.__dict__["state"]
        return value.__dict__
    raise TypeError(f"Unsupported cache object type: {type(value)!r}")


def load_cache(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = SafeUnpickler(handle).load()
    state = object_state(payload)
    if not isinstance(state, dict):
        raise TypeError("Packed cache top-level state is not a dictionary.")
    return state


def to_reference(reference_state: dict, reference_cls: type):
    return reference_cls(
        molecule_id=str(reference_state["molecule_id"]),
        chunk_index=int(reference_state["chunk_index"]),
        num_chunks=int(reference_state["num_chunks"]),
        chunk_path=str(reference_state["chunk_path"]),
        num_structures=int(reference_state["num_structures"]),
        num_focus_structures=int(reference_state["num_focus_structures"]),
        static_num_atoms=int(reference_state["static_num_atoms"]),
        total_atoms=int(reference_state["total_atoms"]),
        total_atom_edges=int(reference_state["total_atom_edges"]),
        total_orbitals=int(reference_state["total_orbitals"]),
        total_rumer_edges=int(reference_state["total_rumer_edges"]),
        total_active_orbitals=int(reference_state["total_active_orbitals"]),
        total_active_rumer_edges=int(reference_state["total_active_rumer_edges"]),
        dataset_id=str(reference_state.get("dataset_id", "default")),
        storage_format=str(reference_state.get("storage_format", "pickle")),
        hdf5_dataset_path=(
            None
            if reference_state.get("hdf5_dataset_path") is None
            else str(reference_state.get("hdf5_dataset_path"))
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source packed cache pickle.")
    parser.add_argument("--output", required=True, help="Filtered packed cache pickle.")
    parser.add_argument(
        "--exclude",
        nargs="+",
        required=True,
        help="One or more full cached molecule ids to remove.",
    )
    args = parser.parse_args()

    source_path = Path(args.input)
    output_path = Path(args.output)
    excluded = set(args.exclude)

    packed_cache_cls, packed_chunk_ref_cls = ensure_pickle_schema()
    state = load_cache(source_path)

    split_ids = dict(state["split_ids"])
    split_structure_counts = dict(state["split_structure_counts"])
    packed_chunk_refs = state.get("packed_chunk_refs")
    split_ids_by_dataset = state.get("split_ids_by_dataset")
    split_structure_counts_by_dataset = state.get("split_structure_counts_by_dataset")
    dataset_ids = list(state.get("dataset_ids") or [])

    if packed_chunk_refs is None:
        raise ValueError("This helper currently expects packed_chunk_refs-based caches.")

    new_split_ids = {}
    new_split_structure_counts = {}
    new_refs = {}
    new_split_ids_by_dataset = {} if split_ids_by_dataset is not None else None
    new_split_structure_counts_by_dataset = (
        {} if split_structure_counts_by_dataset is not None else None
    )

    removed_total = 0
    removed_structure_total = 0
    removed_by_split = {split_name: 0 for split_name in split_ids}

    for split_name, molecule_ids in split_ids.items():
        kept_ids = [molecule_id for molecule_id in molecule_ids if molecule_id not in excluded]
        removed_ids = [molecule_id for molecule_id in molecule_ids if molecule_id in excluded]
        removed_total += len(removed_ids)
        removed_by_split[split_name] = len(removed_ids)
        new_split_ids[split_name] = kept_ids

        kept_refs = []
        split_structure_count = 0
        dataset_structure_counts = {dataset_id: 0 for dataset_id in dataset_ids}

        for raw_ref in packed_chunk_refs[split_name]:
            ref_state = object_state(raw_ref)
            if ref_state["molecule_id"] in excluded:
                removed_structure_total += int(ref_state["num_structures"])
                continue
            reference = to_reference(ref_state, packed_chunk_ref_cls)
            kept_refs.append(reference)
            split_structure_count += int(reference.num_structures)
            dataset_structure_counts.setdefault(reference.dataset_id, 0)
            dataset_structure_counts[reference.dataset_id] += int(reference.num_structures)

        new_refs[split_name] = kept_refs
        new_split_structure_counts[split_name] = split_structure_count

        if new_split_ids_by_dataset is not None:
            dataset_lists = {}
            for dataset_id, dataset_molecule_ids in split_ids_by_dataset[split_name].items():
                dataset_lists[dataset_id] = [
                    molecule_id for molecule_id in dataset_molecule_ids if molecule_id not in excluded
                ]
            new_split_ids_by_dataset[split_name] = dataset_lists

        if new_split_structure_counts_by_dataset is not None:
            for dataset_id in dataset_ids:
                dataset_structure_counts.setdefault(dataset_id, 0)
            new_split_structure_counts_by_dataset[split_name] = dataset_structure_counts

    filtered_cache = packed_cache_cls(
        split_ids=new_split_ids,
        split_structure_counts=new_split_structure_counts,
        packed_chunks=None,
        packed_chunk_refs=new_refs,
        dataset_ids=dataset_ids,
        split_ids_by_dataset=new_split_ids_by_dataset,
        split_structure_counts_by_dataset=new_split_structure_counts_by_dataset,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(filtered_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"input={source_path}")
    print(f"output={output_path}")
    print(f"excluded={len(excluded)}")
    print(f"removed_molecules={removed_total}")
    print(f"removed_structures={removed_structure_total}")
    for split_name in ["train", "val", "test"]:
        print(
            f"{split_name}: molecules={len(new_split_ids[split_name])} "
            f"structures={new_split_structure_counts[split_name]} "
            f"removed={removed_by_split.get(split_name, 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
