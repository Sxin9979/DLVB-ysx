#!/usr/bin/env python3
"""Convert one pickle-backed packed cache into an HDF5-backed packed cache."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from data.hdf5_io import (
    chunkDatasetPath,
    clearPackedChunkStore,
    require_h5py,
    writePackedCacheMetadata,
    writePackedChunk,
)
from data.schema import PackedChunkReference, PackedDatasetCache, PackedMoleculeChunk


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Source packed cache pickle path.")
    parser.add_argument("--dst", required=True, help="Destination HDF5 cache path.")
    return parser.parse_args()


def load_pickle_cache(path: Path) -> PackedDatasetCache:
    """Load one pickle-backed PackedDatasetCache."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, PackedDatasetCache):
        raise TypeError(f"Expected PackedDatasetCache at {path}, got {type(payload)!r}.")
    return payload


def load_pickle_chunk(path: str) -> PackedMoleculeChunk:
    """Load one pickle-backed packed chunk payload."""

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, PackedMoleculeChunk):
        raise TypeError(f"Expected PackedMoleculeChunk at {path}, got {type(payload)!r}.")
    return payload


def convert_cache(src: Path, dst: Path) -> None:
    """Convert one pickle-backed cache into one HDF5-backed cache."""

    require_h5py()
    cache = load_pickle_cache(src)
    if cache.packed_chunk_refs is None:
        raise ValueError("Source cache does not contain packed_chunk_refs; streaming refs are required.")

    clearPackedChunkStore(dst)
    converted_refs: dict[str, list[PackedChunkReference]] = {"train": [], "val": [], "test": []}
    for split_name in ["train", "val", "test"]:
        for reference in cache.packed_chunk_refs[split_name]:
            chunk = load_pickle_chunk(reference.chunk_path)
            dataset_path = chunkDatasetPath(
                split_name=split_name,
                molecule_id=chunk.molecule_id,
                chunk_index=chunk.chunk_index,
            )
            writePackedChunk(dst, dataset_path=dataset_path, chunk=chunk)
            converted_refs[split_name].append(
                PackedChunkReference(
                    molecule_id=reference.molecule_id,
                    chunk_index=reference.chunk_index,
                    num_chunks=reference.num_chunks,
                    chunk_path=str(dst),
                    num_structures=reference.num_structures,
                    num_focus_structures=reference.num_focus_structures,
                    static_num_atoms=reference.static_num_atoms,
                    total_atoms=reference.total_atoms,
                    total_atom_edges=reference.total_atom_edges,
                    total_orbitals=reference.total_orbitals,
                    total_rumer_edges=reference.total_rumer_edges,
                    total_active_orbitals=reference.total_active_orbitals,
                    total_active_rumer_edges=reference.total_active_rumer_edges,
                    dataset_id=reference.dataset_id,
                    storage_format="hdf5",
                    hdf5_dataset_path=dataset_path,
                )
            )

    converted_cache = PackedDatasetCache(
        split_ids=cache.split_ids,
        split_structure_counts=cache.split_structure_counts,
        packed_chunks=None,
        packed_chunk_refs=converted_refs,
        dataset_ids=cache.dataset_ids,
        split_ids_by_dataset=cache.split_ids_by_dataset,
        split_structure_counts_by_dataset=cache.split_structure_counts_by_dataset,
    )
    writePackedCacheMetadata(dst, converted_cache)


def main() -> None:
    """CLI entrypoint."""

    arguments = parse_args()
    convert_cache(src=Path(arguments.src), dst=Path(arguments.dst))


if __name__ == "__main__":
    main()
