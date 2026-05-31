"""Unified ``.xmo`` preprocessing entry for the end-to-end E3VB pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data.hdf5_io import (
    chunkDatasetPath,
    clearPackedChunkStore,
    isHdf5Path,
    readPackedCacheMetadata,
    writePackedCacheMetadata,
    writePackedChunk,
)
from data.lap_pe import lapPeFromEdges
from data.schema import (
    DatasetSourceConfig,
    PackedChunkReference,
    PackedDatasetCache,
    ProcessedDatasetCache,
    ProcessedMolecule,
    ProcessedMoleculeChunk,
    UnifiedSample,
)
from data.xmo_builder import BuiltMoleculeSample, XmoFeatureBuilder
from data.xmo_parser import XmoParser


@dataclass
class UnifiedDataConfig:
    """Configure preprocessing directly from a directory of ``.xmo`` files."""

    xmo_dir: str | None
    seed: int
    split: Tuple[float, float, float]
    processed_cache_path: str | None = None
    packed_cache_path: str | None = None
    max_structures_per_chunk: int | None = None
    lap_pe_k: int = 0
    datasets: Tuple[DatasetSourceConfig, ...] | None = None


class XmoDatasetProcessor:
    """Parse ``.xmo`` files and organize them into current training cache objects."""

    def __init__(self, config: UnifiedDataConfig):
        self.config = config
        self.parser_cls = XmoParser
        self.builder = XmoFeatureBuilder(lap_pe_k=config.lap_pe_k)

    def dataSources(self) -> list[DatasetSourceConfig]:
        """Return configured dataset sources in stable order."""

        configured_sources = getattr(self.config, "datasets", None)
        if configured_sources:
            return list(configured_sources)
        if self.config.xmo_dir is None:
            raise ValueError("UnifiedDataConfig must define xmo_dir when datasets is not provided.")
        return [
            DatasetSourceConfig(
                dataset_id="default",
                xmo_dir=str(self.config.xmo_dir),
                split=tuple(float(x) for x in self.config.split),
            )
        ]

    def usesMultipleDatasets(self) -> bool:
        """Return whether the current config merges more than one dataset."""

        return len(self.dataSources()) > 1

    def cacheMoleculeId(self, dataset_id: str, molecule_id: str) -> str:
        """Build one cache-stable molecule id, prefixing when multiple datasets are merged."""

        if not self.usesMultipleDatasets():
            return str(molecule_id)
        return f"{dataset_id}::{molecule_id}"

    def deriveLapPeFields(
        self,
        num_atoms: int,
        atom_senders: np.ndarray,
        atom_receivers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derive Laplacian PE fields for one structure when older caches miss them."""

        return lapPeFromEdges(
            num_nodes=num_atoms,
            senders=np.asarray(atom_senders, dtype=np.int32),
            receivers=np.asarray(atom_receivers, dtype=np.int32),
            k=int(self.config.lap_pe_k),
            eps=1.0e-12,
            add_self_loops=False,
        )

    def deriveActiveFields(
        self,
        orbital_role: np.ndarray,
        rumer_senders: np.ndarray,
        rumer_receivers: np.ndarray,
        rumer_edge_type: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Derive active-only orbital gather indices and active Rumer topology."""

        active_orbital_index = np.flatnonzero(np.asarray(orbital_role, dtype=np.int32) == 2).astype(np.int32)
        active_local_map = {
            int(full_orbital_id): local_active_id
            for local_active_id, full_orbital_id in enumerate(active_orbital_index.tolist())
        }

        active_rumer_senders = []
        active_rumer_receivers = []
        active_rumer_edge_type = []
        for sender, receiver, edge_type in zip(
            np.asarray(rumer_senders, dtype=np.int32).tolist(),
            np.asarray(rumer_receivers, dtype=np.int32).tolist(),
            np.asarray(rumer_edge_type, dtype=np.int32).tolist(),
        ):
            if edge_type != 2:
                continue
            if (sender not in active_local_map) or (receiver not in active_local_map):
                continue
            active_rumer_senders.append(active_local_map[sender])
            active_rumer_receivers.append(active_local_map[receiver])
            active_rumer_edge_type.append(2)

        return (
            active_orbital_index,
            np.asarray(active_rumer_senders, dtype=np.int32),
            np.asarray(active_rumer_receivers, dtype=np.int32),
            np.asarray(active_rumer_edge_type, dtype=np.int32),
        )

    def loadProcessedCache(self) -> ProcessedDatasetCache | None:
        """Load one processed-dataset cache from disk when configured."""

        if self.config.processed_cache_path is None:
            return None
        path = Path(self.config.processed_cache_path)
        if not path.exists():
            return None
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, ProcessedDatasetCache):
            raise ValueError(f"Invalid processed cache payload at {path}.")
        return payload

    def loadPackedCache(self) -> PackedDatasetCache | None:
        """Load one packed-dataset cache from disk when configured."""

        if self.config.packed_cache_path is None:
            return None
        path = Path(self.config.packed_cache_path)
        if not path.exists():
            return None
        if isHdf5Path(path):
            payload = readPackedCacheMetadata(path)
            if not isinstance(payload, PackedDatasetCache):
                raise ValueError(f"Invalid packed cache payload at {path}.")
            return payload
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, PackedDatasetCache):
            raise ValueError(f"Invalid packed cache payload at {path}.")
        return payload

    def packedChunkDirectory(self) -> Path | None:
        """Return the directory used to store streaming packed chunk payloads."""

        if self.config.packed_cache_path is None:
            return None
        if isHdf5Path(self.config.packed_cache_path):
            return Path(self.config.packed_cache_path)
        cache_path = Path(self.config.packed_cache_path)
        return cache_path.parent / f"{cache_path.stem}_chunks"

    def packedChunkPath(self, split_name: str, molecule_id: str, chunk_index: int) -> Path:
        """Build one deterministic file path for a packed chunk payload."""

        chunk_dir = self.packedChunkDirectory()
        if chunk_dir is None:
            raise ValueError("packed_cache_path must be configured before writing packed chunk files.")
        if isHdf5Path(chunk_dir):
            raise ValueError("packedChunkPath is only valid for pickle-backed chunk storage.")
        safe_molecule_id = molecule_id.replace("/", "_")
        return chunk_dir / split_name / f"{safe_molecule_id}_chunk{int(chunk_index):04d}.pkl"

    def packedChunkPathForView(
        self,
        split_name: str,
        molecule_id: str,
        chunk_index: int,
        view_name: str,
    ) -> Path:
        """Build one deterministic file path for a packed chunk payload under one view."""

        chunk_dir = self.packedChunkDirectory()
        if chunk_dir is None:
            raise ValueError("packed_cache_path must be configured before writing packed chunk files.")
        if isHdf5Path(chunk_dir):
            raise ValueError("packedChunkPathForView is only valid for pickle-backed chunk storage.")
        safe_molecule_id = molecule_id.replace("/", "_")
        return chunk_dir / view_name / split_name / f"{safe_molecule_id}_chunk{int(chunk_index):04d}.pkl"

    def saveProcessedCache(self, cache: ProcessedDatasetCache) -> None:
        """Persist one processed cache when a target path is configured."""

        if self.config.processed_cache_path is None:
            return
        path = Path(self.config.processed_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(cache, handle)

    def savePackedCache(self, cache: PackedDatasetCache) -> None:
        """Persist one packed cache when a target path is configured."""

        if self.config.packed_cache_path is None:
            return
        path = Path(self.config.packed_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isHdf5Path(path):
            writePackedCacheMetadata(path, cache)
            return
        with open(path, "wb") as handle:
            pickle.dump(cache, handle)

    def savePackedChunk(
        self,
        split_name: str,
        chunk,
        view_name: str = "full",
    ) -> PackedChunkReference:
        """Persist one packed chunk payload and return a lightweight reference."""

        if isHdf5Path(self.config.packed_cache_path):
            chunk_store_path = Path(self.config.packed_cache_path)
            dataset_path = chunkDatasetPath(
                split_name=split_name,
                molecule_id=chunk.molecule_id,
                chunk_index=chunk.chunk_index,
                view_name=view_name,
            )
            writePackedChunk(chunk_store_path, dataset_path=dataset_path, chunk=chunk)
            chunk_path_value = str(chunk_store_path)
            storage_format = "hdf5"
            hdf5_dataset_path = dataset_path
        else:
            chunk_path = self.packedChunkPathForView(
                split_name=split_name,
                molecule_id=chunk.molecule_id,
                chunk_index=chunk.chunk_index,
                view_name=view_name,
            )
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            with open(chunk_path, "wb") as handle:
                pickle.dump(chunk, handle)
            chunk_path_value = str(chunk_path)
            storage_format = "pickle"
            hdf5_dataset_path = None
        return PackedChunkReference(
            molecule_id=chunk.molecule_id,
            chunk_index=int(chunk.chunk_index),
            num_chunks=int(chunk.num_chunks),
            chunk_path=chunk_path_value,
            num_structures=int(chunk.atom_n_node.shape[0]),
            num_focus_structures=int(
                np.asarray(
                    getattr(
                        chunk,
                        "top_mass_focus_mask",
                        np.zeros((int(chunk.atom_n_node.shape[0]),), dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).sum()
            ),
            static_num_atoms=int(np.asarray(chunk.atom_numbers, dtype=np.int32).shape[0]),
            total_atoms=int(np.asarray(chunk.atom_n_node, dtype=np.int32).sum()),
            total_atom_edges=int(np.asarray(chunk.atom_n_edge, dtype=np.int32).sum()),
            total_orbitals=int(np.asarray(chunk.rumer_n_node, dtype=np.int32).sum()),
            total_rumer_edges=int(np.asarray(chunk.rumer_n_edge, dtype=np.int32).sum()),
            total_active_orbitals=int(np.asarray(chunk.active_rumer_n_node, dtype=np.int32).sum()),
            total_active_rumer_edges=int(np.asarray(chunk.active_rumer_n_edge, dtype=np.int32).sum()),
            dataset_id=str(getattr(chunk, "dataset_id", "default")),
            storage_format=storage_format,
            hdf5_dataset_path=hdf5_dataset_path,
        )

    def selectPackedViewStructures(
        self,
        molecule: ProcessedMolecule,
        view_name: str,
    ) -> List:
        """Select one structure list for the requested offline packed view."""

        if view_name == "full":
            return list(molecule.structures)
        if view_name == "focus_only":
            selected = [
                structure
                for structure in molecule.structures
                if float(getattr(structure, "top_mass_focus", 0.0)) > 0.5
            ]
            if (len(selected) == 0) and (len(molecule.structures) > 0):
                selected = [
                    max(
                        molecule.structures,
                        key=lambda structure: float(getattr(structure, "target", 0.0)),
                    )
                ]
            return selected
        raise ValueError(f"Unsupported packed molecule view: {view_name}")

    def buildPackedViewMolecule(
        self,
        molecule: ProcessedMolecule,
        view_name: str,
    ) -> ProcessedMolecule:
        """Build one molecule object whose structures match the requested packed view."""

        return ProcessedMolecule(
            dataset_id=str(getattr(molecule, "dataset_id", "default")),
            molecule_id=molecule.molecule_id,
            atom_numbers=molecule.atom_numbers,
            atom_positions=molecule.atom_positions,
            local_frame_e1=molecule.local_frame_e1,
            local_frame_e2=molecule.local_frame_e2,
            local_frame_e3=molecule.local_frame_e3,
            structures=self.selectPackedViewStructures(molecule, view_name=view_name),
        )

    def discoverXmoFiles(self, xmo_dir: str | None = None) -> list[Path]:
        """Return sorted ``.xmo`` files from the configured input directory."""

        resolved_xmo_dir = self.config.xmo_dir if xmo_dir is None else xmo_dir
        if resolved_xmo_dir is None:
            raise ValueError("discoverXmoFiles requires one concrete xmo_dir.")
        xmo_dir_path = Path(resolved_xmo_dir)
        if not xmo_dir_path.exists():
            raise FileNotFoundError(f"Configured xmo_dir does not exist: {xmo_dir_path}")
        files = sorted(xmo_dir_path.glob("*.xmo"))
        if len(files) == 0:
            raise FileNotFoundError(f"No .xmo files were found under {xmo_dir_path}")
        return files

    def normalizeVector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize one 3D vector with an epsilon fallback."""

        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return (vector / norm).astype(np.float32)

    def chooseReferenceAxis(self, direction: np.ndarray) -> np.ndarray:
        """Choose the least-aligned Cartesian axis to stabilize local frames."""

        basis = np.eye(3, dtype=np.float32)
        index = int(np.argmin(np.abs(basis @ direction)))
        return basis[index]

    def orthogonalDirection(self, direction: np.ndarray) -> np.ndarray:
        """Build one unit vector orthogonal to the input direction."""

        reference = self.chooseReferenceAxis(direction)
        orthogonal = reference - float(np.dot(reference, direction)) * direction
        return self.normalizeVector(orthogonal.astype(np.float32))

    def buildActiveAtomMask(
        self,
        num_atoms: int,
        orbital_atom_index: np.ndarray,
        orbital_role: np.ndarray,
    ) -> np.ndarray:
        """Mark atoms that own at least one active orbital."""

        active_mask = np.zeros((num_atoms,), dtype=bool)
        active_owner = np.asarray(orbital_atom_index, dtype=np.int32)[
            np.asarray(orbital_role, dtype=np.int32) == 2,
            0,
        ]
        active_mask[np.asarray(active_owner, dtype=np.int32)] = True
        return active_mask

    def buildStaticBondAdjacency(
        self,
        num_atoms: int,
        structures: list[ProcessedStructureSample],
    ) -> np.ndarray:
        """Build one molecule-static undirected adjacency by unioning structure edges."""

        adjacency = np.zeros((num_atoms, num_atoms), dtype=bool)
        for structure in structures:
            senders = np.asarray(structure.atom_senders, dtype=np.int32)
            receivers = np.asarray(structure.atom_receivers, dtype=np.int32)
            adjacency[senders, receivers] = True
            adjacency[receivers, senders] = True
        np.fill_diagonal(adjacency, False)
        return adjacency

    def orderedFrameNeighborCandidates(
        self,
        atom_id: int,
        distance: np.ndarray,
        atom_numbers: np.ndarray,
        active_atom_mask: np.ndarray | None,
        bonded_adjacency: np.ndarray | None,
    ) -> list[int]:
        """Rank frame-defining neighbor candidates by chemistry-aware priority."""

        num_atoms = int(distance.shape[0])
        heavy_mask = np.asarray(atom_numbers, dtype=np.int32) > 1
        active_mask = (
            np.asarray(active_atom_mask, dtype=bool)
            if active_atom_mask is not None
            else np.zeros((num_atoms,), dtype=bool)
        )
        bonded_mask = (
            np.asarray(bonded_adjacency[atom_id], dtype=bool)
            if bonded_adjacency is not None
            else np.ones((num_atoms,), dtype=bool)
        )
        base_mask = np.ones((num_atoms,), dtype=bool)
        base_mask[atom_id] = False

        priority_masks: list[np.ndarray] = []
        if bool(active_mask[atom_id]):
            priority_masks.append(base_mask & bonded_mask & heavy_mask & active_mask)
        priority_masks.append(base_mask & bonded_mask & heavy_mask)
        priority_masks.append(base_mask & heavy_mask)
        priority_masks.append(base_mask & bonded_mask)
        priority_masks.append(base_mask)

        ordered: list[int] = []
        seen: set[int] = set()
        atom_distance = np.asarray(distance[atom_id], dtype=np.float32)
        for mask in priority_masks:
            candidate_ids = np.flatnonzero(mask)
            if candidate_ids.size == 0:
                continue
            ranked = candidate_ids[np.argsort(atom_distance[candidate_ids])]
            for neighbor_id in ranked.tolist():
                neighbor_id = int(neighbor_id)
                if neighbor_id not in seen:
                    ordered.append(neighbor_id)
                    seen.add(neighbor_id)
        return ordered

    def buildLocalFramesForPositions(
        self,
        positions: np.ndarray,
        atom_numbers: np.ndarray,
        active_atom_mask: np.ndarray | None = None,
        bonded_adjacency: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build one sigma/in-plane/normal frame per atom from geometry and bonded heavy-atom priorities."""

        positions = np.asarray(positions, dtype=np.float32)
        atom_numbers = np.asarray(atom_numbers, dtype=np.int32)
        num_atoms = int(positions.shape[0])
        if num_atoms == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            return empty, empty, empty

        distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        distance = distance + np.eye(num_atoms, dtype=np.float32) * 1e6

        e1_parts = []
        e2_parts = []
        e3_parts = []
        default_e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        default_e2 = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        default_e3 = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        for atom_id in range(num_atoms):
            if num_atoms == 1:
                e1_parts.append(default_e1)
                e2_parts.append(default_e2)
                e3_parts.append(default_e3)
                continue

            ranked_neighbors = self.orderedFrameNeighborCandidates(
                atom_id=atom_id,
                distance=distance,
                atom_numbers=atom_numbers,
                active_atom_mask=active_atom_mask,
                bonded_adjacency=bonded_adjacency,
            )
            if len(ranked_neighbors) == 0:
                e1_parts.append(default_e1)
                e2_parts.append(default_e2)
                e3_parts.append(default_e3)
                continue

            first_neighbor = int(ranked_neighbors[0])
            sigma_axis = self.normalizeVector((positions[first_neighbor] - positions[atom_id]).astype(np.float32))

            inplane_seed = None
            if num_atoms >= 3:
                for neighbor_id in ranked_neighbors[1:]:
                    candidate = positions[int(neighbor_id)] - positions[atom_id]
                    projected = candidate - float(np.dot(candidate, sigma_axis)) * sigma_axis
                    if float(np.linalg.norm(projected)) >= 1e-8:
                        inplane_seed = self.normalizeVector(projected.astype(np.float32))
                        break

            if inplane_seed is None:
                inplane_seed = self.orthogonalDirection(sigma_axis)

            plane_normal = self.normalizeVector(np.cross(sigma_axis, inplane_seed).astype(np.float32))
            inplane_perp = self.normalizeVector(np.cross(plane_normal, sigma_axis).astype(np.float32))
            e1_parts.append(sigma_axis)
            e2_parts.append(inplane_perp)
            e3_parts.append(plane_normal)

        return (
            np.stack(e1_parts, axis=0),
            np.stack(e2_parts, axis=0),
            np.stack(e3_parts, axis=0),
        )

    def createMoleculeSplits(
        self,
        molecule_ids: list[str],
        split: Tuple[float, float, float] | None = None,
        seed_offset: int = 0,
    ) -> Dict[str, List[str]]:
        """Create a deterministic molecule-level train/val/test split."""

        shuffled_ids = list(molecule_ids)
        rng = np.random.default_rng(int(self.config.seed) + int(seed_offset))
        rng.shuffle(shuffled_ids)

        ratio = np.asarray(self.config.split if split is None else split, dtype=np.float64)
        ratio = ratio / ratio.sum()
        total = len(shuffled_ids)
        num_train = min(int(round(float(ratio[0]) * total)), total)
        num_val = min(int(round(float(ratio[1]) * total)), total - num_train)

        return {
            "train": shuffled_ids[:num_train],
            "val": shuffled_ids[num_train : num_train + num_val],
            "test": shuffled_ids[num_train + num_val :],
        }

    def buildProcessedMolecule(
        self,
        built: BuiltMoleculeSample,
        dataset_id: str,
    ) -> ProcessedMolecule:
        """Convert built graph fields into the cached processed-molecule object."""

        active_atom_mask = (
            self.buildActiveAtomMask(
                num_atoms=int(built.atom_numbers.shape[0]),
                orbital_atom_index=built.structures[0].orbital_atom_index,
                orbital_role=built.structures[0].orbital_role,
            )
            if len(built.structures) > 0
            else np.zeros((int(built.atom_numbers.shape[0]),), dtype=bool)
        )
        bonded_adjacency = self.buildStaticBondAdjacency(
            num_atoms=int(built.atom_numbers.shape[0]),
            structures=built.structures,
        )
        local_frame_e1, local_frame_e2, local_frame_e3 = self.buildLocalFramesForPositions(
            positions=built.atom_positions,
            atom_numbers=built.atom_numbers,
            active_atom_mask=active_atom_mask,
            bonded_adjacency=bonded_adjacency,
        )
        return ProcessedMolecule(
            dataset_id=dataset_id,
            molecule_id=self.cacheMoleculeId(dataset_id=dataset_id, molecule_id=built.molecule_id),
            atom_numbers=np.asarray(built.atom_numbers, dtype=np.int32),
            atom_positions=np.asarray(built.atom_positions, dtype=np.float32),
            local_frame_e1=local_frame_e1,
            local_frame_e2=local_frame_e2,
            local_frame_e3=local_frame_e3,
            structures=list(built.structures),
        )

    def buildProcessedDatasetCache(self) -> ProcessedDatasetCache:
        """Parse all configured ``.xmo`` files and build the processed-dataset cache."""

        processed_molecules: Dict[str, ProcessedMolecule] = {}
        split_ids_by_dataset: Dict[str, Dict[str, List[str]]] = {
            "train": {},
            "val": {},
            "test": {},
        }
        merged_split_ids: Dict[str, List[str]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        dataset_ids: list[str] = []

        for dataset_offset, source in enumerate(self.dataSources()):
            dataset_processed_molecules: Dict[str, ProcessedMolecule] = {}
            for path in self.discoverXmoFiles(source.xmo_dir):
                try:
                    parsed = self.parser_cls(path).parse()
                    built = self.builder.build(parsed)
                    processed = self.buildProcessedMolecule(built, dataset_id=source.dataset_id)
                except Exception as exc:
                    print(
                        "SKIP_XMO "
                        f"dataset_id={source.dataset_id} "
                        f"path={path} "
                        f"error_type={type(exc).__name__} "
                        f"error={exc}",
                        flush=True,
                    )
                    continue
                if len(processed.structures) == 0:
                    continue
                dataset_processed_molecules[processed.molecule_id] = processed
                processed_molecules[processed.molecule_id] = processed

            if len(dataset_processed_molecules) == 0:
                raise RuntimeError(
                    f"No valid molecules were parsed for dataset_id={source.dataset_id} "
                    f"from xmo_dir={source.xmo_dir}."
                )

            dataset_ids.append(source.dataset_id)
            dataset_split_ids = self.createMoleculeSplits(
                molecule_ids=sorted(dataset_processed_molecules.keys()),
                split=source.split,
                seed_offset=dataset_offset,
            )
            for split_name in ["train", "val", "test"]:
                split_ids_by_dataset[split_name][source.dataset_id] = list(dataset_split_ids[split_name])
                merged_split_ids[split_name].extend(dataset_split_ids[split_name])

        if len(processed_molecules) == 0:
            raise RuntimeError("No valid molecules were parsed from the configured .xmo directory.")

        return ProcessedDatasetCache(
            split_ids=merged_split_ids,
            molecules=processed_molecules,
            dataset_ids=dataset_ids,
            split_ids_by_dataset=split_ids_by_dataset,
        )

    def createProcessedDatasetCache(self, force_rebuild: bool = False) -> ProcessedDatasetCache:
        """Load processed cache from disk or rebuild it from ``.xmo`` files."""

        cache = None if force_rebuild else self.loadProcessedCache()
        if cache is not None:
            return cache
        cache = self.buildProcessedDatasetCache()
        self.saveProcessedCache(cache)
        return cache

    def requireProcessedDatasetCache(self) -> ProcessedDatasetCache:
        """Load one previously exported processed cache."""

        cache = self.loadProcessedCache()
        if cache is None:
            raise FileNotFoundError(
                "Processed dataset cache was not found. Run the preprocess stage first."
            )
        return cache

    def materializeUnifiedSamples(self, molecule: ProcessedMolecule) -> List[UnifiedSample]:
        """Expand one processed molecule into per-structure unified samples."""

        return [
            UnifiedSample(
                dataset_id=str(getattr(molecule, "dataset_id", "default")),
                molecule_id=molecule.molecule_id,
                vb_index=structure.vb_index,
                atom_numbers=molecule.atom_numbers,
                atom_positions=molecule.atom_positions,
                atom_node_features=structure.atom_node_features,
                atom_senders=structure.atom_senders,
                atom_receivers=structure.atom_receivers,
                atom_pair_features=structure.atom_pair_features,
                lap_evals=(
                    np.asarray(structure.lap_evals, dtype=np.float32)
                    if hasattr(structure, "lap_evals")
                    else self.deriveLapPeFields(
                        num_atoms=int(molecule.atom_numbers.shape[0]),
                        atom_senders=structure.atom_senders,
                        atom_receivers=structure.atom_receivers,
                    )[0]
                ),
                lap_evecs=(
                    np.asarray(structure.lap_evecs, dtype=np.float32)
                    if hasattr(structure, "lap_evecs")
                    else self.deriveLapPeFields(
                        num_atoms=int(molecule.atom_numbers.shape[0]),
                        atom_senders=structure.atom_senders,
                        atom_receivers=structure.atom_receivers,
                    )[1]
                ),
                orbital_atom_index=structure.orbital_atom_index,
                orbital_role=structure.orbital_role,
                active_slot_index=structure.active_slot_index,
                rumer_senders=structure.rumer_senders,
                rumer_receivers=structure.rumer_receivers,
                rumer_edge_type=structure.rumer_edge_type,
                active_orbital_index=(
                    np.asarray(structure.active_orbital_index, dtype=np.int32)
                    if hasattr(structure, "active_orbital_index")
                    else self.deriveActiveFields(
                        orbital_role=structure.orbital_role,
                        rumer_senders=structure.rumer_senders,
                        rumer_receivers=structure.rumer_receivers,
                        rumer_edge_type=structure.rumer_edge_type,
                    )[0]
                ),
                active_rumer_senders=(
                    np.asarray(structure.active_rumer_senders, dtype=np.int32)
                    if hasattr(structure, "active_rumer_senders")
                    else self.deriveActiveFields(
                        orbital_role=structure.orbital_role,
                        rumer_senders=structure.rumer_senders,
                        rumer_receivers=structure.rumer_receivers,
                        rumer_edge_type=structure.rumer_edge_type,
                    )[1]
                ),
                active_rumer_receivers=(
                    np.asarray(structure.active_rumer_receivers, dtype=np.int32)
                    if hasattr(structure, "active_rumer_receivers")
                    else self.deriveActiveFields(
                        orbital_role=structure.orbital_role,
                        rumer_senders=structure.rumer_senders,
                        rumer_receivers=structure.rumer_receivers,
                        rumer_edge_type=structure.rumer_edge_type,
                    )[2]
                ),
                active_rumer_edge_type=(
                    np.asarray(structure.active_rumer_edge_type, dtype=np.int32)
                    if hasattr(structure, "active_rumer_edge_type")
                    else self.deriveActiveFields(
                        orbital_role=structure.orbital_role,
                        rumer_senders=structure.rumer_senders,
                        rumer_receivers=structure.rumer_receivers,
                        rumer_edge_type=structure.rumer_edge_type,
                    )[3]
                ),
                target=structure.target,
                target_max=structure.target_max,
                top_mass_focus=float(getattr(structure, "top_mass_focus", 0.0)),
            )
            for structure in molecule.structures
        ]

    def createSplitMoleculeGroups(self) -> Dict[str, List[List[UnifiedSample]]]:
        """Create per-split groups where each group contains one molecule's structures."""

        cache = self.createProcessedDatasetCache()
        return {
            split_name: [
                self.materializeUnifiedSamples(cache.molecules[molecule_id])
                for molecule_id in cache.split_ids[split_name]
            ]
            for split_name in ["train", "val", "test"]
        }

    def createSplitSamples(self) -> Dict[str, List[UnifiedSample]]:
        """Flatten split-wise molecule groups into split-wise sample lists."""

        split_groups = self.createSplitMoleculeGroups()
        return {
            split_name: [sample for group in split_groups[split_name] for sample in group]
            for split_name in ["train", "val", "test"]
        }

    def createSplitProcessedMolecules(self) -> Dict[str, List[ProcessedMolecule]]:
        """Create processed-molecule lists for every split."""

        cache = self.createProcessedDatasetCache()
        return {
            split_name: [cache.molecules[molecule_id] for molecule_id in cache.split_ids[split_name]]
            for split_name in ["train", "val", "test"]
        }

    def chunkProcessedMolecule(self, molecule: ProcessedMolecule) -> List[ProcessedMoleculeChunk]:
        """Split one molecule into one or more single-molecule chunks."""

        max_structures = self.config.max_structures_per_chunk
        if (max_structures is None) or (max_structures <= 0):
            max_structures = len(molecule.structures)

        chunks: List[ProcessedMoleculeChunk] = []
        total_chunks = max(1, int(np.ceil(len(molecule.structures) / max_structures)))
        for chunk_index, start in enumerate(range(0, len(molecule.structures), max_structures)):
            chunks.append(
                ProcessedMoleculeChunk(
                    dataset_id=str(getattr(molecule, "dataset_id", "default")),
                    molecule_id=molecule.molecule_id,
                    chunk_index=chunk_index,
                    num_chunks=total_chunks,
                    atom_numbers=molecule.atom_numbers,
                    atom_positions=molecule.atom_positions,
                    local_frame_e1=molecule.local_frame_e1,
                    local_frame_e2=molecule.local_frame_e2,
                    local_frame_e3=molecule.local_frame_e3,
                    structures=molecule.structures[start : start + max_structures],
                )
            )
        return chunks

    def createSplitMoleculeChunks(self) -> Dict[str, List[ProcessedMoleculeChunk]]:
        """Create split-wise chunk lists while preserving single-molecule grouping."""

        split_molecules = self.createSplitProcessedMolecules()
        split_chunks: Dict[str, List[ProcessedMoleculeChunk]] = {"train": [], "val": [], "test": []}
        for split_name in ["train", "val", "test"]:
            for molecule in split_molecules[split_name]:
                split_chunks[split_name].extend(self.chunkProcessedMolecule(molecule))
        return split_chunks

    def buildPackedDatasetCache(self) -> PackedDatasetCache:
        """Pack processed molecules into on-disk single-molecule chunk payloads."""

        from data.grain_pipeline import GraphPackingAdapter

        processed_cache = self.createProcessedDatasetCache()
        adapter = GraphPackingAdapter()
        dataset_ids = list(getattr(processed_cache, "dataset_ids", None) or ["default"])
        split_ids_by_dataset = getattr(processed_cache, "split_ids_by_dataset", None)
        if isHdf5Path(self.config.packed_cache_path):
            clearPackedChunkStore(self.config.packed_cache_path)
        view_names = ("full", "focus_only")
        if self.packedChunkDirectory() is None:
            split_chunks = self.createSplitMoleculeChunks()
            packed_chunks = {
                split_name: [adapter.packChunk(chunk) for chunk in split_chunks[split_name]]
                for split_name in ["train", "val", "test"]
            }
            split_structure_counts = {
                split_name: int(sum(int(chunk.atom_n_node.shape[0]) for chunk in packed_chunks[split_name]))
                for split_name in ["train", "val", "test"]
            }
            split_structure_counts_by_dataset: Dict[str, Dict[str, int]] = {
                split_name: {dataset_id: 0 for dataset_id in dataset_ids}
                for split_name in ["train", "val", "test"]
            }
            for split_name in ["train", "val", "test"]:
                for chunk in packed_chunks[split_name]:
                    split_structure_counts_by_dataset[split_name].setdefault(chunk.dataset_id, 0)
                    split_structure_counts_by_dataset[split_name][chunk.dataset_id] += int(
                        chunk.atom_n_node.shape[0]
                    )
            return PackedDatasetCache(
                split_ids=processed_cache.split_ids,
                split_structure_counts=split_structure_counts,
                packed_chunks_by_view={
                    "full": packed_chunks,
                    "focus_only": {
                        split_name: [
                            adapter.packChunk(chunk)
                            for molecule in self.createSplitProcessedMolecules()[split_name]
                            for chunk in self.chunkProcessedMolecule(
                                self.buildPackedViewMolecule(molecule, view_name="focus_only")
                            )
                            if len(chunk.structures) > 0
                        ]
                        for split_name in ["train", "val", "test"]
                    },
                },
                split_structure_counts_by_view={
                    "full": split_structure_counts,
                    "focus_only": {
                        split_name: int(
                            sum(
                                int(chunk.atom_n_node.shape[0])
                                for chunk in [
                                    adapter.packChunk(view_chunk)
                                    for molecule in self.createSplitProcessedMolecules()[split_name]
                                    for view_chunk in self.chunkProcessedMolecule(
                                        self.buildPackedViewMolecule(molecule, view_name="focus_only")
                                    )
                                    if len(view_chunk.structures) > 0
                                ]
                            )
                        )
                        for split_name in ["train", "val", "test"]
                    },
                },
                packed_chunks=packed_chunks,
                packed_chunk_refs=None,
                dataset_ids=dataset_ids,
                split_ids_by_dataset=split_ids_by_dataset,
                split_structure_counts_by_dataset=split_structure_counts_by_dataset,
            )
        split_molecules = self.createSplitProcessedMolecules()
        packed_chunk_refs_by_view: Dict[str, Dict[str, List[PackedChunkReference]]] = {
            view_name: {"train": [], "val": [], "test": []}
            for view_name in view_names
        }
        split_structure_counts_by_view: Dict[str, Dict[str, int]] = {
            view_name: {"train": 0, "val": 0, "test": 0}
            for view_name in view_names
        }
        split_structure_counts_by_dataset_by_view: Dict[str, Dict[str, Dict[str, int]]] = {
            view_name: {
                split_name: {dataset_id: 0 for dataset_id in dataset_ids}
                for split_name in ["train", "val", "test"]
            }
            for view_name in view_names
        }

        for view_name in view_names:
            for split_name in ["train", "val", "test"]:
                for molecule in split_molecules[split_name]:
                    view_molecule = self.buildPackedViewMolecule(molecule, view_name=view_name)
                    for chunk in self.chunkProcessedMolecule(view_molecule):
                        if len(chunk.structures) == 0:
                            continue
                        packed_chunk = adapter.packChunk(chunk)
                        packed_chunk_refs_by_view[view_name][split_name].append(
                            self.savePackedChunk(
                                split_name=split_name,
                                chunk=packed_chunk,
                                view_name=view_name,
                            )
                        )
                        split_structure_counts_by_view[view_name][split_name] += int(
                            packed_chunk.atom_n_node.shape[0]
                        )
                        split_structure_counts_by_dataset_by_view[view_name][split_name].setdefault(
                            packed_chunk.dataset_id,
                            0,
                        )
                        split_structure_counts_by_dataset_by_view[view_name][split_name][
                            packed_chunk.dataset_id
                        ] += int(packed_chunk.atom_n_node.shape[0])

        return PackedDatasetCache(
            split_ids=processed_cache.split_ids,
            split_structure_counts=split_structure_counts_by_view["full"],
            packed_chunks_by_view=None,
            packed_chunk_refs_by_view=packed_chunk_refs_by_view,
            split_structure_counts_by_view=split_structure_counts_by_view,
            split_structure_counts_by_dataset_by_view=split_structure_counts_by_dataset_by_view,
            packed_chunks=None,
            packed_chunk_refs=packed_chunk_refs_by_view["full"],
            dataset_ids=dataset_ids,
            split_ids_by_dataset=split_ids_by_dataset,
            split_structure_counts_by_dataset=split_structure_counts_by_dataset_by_view["full"],
        )

    def createPackedDatasetCache(self, force_rebuild: bool = False) -> PackedDatasetCache:
        """Load packed cache from disk or rebuild it from processed molecules."""

        cache = None if force_rebuild else self.loadPackedCache()
        if cache is not None:
            return cache
        cache = self.buildPackedDatasetCache()
        self.savePackedCache(cache)
        return cache

    def requirePackedDatasetCache(self) -> PackedDatasetCache:
        """Load one previously exported packed cache."""

        cache = self.loadPackedCache()
        if cache is None:
            raise FileNotFoundError(
                "Packed dataset cache was not found. Run the preprocess stage first."
            )
        return cache


UnifiedSampleProcessor = XmoDatasetProcessor
