"""Grain + Jraph pipeline for unified E3VB samples."""

import os
import pickle
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import jax.numpy as jnp
import jraph
import numpy as np
from grain import python as grain

from data.lap_pe import lapPeFromEdges, repeatLapEvalsPerNode
from data.schema import (
    FixedBucketSpec,
    PackedChunkReference,
    PackedMoleculeChunk,
    ProcessedMoleculeChunk,
    UnifiedBatch,
    UnifiedSample,
)
from utils.top_mass import computeTopMassFocusMask, selectTopMassStructureIndices


class GraphPackingAdapter:
    """
    Pack a list of UnifiedSample objects into one UnifiedBatch.

    This adapter is the bridge between:
    - processed preprocessing outputs, and
    - end-to-end model inputs with Jraph graphs.
    """

    def __init__(self, lap_pe_k: int = 0):
        """
        Initialize one graph packing adapter.

        Arguments:
        - lap_pe_k: Number of Laplacian eigenvectors retained per atom graph.
        """

        self.lap_pe_k = int(lap_pe_k)

    def buildAtomGraph(self, samples: List[UnifiedSample]) -> jraph.GraphsTuple:
        """
        Build one batched atom-level graph.
        """

        atom_graphs: List[jraph.GraphsTuple] = []
        for sample in samples:
            nodes = {
                "features": jnp.asarray(sample.atom_node_features, dtype=jnp.float32),
                "numbers": jnp.asarray(sample.atom_numbers, dtype=jnp.int32),
                "positions": jnp.asarray(sample.atom_positions, dtype=jnp.float32),
            }
            edges = {
                "pair": jnp.asarray(sample.atom_pair_features, dtype=jnp.float32),
            }
            atom_graphs.append(
                jraph.GraphsTuple(
                    nodes=nodes,
                    edges=edges,
                    senders=jnp.asarray(sample.atom_senders, dtype=jnp.int32),
                    receivers=jnp.asarray(sample.atom_receivers, dtype=jnp.int32),
                    n_node=jnp.asarray([sample.atom_numbers.shape[0]], dtype=jnp.int32),
                    n_edge=jnp.asarray([sample.atom_senders.shape[0]], dtype=jnp.int32),
                    globals=None,
                )
            )
        return jraph.batch(atom_graphs)

    def buildRumerGraph(self, samples: List[UnifiedSample], orbital_feature_dim: int) -> jraph.GraphsTuple:
        """
        Build one batched Rumer graph with static topology.
        """

        rumer_graphs: List[jraph.GraphsTuple] = []
        for sample in samples:
            num_orbitals = sample.orbital_role.shape[0]
            rumer_graphs.append(
                jraph.GraphsTuple(
                    nodes=jnp.zeros((num_orbitals, orbital_feature_dim), dtype=jnp.float32),
                    edges={
                        "edge_type": jnp.asarray(sample.rumer_edge_type, dtype=jnp.int32),
                    },
                    senders=jnp.asarray(sample.rumer_senders, dtype=jnp.int32),
                    receivers=jnp.asarray(sample.rumer_receivers, dtype=jnp.int32),
                    n_node=jnp.asarray([num_orbitals], dtype=jnp.int32),
                    n_edge=jnp.asarray([sample.rumer_senders.shape[0]], dtype=jnp.int32),
                    globals=None,
                )
            )
        return jraph.batch(rumer_graphs)

    def buildActiveRumerGraph(self, samples: List[UnifiedSample], orbital_feature_dim: int) -> jraph.GraphsTuple:
        """
        Build one batched active-only Rumer graph with static topology.
        """

        rumer_graphs: List[jraph.GraphsTuple] = []
        for sample in samples:
            num_active_orbitals = int(sample.active_orbital_index.shape[0])
            rumer_graphs.append(
                jraph.GraphsTuple(
                    nodes=jnp.zeros((num_active_orbitals, orbital_feature_dim), dtype=jnp.float32),
                    edges={
                        "edge_type": jnp.asarray(sample.active_rumer_edge_type, dtype=jnp.int32),
                    },
                    senders=jnp.asarray(sample.active_rumer_senders, dtype=jnp.int32),
                    receivers=jnp.asarray(sample.active_rumer_receivers, dtype=jnp.int32),
                    n_node=jnp.asarray([num_active_orbitals], dtype=jnp.int32),
                    n_edge=jnp.asarray([sample.active_rumer_senders.shape[0]], dtype=jnp.int32),
                    globals=None,
                )
            )
        return jraph.batch(rumer_graphs)

    def buildOrbitalMetadata(self, samples: List[UnifiedSample]) -> Dict[str, jnp.ndarray]:
        """
        Build batched orbital metadata arrays with correct global atom offsets.
        """

        orbital_atom_index_parts = []
        orbital_role_parts = []
        active_slot_parts = []
        active_orbital_index_parts = []
        targets = []
        num_atoms_per_graph = []
        num_orbitals_per_graph = []

        atom_offset = 0
        orbital_offset = 0
        for sample in samples:
            num_atoms = int(sample.atom_numbers.shape[0])
            num_orbitals = int(sample.orbital_role.shape[0])

            shifted = sample.orbital_atom_index.copy()
            shifted[:, 0] = shifted[:, 0] + atom_offset
            shifted[:, 1] = shifted[:, 1] + atom_offset

            orbital_atom_index_parts.append(shifted)
            orbital_role_parts.append(sample.orbital_role)
            active_slot_parts.append(sample.active_slot_index)
            active_orbital_index_parts.append(sample.active_orbital_index + orbital_offset)
            targets.append(sample.target)
            num_atoms_per_graph.append(num_atoms)
            num_orbitals_per_graph.append(num_orbitals)
            atom_offset += num_atoms
            orbital_offset += num_orbitals

        return {
            "orbital_atom_index": jnp.asarray(
                jnp.concatenate([jnp.asarray(x, dtype=jnp.int32) for x in orbital_atom_index_parts], axis=0),
                dtype=jnp.int32,
            ),
            "orbital_role": jnp.asarray(
                jnp.concatenate([jnp.asarray(x, dtype=jnp.int32) for x in orbital_role_parts], axis=0),
                dtype=jnp.int32,
            ),
            "active_slot_index": jnp.asarray(
                jnp.concatenate([jnp.asarray(x, dtype=jnp.int32) for x in active_slot_parts], axis=0),
                dtype=jnp.int32,
            ),
            "active_orbital_index": jnp.asarray(
                jnp.concatenate([jnp.asarray(x, dtype=jnp.int32) for x in active_orbital_index_parts], axis=0),
                dtype=jnp.int32,
            ),
            "num_atoms_per_graph": jnp.asarray(num_atoms_per_graph, dtype=jnp.int32),
            "num_orbitals_per_graph": jnp.asarray(num_orbitals_per_graph, dtype=jnp.int32),
            "targets": jnp.asarray(targets, dtype=jnp.float32),
        }

    def buildLapPeMetadata(self, samples: List[UnifiedSample]) -> Dict[str, jnp.ndarray]:
        """
        Build batched Laplacian PE tensors.

        Returns:
        - lap_evals: Node-aligned repeated eigenvalues, shape [total_atoms, lap_pe_k].
        - lap_evecs: Concatenated eigenvectors, shape [total_atoms, lap_pe_k].
        """

        lap_eval_parts = []
        lap_evec_parts = []
        for sample in samples:
            num_atoms = int(sample.atom_numbers.shape[0])
            lap_evals = np.asarray(sample.lap_evals, dtype=np.float32)
            lap_evecs = np.asarray(sample.lap_evecs, dtype=np.float32)
            lap_eval_parts.append(repeatLapEvalsPerNode(num_nodes=num_atoms, lap_evals=lap_evals))
            lap_evec_parts.append(lap_evecs)

        lap_dim = 0 if len(lap_eval_parts) == 0 else int(lap_eval_parts[0].shape[1])
        return {
            "lap_evals": jnp.asarray(
                self.concatenateOrEmpty(lap_eval_parts, (lap_dim,), np.float32),
                dtype=jnp.float32,
            ),
            "lap_evecs": jnp.asarray(
                self.concatenateOrEmpty(lap_evec_parts, (lap_dim,), np.float32),
                dtype=jnp.float32,
            ),
        }

    def concatenateOrEmpty(self, arrays: Sequence[np.ndarray], shape_suffix: tuple[int, ...], dtype) -> np.ndarray:
        """
        Concatenate numpy arrays or create one empty fallback with a fixed suffix.
        """

        if len(arrays) == 0:
            return np.zeros((0,) + shape_suffix, dtype=dtype)
        return np.concatenate(arrays, axis=0).astype(dtype, copy=False)

    def padVector(self, value: np.ndarray, target_size: int, fill_value, dtype) -> np.ndarray:
        """
        Pad one 1D numpy array to a fixed length.
        """

        value = np.asarray(value, dtype=dtype)
        if int(value.shape[0]) >= int(target_size):
            return value
        pad_size = int(target_size) - int(value.shape[0])
        padding = np.full((pad_size,), fill_value, dtype=dtype)
        return np.concatenate([value, padding], axis=0)

    def padMatrix(
        self,
        value: np.ndarray,
        target_size: int,
        shape_suffix: tuple[int, ...],
        fill_value,
        dtype,
    ) -> np.ndarray:
        """
        Pad one leading-dimension matrix/tensor to a fixed length.
        """

        value = np.asarray(value, dtype=dtype)
        if int(value.shape[0]) >= int(target_size):
            return value
        pad_size = int(target_size) - int(value.shape[0])
        padding = np.full((pad_size,) + shape_suffix, fill_value, dtype=dtype)
        return np.concatenate([value, padding], axis=0)

    def roundUpToMultiple(self, value: int, multiple: int) -> int:
        """
        Round one positive integer up to the next multiple.
        """

        if multiple <= 0:
            raise ValueError("Bucket multiple must be positive.")
        return int(((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple))

    def buildFixedBucketSpec(
        self,
        chunks: Sequence[PackedMoleculeChunk],
        graph_step: int,
        static_atom_step: int,
        atom_step: int,
        atom_edge_step: int,
        orbital_step: int,
        rumer_edge_step: int,
        active_orbital_step: int,
        active_edge_step: int,
    ) -> FixedBucketSpec:
        """
        Build one coarse fixed-bucket target from the current real batch totals.
        """

        total_graphs = int(sum(int(chunk.atom_n_node.shape[0]) for chunk in chunks))
        total_static_atoms = int(sum(int(np.asarray(chunk.atom_numbers, dtype=np.int32).shape[0]) for chunk in chunks))
        total_atoms = int(sum(int(np.asarray(chunk.atom_n_node, dtype=np.int32).sum()) for chunk in chunks))
        total_atom_edges = int(sum(int(np.asarray(chunk.atom_n_edge, dtype=np.int32).sum()) for chunk in chunks))
        total_orbitals = int(sum(int(np.asarray(chunk.rumer_n_node, dtype=np.int32).sum()) for chunk in chunks))
        total_rumer_edges = int(sum(int(np.asarray(chunk.rumer_n_edge, dtype=np.int32).sum()) for chunk in chunks))
        total_active_orbitals = int(
            sum(
                int(
                    (
                        np.asarray(chunk.active_rumer_n_node, dtype=np.int32)
                        if hasattr(chunk, "active_rumer_n_node")
                        else self.deriveActiveFieldsFromPackedChunk(chunk)[4]
                    ).sum()
                )
                for chunk in chunks
            )
        )
        total_active_rumer_edges = int(
            sum(
                int(
                    (
                        np.asarray(chunk.active_rumer_n_edge, dtype=np.int32)
                        if hasattr(chunk, "active_rumer_n_edge")
                        else self.deriveActiveFieldsFromPackedChunk(chunk)[5]
                    ).sum()
                )
                for chunk in chunks
            )
        )

        return FixedBucketSpec(
            total_graphs=self.roundUpToMultiple(total_graphs + 1, graph_step),
            total_static_atoms=self.roundUpToMultiple(total_static_atoms + 1, static_atom_step),
            total_atoms=self.roundUpToMultiple(total_atoms + 1, atom_step),
            total_atom_edges=self.roundUpToMultiple(total_atom_edges, atom_edge_step),
            total_orbitals=self.roundUpToMultiple(total_orbitals + 1, orbital_step),
            total_rumer_edges=self.roundUpToMultiple(total_rumer_edges, rumer_edge_step),
            total_active_orbitals=self.roundUpToMultiple(total_active_orbitals + 1, active_orbital_step),
            total_active_rumer_edges=self.roundUpToMultiple(total_active_rumer_edges, active_edge_step),
        )

    def loadPackedChunk(self, chunk_ref: PackedChunkReference) -> PackedMoleculeChunk:
        """
        Load one packed chunk payload from disk on demand.
        """

        with open(chunk_ref.chunk_path, "rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, PackedMoleculeChunk):
            raise ValueError(f"Invalid packed chunk payload at {chunk_ref.chunk_path}.")
        return payload

    def deriveActiveFieldsFromPackedChunk(
        self,
        chunk: PackedMoleculeChunk,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Derive active-only orbital indices and active Rumer topology from full packed fields.

        This is used as a compatibility fallback for older packed caches that do not
        yet store the active-only fields explicitly.
        """

        active_orbital_index_parts = []
        active_rumer_senders_parts = []
        active_rumer_receivers_parts = []
        active_rumer_edge_type_parts = []
        active_rumer_n_node = []
        active_rumer_n_edge = []

        orbital_offset = 0
        edge_offset = 0
        active_offset = 0

        for num_orbitals, num_edges in zip(
            np.asarray(chunk.rumer_n_node, dtype=np.int32).tolist(),
            np.asarray(chunk.rumer_n_edge, dtype=np.int32).tolist(),
        ):
            role_slice = np.asarray(chunk.orbital_role[orbital_offset : orbital_offset + num_orbitals], dtype=np.int32)
            local_active_index = np.flatnonzero(role_slice == 2).astype(np.int32)
            active_orbital_index_parts.append(local_active_index + orbital_offset)
            active_rumer_n_node.append(int(local_active_index.shape[0]))

            active_local_map = {
                int(orbital_offset + local_full_index): local_active_offset
                for local_active_offset, local_full_index in enumerate(local_active_index.tolist())
            }

            edge_senders = np.asarray(
                chunk.rumer_senders[edge_offset : edge_offset + num_edges],
                dtype=np.int32,
            )
            edge_receivers = np.asarray(
                chunk.rumer_receivers[edge_offset : edge_offset + num_edges],
                dtype=np.int32,
            )
            edge_types = np.asarray(
                chunk.rumer_edge_type[edge_offset : edge_offset + num_edges],
                dtype=np.int32,
            )

            local_active_edge_count = 0
            for sender, receiver, edge_type in zip(
                edge_senders.tolist(),
                edge_receivers.tolist(),
                edge_types.tolist(),
            ):
                if edge_type != 2:
                    continue
                if (sender not in active_local_map) or (receiver not in active_local_map):
                    continue
                active_rumer_senders_parts.append(
                    np.asarray([active_local_map[sender] + active_offset], dtype=np.int32)
                )
                active_rumer_receivers_parts.append(
                    np.asarray([active_local_map[receiver] + active_offset], dtype=np.int32)
                )
                active_rumer_edge_type_parts.append(np.asarray([2], dtype=np.int32))
                local_active_edge_count += 1

            active_rumer_n_edge.append(local_active_edge_count)
            orbital_offset += num_orbitals
            edge_offset += num_edges
            active_offset += int(local_active_index.shape[0])

        return (
            self.concatenateOrEmpty(active_orbital_index_parts, tuple(), np.int32),
            self.concatenateOrEmpty(active_rumer_senders_parts, tuple(), np.int32),
            self.concatenateOrEmpty(active_rumer_receivers_parts, tuple(), np.int32),
            self.concatenateOrEmpty(active_rumer_edge_type_parts, tuple(), np.int32),
            np.asarray(active_rumer_n_node, dtype=np.int32),
            np.asarray(active_rumer_n_edge, dtype=np.int32),
        )

    def deriveLapPeFieldsFromPackedChunk(
        self,
        chunk: PackedMoleculeChunk,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Derive node-aligned Laplacian PE fields for older packed chunks.
        """

        lap_eval_parts = []
        lap_evec_parts = []
        atom_offset = 0
        edge_offset = 0
        lap_pe_k = int(getattr(chunk, "lap_pe_k", self.lap_pe_k))

        for num_atoms, num_edges in zip(
            np.asarray(chunk.atom_n_node, dtype=np.int32).tolist(),
            np.asarray(chunk.atom_n_edge, dtype=np.int32).tolist(),
        ):
            senders = np.asarray(chunk.atom_senders[edge_offset : edge_offset + num_edges], dtype=np.int32)
            receivers = np.asarray(chunk.atom_receivers[edge_offset : edge_offset + num_edges], dtype=np.int32)
            local_senders = senders - atom_offset
            local_receivers = receivers - atom_offset
            lap_evals, lap_evecs = lapPeFromEdges(
                num_nodes=int(num_atoms),
                senders=local_senders,
                receivers=local_receivers,
                k=lap_pe_k,
                eps=1.0e-12,
                add_self_loops=False,
            )
            lap_eval_parts.append(repeatLapEvalsPerNode(num_nodes=int(num_atoms), lap_evals=lap_evals))
            lap_evec_parts.append(lap_evecs)
            atom_offset += int(num_atoms)
            edge_offset += int(num_edges)

        return (
            self.concatenateOrEmpty(lap_eval_parts, (lap_pe_k,), np.float32),
            self.concatenateOrEmpty(lap_evec_parts, (lap_pe_k,), np.float32),
        )

    def chunkTopMassFocusMask(
        self,
        chunk: PackedMoleculeChunk,
    ) -> np.ndarray:
        """
        Return one chunk-local focus mask, defaulting to zeros for older caches.
        """

        top_mass_focus_mask = getattr(chunk, "top_mass_focus_mask", None)
        if top_mass_focus_mask is None:
            return np.zeros((int(chunk.atom_n_node.shape[0]),), dtype=np.float32)
        return np.asarray(top_mass_focus_mask, dtype=np.float32)

    def chunkDatasetId(
        self,
        chunk: PackedMoleculeChunk | PackedChunkReference,
    ) -> str:
        """
        Return the dataset id attached to one chunk, defaulting to ``default``.
        """

        return str(getattr(chunk, "dataset_id", "default"))

    def clonePackedChunkWithFocusMask(
        self,
        chunk: PackedMoleculeChunk,
        top_mass_focus_mask: np.ndarray,
    ) -> PackedMoleculeChunk:
        """
        Return one shallow copy of a packed chunk with an updated focus mask.
        """

        return replace(
            chunk,
            top_mass_focus_mask=np.asarray(top_mass_focus_mask, dtype=np.float32),
        )

    def subsetPackedChunk(
        self,
        chunk: PackedMoleculeChunk,
        selected_structure_index: np.ndarray,
        top_mass_focus_mask: np.ndarray | None = None,
    ) -> PackedMoleculeChunk:
        """
        Create one new packed chunk containing only the selected structures.
        """

        selected_structure_index = np.asarray(selected_structure_index, dtype=np.int32)
        if selected_structure_index.ndim != 1:
            raise ValueError("selected_structure_index must be one-dimensional.")
        if selected_structure_index.shape[0] == 0:
            raise ValueError("subsetPackedChunk requires at least one selected structure.")

        total_structures = int(chunk.atom_n_node.shape[0])
        if np.any(selected_structure_index < 0) or np.any(selected_structure_index >= total_structures):
            raise ValueError("selected_structure_index contains out-of-range entries.")
        if np.any(np.diff(selected_structure_index) < 0):
            selected_structure_index = np.sort(selected_structure_index)

        if top_mass_focus_mask is None:
            chunk_focus_mask = self.chunkTopMassFocusMask(chunk)
        else:
            chunk_focus_mask = np.asarray(top_mass_focus_mask, dtype=np.float32)
            if int(chunk_focus_mask.shape[0]) != total_structures:
                raise ValueError("top_mass_focus_mask length must match the number of structures in the chunk.")

        if np.array_equal(selected_structure_index, np.arange(total_structures, dtype=np.int32)):
            return self.clonePackedChunkWithFocusMask(
                chunk=chunk,
                top_mass_focus_mask=chunk_focus_mask,
            )

        if hasattr(chunk, "lap_evals") and hasattr(chunk, "lap_evecs"):
            chunk_lap_evals = np.asarray(chunk.lap_evals, dtype=np.float32)
            chunk_lap_evecs = np.asarray(chunk.lap_evecs, dtype=np.float32)
        else:
            chunk_lap_evals, chunk_lap_evecs = self.deriveLapPeFieldsFromPackedChunk(chunk)

        if hasattr(chunk, "active_orbital_index") and hasattr(chunk, "active_rumer_n_node"):
            chunk_active_orbital_index = np.asarray(chunk.active_orbital_index, dtype=np.int32)
            chunk_active_rumer_senders = np.asarray(chunk.active_rumer_senders, dtype=np.int32)
            chunk_active_rumer_receivers = np.asarray(chunk.active_rumer_receivers, dtype=np.int32)
            chunk_active_rumer_edge_type = np.asarray(chunk.active_rumer_edge_type, dtype=np.int32)
            chunk_active_rumer_n_node = np.asarray(chunk.active_rumer_n_node, dtype=np.int32)
            chunk_active_rumer_n_edge = np.asarray(chunk.active_rumer_n_edge, dtype=np.int32)
        else:
            (
                chunk_active_orbital_index,
                chunk_active_rumer_senders,
                chunk_active_rumer_receivers,
                chunk_active_rumer_edge_type,
                chunk_active_rumer_n_node,
                chunk_active_rumer_n_edge,
            ) = self.deriveActiveFieldsFromPackedChunk(chunk)

        selected_lookup = set(int(index) for index in selected_structure_index.tolist())
        atom_node_feature_parts = []
        atom_senders_parts = []
        atom_receivers_parts = []
        atom_pair_feature_parts = []
        lap_eval_parts = []
        lap_evec_parts = []
        orbital_atom_index_parts = []
        orbital_role_parts = []
        active_slot_index_parts = []
        rumer_senders_parts = []
        rumer_receivers_parts = []
        rumer_edge_type_parts = []
        atom_n_node = []
        atom_n_edge = []
        rumer_n_node = []
        rumer_n_edge = []
        active_orbital_index_parts = []
        active_rumer_senders_parts = []
        active_rumer_receivers_parts = []
        active_rumer_edge_type_parts = []
        active_rumer_n_node_parts = []
        active_rumer_n_edge_parts = []
        targets = []
        selected_focus_parts = []

        atom_cursor = 0
        atom_edge_cursor = 0
        orbital_cursor = 0
        rumer_edge_cursor = 0
        active_orbital_cursor = 0
        active_rumer_edge_cursor = 0
        new_atom_offset = 0
        new_orbital_offset = 0
        new_active_orbital_offset = 0

        for structure_index in range(total_structures):
            num_atoms = int(np.asarray(chunk.atom_n_node, dtype=np.int32)[structure_index])
            num_atom_edges = int(np.asarray(chunk.atom_n_edge, dtype=np.int32)[structure_index])
            num_orbitals = int(np.asarray(chunk.rumer_n_node, dtype=np.int32)[structure_index])
            num_rumer_edges = int(np.asarray(chunk.rumer_n_edge, dtype=np.int32)[structure_index])
            num_active_orbitals = int(chunk_active_rumer_n_node[structure_index])
            num_active_edges = int(chunk_active_rumer_n_edge[structure_index])

            atom_slice = slice(atom_cursor, atom_cursor + num_atoms)
            atom_edge_slice = slice(atom_edge_cursor, atom_edge_cursor + num_atom_edges)
            orbital_slice = slice(orbital_cursor, orbital_cursor + num_orbitals)
            rumer_edge_slice = slice(rumer_edge_cursor, rumer_edge_cursor + num_rumer_edges)
            active_orbital_slice = slice(active_orbital_cursor, active_orbital_cursor + num_active_orbitals)
            active_edge_slice = slice(active_rumer_edge_cursor, active_rumer_edge_cursor + num_active_edges)

            if structure_index in selected_lookup:
                atom_node_feature_parts.append(np.asarray(chunk.atom_node_features[atom_slice], dtype=np.float32))
                atom_senders_parts.append(
                    np.asarray(chunk.atom_senders[atom_edge_slice], dtype=np.int32) - atom_cursor + new_atom_offset
                )
                atom_receivers_parts.append(
                    np.asarray(chunk.atom_receivers[atom_edge_slice], dtype=np.int32) - atom_cursor + new_atom_offset
                )
                atom_pair_feature_parts.append(np.asarray(chunk.atom_pair_features[atom_edge_slice], dtype=np.float32))
                lap_eval_parts.append(np.asarray(chunk_lap_evals[atom_slice], dtype=np.float32))
                lap_evec_parts.append(np.asarray(chunk_lap_evecs[atom_slice], dtype=np.float32))

                shifted_orbital_atom_index = np.asarray(chunk.orbital_atom_index[orbital_slice], dtype=np.int32).copy()
                shifted_orbital_atom_index = shifted_orbital_atom_index - atom_cursor + new_atom_offset
                orbital_atom_index_parts.append(shifted_orbital_atom_index)
                orbital_role_parts.append(np.asarray(chunk.orbital_role[orbital_slice], dtype=np.int32))
                active_slot_index_parts.append(np.asarray(chunk.active_slot_index[orbital_slice], dtype=np.int32))

                rumer_senders_parts.append(
                    np.asarray(chunk.rumer_senders[rumer_edge_slice], dtype=np.int32)
                    - orbital_cursor
                    + new_orbital_offset
                )
                rumer_receivers_parts.append(
                    np.asarray(chunk.rumer_receivers[rumer_edge_slice], dtype=np.int32)
                    - orbital_cursor
                    + new_orbital_offset
                )
                rumer_edge_type_parts.append(np.asarray(chunk.rumer_edge_type[rumer_edge_slice], dtype=np.int32))

                active_orbital_index_parts.append(
                    np.asarray(chunk_active_orbital_index[active_orbital_slice], dtype=np.int32)
                    - orbital_cursor
                    + new_orbital_offset
                )
                active_rumer_senders_parts.append(
                    np.asarray(chunk_active_rumer_senders[active_edge_slice], dtype=np.int32)
                    - active_orbital_cursor
                    + new_active_orbital_offset
                )
                active_rumer_receivers_parts.append(
                    np.asarray(chunk_active_rumer_receivers[active_edge_slice], dtype=np.int32)
                    - active_orbital_cursor
                    + new_active_orbital_offset
                )
                active_rumer_edge_type_parts.append(
                    np.asarray(chunk_active_rumer_edge_type[active_edge_slice], dtype=np.int32)
                )

                atom_n_node.append(num_atoms)
                atom_n_edge.append(num_atom_edges)
                rumer_n_node.append(num_orbitals)
                rumer_n_edge.append(num_rumer_edges)
                active_rumer_n_node_parts.append(num_active_orbitals)
                active_rumer_n_edge_parts.append(num_active_edges)
                targets.append(float(np.asarray(chunk.targets, dtype=np.float32)[structure_index]))
                selected_focus_parts.append(float(chunk_focus_mask[structure_index]))

                new_atom_offset += num_atoms
                new_orbital_offset += num_orbitals
                new_active_orbital_offset += num_active_orbitals

            atom_cursor += num_atoms
            atom_edge_cursor += num_atom_edges
            orbital_cursor += num_orbitals
            rumer_edge_cursor += num_rumer_edges
            active_orbital_cursor += num_active_orbitals
            active_rumer_edge_cursor += num_active_edges

        lap_dim = 0 if len(lap_eval_parts) == 0 else int(lap_eval_parts[0].shape[1])
        return PackedMoleculeChunk(
            molecule_id=chunk.molecule_id,
            chunk_index=chunk.chunk_index,
            num_chunks=chunk.num_chunks,
            atom_numbers=np.asarray(chunk.atom_numbers, dtype=np.int32),
            atom_positions=np.asarray(chunk.atom_positions, dtype=np.float32),
            local_frame_e1=np.asarray(chunk.local_frame_e1, dtype=np.float32),
            local_frame_e2=np.asarray(chunk.local_frame_e2, dtype=np.float32),
            local_frame_e3=np.asarray(chunk.local_frame_e3, dtype=np.float32),
            atom_node_features=self.concatenateOrEmpty(atom_node_feature_parts, (3,), np.float32),
            atom_senders=self.concatenateOrEmpty(atom_senders_parts, tuple(), np.int32),
            atom_receivers=self.concatenateOrEmpty(atom_receivers_parts, tuple(), np.int32),
            atom_pair_features=self.concatenateOrEmpty(atom_pair_feature_parts, (2,), np.float32),
            lap_evals=self.concatenateOrEmpty(lap_eval_parts, (lap_dim,), np.float32),
            lap_evecs=self.concatenateOrEmpty(lap_evec_parts, (lap_dim,), np.float32),
            atom_n_node=np.asarray(atom_n_node, dtype=np.int32),
            atom_n_edge=np.asarray(atom_n_edge, dtype=np.int32),
            orbital_atom_index=self.concatenateOrEmpty(orbital_atom_index_parts, (2,), np.int32),
            orbital_role=self.concatenateOrEmpty(orbital_role_parts, tuple(), np.int32),
            active_slot_index=self.concatenateOrEmpty(active_slot_index_parts, tuple(), np.int32),
            rumer_senders=self.concatenateOrEmpty(rumer_senders_parts, tuple(), np.int32),
            rumer_receivers=self.concatenateOrEmpty(rumer_receivers_parts, tuple(), np.int32),
            rumer_edge_type=self.concatenateOrEmpty(rumer_edge_type_parts, tuple(), np.int32),
            rumer_n_node=np.asarray(rumer_n_node, dtype=np.int32),
            rumer_n_edge=np.asarray(rumer_n_edge, dtype=np.int32),
            active_orbital_index=self.concatenateOrEmpty(active_orbital_index_parts, tuple(), np.int32),
            active_rumer_senders=self.concatenateOrEmpty(active_rumer_senders_parts, tuple(), np.int32),
            active_rumer_receivers=self.concatenateOrEmpty(active_rumer_receivers_parts, tuple(), np.int32),
            active_rumer_edge_type=self.concatenateOrEmpty(active_rumer_edge_type_parts, tuple(), np.int32),
            active_rumer_n_node=np.asarray(active_rumer_n_node_parts, dtype=np.int32),
            active_rumer_n_edge=np.asarray(active_rumer_n_edge_parts, dtype=np.int32),
            targets=np.asarray(targets, dtype=np.float32),
            top_mass_focus_mask=np.asarray(selected_focus_parts, dtype=np.float32),
            dataset_id=self.chunkDatasetId(chunk),
        )

    def packBatch(
        self,
        samples: List[UnifiedSample],
        orbital_feature_dim: int,
        num_structures_per_molecule: Sequence[int] | None = None,
        dataset_index_per_molecule: Sequence[int] | None = None,
        local_frame_e1: np.ndarray | None = None,
        local_frame_e2: np.ndarray | None = None,
        local_frame_e3: np.ndarray | None = None,
    ) -> UnifiedBatch:
        """
        Pack one list of UnifiedSample objects into UnifiedBatch.
        """

        if num_structures_per_molecule is None:
            num_structures_per_molecule = [1 for _ in samples]
        if dataset_index_per_molecule is None:
            dataset_index_per_molecule = [0 for _ in num_structures_per_molecule]
        if (local_frame_e1 is None) or (local_frame_e2 is None) or (local_frame_e3 is None):
            from model.orbital_projection import LocalOrbitalProjector

            projector = LocalOrbitalProjector()
            positions = jnp.concatenate(
                [jnp.asarray(sample.atom_positions, dtype=jnp.float32) for sample in samples],
                axis=0,
            )
            atom_counts = jnp.asarray([sample.atom_numbers.shape[0] for sample in samples], dtype=jnp.int32)
            computed_e1, computed_e2, computed_e3 = projector.buildLocalFrames(positions, atom_counts)
            local_frame_e1 = np.asarray(computed_e1, dtype=np.float32)
            local_frame_e2 = np.asarray(computed_e2, dtype=np.float32)
            local_frame_e3 = np.asarray(computed_e3, dtype=np.float32)

        metadata = self.buildOrbitalMetadata(samples)
        lap_metadata = self.buildLapPeMetadata(samples)
        static_atom_numbers = jnp.concatenate(
            [jnp.asarray(sample.atom_numbers, dtype=jnp.int32) for sample in samples],
            axis=0,
        )
        static_atom_positions = jnp.concatenate(
            [jnp.asarray(sample.atom_positions, dtype=jnp.float32) for sample in samples],
            axis=0,
        )
        expanded_atom_to_static_atom_index = []
        static_atom_offset = 0
        for sample in samples:
            num_atoms = int(sample.atom_numbers.shape[0])
            expanded_atom_to_static_atom_index.append(
                jnp.arange(num_atoms, dtype=jnp.int32) + static_atom_offset
            )
            static_atom_offset += num_atoms
        return UnifiedBatch(
            atom_graph=self.buildAtomGraph(samples),
            rumer_graph=self.buildRumerGraph(samples, orbital_feature_dim),
            active_rumer_graph=self.buildActiveRumerGraph(samples, orbital_feature_dim),
            orbital_atom_index=metadata["orbital_atom_index"],
            orbital_role=metadata["orbital_role"],
            active_slot_index=metadata["active_slot_index"],
            active_orbital_index=metadata["active_orbital_index"],
            lap_evals=lap_metadata["lap_evals"],
            lap_evecs=lap_metadata["lap_evecs"],
            expanded_atom_to_static_atom_index=jnp.concatenate(
                expanded_atom_to_static_atom_index,
                axis=0,
            ),
            static_atom_numbers=static_atom_numbers,
            static_atom_positions=static_atom_positions,
            num_atoms_per_graph=metadata["num_atoms_per_graph"],
            num_orbitals_per_graph=metadata["num_orbitals_per_graph"],
            num_structures_per_molecule=jnp.asarray(num_structures_per_molecule, dtype=jnp.int32),
            dataset_index_per_molecule=jnp.asarray(dataset_index_per_molecule, dtype=jnp.int32),
            num_molecules_in_batch=int(len(num_structures_per_molecule)),
            local_frame_e1=jnp.asarray(local_frame_e1, dtype=jnp.float32),
            local_frame_e2=jnp.asarray(local_frame_e2, dtype=jnp.float32),
            local_frame_e3=jnp.asarray(local_frame_e3, dtype=jnp.float32),
            top_mass_focus_mask=jnp.asarray(
                [float(getattr(sample, "top_mass_focus", 0.0)) for sample in samples],
                dtype=jnp.float32,
            ),
            targets=metadata["targets"],
            sample_mask=jnp.ones_like(metadata["targets"], dtype=jnp.float32),
        )

    def packChunk(self, chunk: ProcessedMoleculeChunk) -> PackedMoleculeChunk:
        """
        Offline-pack one single-molecule chunk for direct train/eval loading.
        """

        atom_node_feature_parts = []
        atom_senders_parts = []
        atom_receivers_parts = []
        atom_pair_feature_parts = []
        lap_eval_parts = []
        lap_evec_parts = []
        orbital_atom_index_parts = []
        orbital_role_parts = []
        active_slot_index_parts = []
        rumer_senders_parts = []
        rumer_receivers_parts = []
        rumer_edge_type_parts = []
        atom_n_node = []
        atom_n_edge = []
        rumer_n_node = []
        rumer_n_edge = []
        active_orbital_index_parts = []
        active_rumer_senders_parts = []
        active_rumer_receivers_parts = []
        active_rumer_edge_type_parts = []
        active_rumer_n_node = []
        active_rumer_n_edge = []
        targets = []
        top_mass_focus_mask = []

        atom_offset = 0
        orbital_offset = 0
        active_orbital_offset = 0
        num_atoms = int(chunk.atom_numbers.shape[0])

        for structure in chunk.structures:
            num_orbitals = int(structure.orbital_role.shape[0])
            atom_node_feature_parts.append(np.asarray(structure.atom_node_features, dtype=np.float32))
            atom_senders_parts.append(np.asarray(structure.atom_senders, dtype=np.int32) + atom_offset)
            atom_receivers_parts.append(np.asarray(structure.atom_receivers, dtype=np.int32) + atom_offset)
            atom_pair_feature_parts.append(np.asarray(structure.atom_pair_features, dtype=np.float32))
            lap_eval_parts.append(
                repeatLapEvalsPerNode(
                    num_nodes=num_atoms,
                    lap_evals=np.asarray(structure.lap_evals, dtype=np.float32),
                )
                if hasattr(structure, "lap_evals")
                else repeatLapEvalsPerNode(
                    num_nodes=num_atoms,
                    lap_evals=lapPeFromEdges(
                        num_nodes=num_atoms,
                        senders=np.asarray(structure.atom_senders, dtype=np.int32),
                        receivers=np.asarray(structure.atom_receivers, dtype=np.int32),
                        k=self.lap_pe_k,
                        eps=1.0e-12,
                        add_self_loops=False,
                    )[0],
                )
            )
            lap_evec_parts.append(
                np.asarray(structure.lap_evecs, dtype=np.float32)
                if hasattr(structure, "lap_evecs")
                else lapPeFromEdges(
                    num_nodes=num_atoms,
                    senders=np.asarray(structure.atom_senders, dtype=np.int32),
                    receivers=np.asarray(structure.atom_receivers, dtype=np.int32),
                    k=self.lap_pe_k,
                    eps=1.0e-12,
                    add_self_loops=False,
                )[1]
            )

            shifted_orbital_atom_index = np.asarray(structure.orbital_atom_index, dtype=np.int32).copy()
            shifted_orbital_atom_index[:, 0] = shifted_orbital_atom_index[:, 0] + atom_offset
            shifted_orbital_atom_index[:, 1] = shifted_orbital_atom_index[:, 1] + atom_offset
            orbital_atom_index_parts.append(shifted_orbital_atom_index)
            orbital_role_parts.append(np.asarray(structure.orbital_role, dtype=np.int32))
            active_slot_index_parts.append(np.asarray(structure.active_slot_index, dtype=np.int32))

            rumer_senders_parts.append(np.asarray(structure.rumer_senders, dtype=np.int32) + orbital_offset)
            rumer_receivers_parts.append(np.asarray(structure.rumer_receivers, dtype=np.int32) + orbital_offset)
            rumer_edge_type_parts.append(np.asarray(structure.rumer_edge_type, dtype=np.int32))

            shifted_active_orbital_index = np.asarray(
                structure.active_orbital_index,
                dtype=np.int32,
            ) + orbital_offset
            active_orbital_index_parts.append(shifted_active_orbital_index)
            active_rumer_senders_parts.append(
                np.asarray(structure.active_rumer_senders, dtype=np.int32) + active_orbital_offset
            )
            active_rumer_receivers_parts.append(
                np.asarray(structure.active_rumer_receivers, dtype=np.int32) + active_orbital_offset
            )
            active_rumer_edge_type_parts.append(np.asarray(structure.active_rumer_edge_type, dtype=np.int32))

            atom_n_node.append(num_atoms)
            atom_n_edge.append(int(structure.atom_senders.shape[0]))
            rumer_n_node.append(num_orbitals)
            rumer_n_edge.append(int(structure.rumer_senders.shape[0]))
            active_rumer_n_node.append(int(structure.active_orbital_index.shape[0]))
            active_rumer_n_edge.append(int(structure.active_rumer_senders.shape[0]))
            targets.append(float(structure.target))
            top_mass_focus_mask.append(float(getattr(structure, "top_mass_focus", 0.0)))

            atom_offset += num_atoms
            orbital_offset += num_orbitals
            active_orbital_offset += int(structure.active_orbital_index.shape[0])

        return PackedMoleculeChunk(
            molecule_id=chunk.molecule_id,
            chunk_index=chunk.chunk_index,
            num_chunks=chunk.num_chunks,
            atom_numbers=np.asarray(chunk.atom_numbers, dtype=np.int32),
            atom_positions=np.asarray(chunk.atom_positions, dtype=np.float32),
            local_frame_e1=np.asarray(chunk.local_frame_e1, dtype=np.float32),
            local_frame_e2=np.asarray(chunk.local_frame_e2, dtype=np.float32),
            local_frame_e3=np.asarray(chunk.local_frame_e3, dtype=np.float32),
            atom_node_features=self.concatenateOrEmpty(atom_node_feature_parts, (3,), np.float32),
            atom_senders=self.concatenateOrEmpty(atom_senders_parts, tuple(), np.int32),
            atom_receivers=self.concatenateOrEmpty(atom_receivers_parts, tuple(), np.int32),
            atom_pair_features=self.concatenateOrEmpty(atom_pair_feature_parts, (2,), np.float32),
            lap_evals=self.concatenateOrEmpty(lap_eval_parts, (0 if len(lap_eval_parts) == 0 else lap_eval_parts[0].shape[1],), np.float32),
            lap_evecs=self.concatenateOrEmpty(lap_evec_parts, (0 if len(lap_evec_parts) == 0 else lap_evec_parts[0].shape[1],), np.float32),
            atom_n_node=np.asarray(atom_n_node, dtype=np.int32),
            atom_n_edge=np.asarray(atom_n_edge, dtype=np.int32),
            orbital_atom_index=self.concatenateOrEmpty(orbital_atom_index_parts, (2,), np.int32),
            orbital_role=self.concatenateOrEmpty(orbital_role_parts, tuple(), np.int32),
            active_slot_index=self.concatenateOrEmpty(active_slot_index_parts, tuple(), np.int32),
            rumer_senders=self.concatenateOrEmpty(rumer_senders_parts, tuple(), np.int32),
            rumer_receivers=self.concatenateOrEmpty(rumer_receivers_parts, tuple(), np.int32),
            rumer_edge_type=self.concatenateOrEmpty(rumer_edge_type_parts, tuple(), np.int32),
            rumer_n_node=np.asarray(rumer_n_node, dtype=np.int32),
            rumer_n_edge=np.asarray(rumer_n_edge, dtype=np.int32),
            active_orbital_index=self.concatenateOrEmpty(active_orbital_index_parts, tuple(), np.int32),
            active_rumer_senders=self.concatenateOrEmpty(active_rumer_senders_parts, tuple(), np.int32),
            active_rumer_receivers=self.concatenateOrEmpty(active_rumer_receivers_parts, tuple(), np.int32),
            active_rumer_edge_type=self.concatenateOrEmpty(active_rumer_edge_type_parts, tuple(), np.int32),
            active_rumer_n_node=np.asarray(active_rumer_n_node, dtype=np.int32),
            active_rumer_n_edge=np.asarray(active_rumer_n_edge, dtype=np.int32),
            targets=np.asarray(targets, dtype=np.float32),
            top_mass_focus_mask=np.asarray(top_mass_focus_mask, dtype=np.float32),
            dataset_id=getattr(chunk, "dataset_id", "default"),
        )

    def packPackedChunkBatch(
        self,
        chunks: List[PackedMoleculeChunk],
        orbital_feature_dim: int,
        fixed_bucket_spec: FixedBucketSpec | None = None,
        molecule_batch_size_target: int | None = None,
        num_structures_per_molecule: Sequence[int] | None = None,
        num_molecules_in_batch: int | None = None,
        dataset_index_per_molecule: Sequence[int] | None = None,
        top_mass_focus_mask: np.ndarray | None = None,
    ) -> UnifiedBatch:
        """
        Combine offline-packed single-molecule chunks into one runtime UnifiedBatch.
        """

        atom_feature_parts = []
        expanded_atom_to_static_atom_index_parts = []
        static_atom_number_parts = []
        static_atom_position_parts = []
        atom_pair_feature_parts = []
        lap_eval_parts = []
        lap_evec_parts = []
        atom_senders_parts = []
        atom_receivers_parts = []
        atom_n_node_parts = []
        atom_n_edge_parts = []
        orbital_atom_index_parts = []
        orbital_role_parts = []
        active_slot_index_parts = []
        rumer_senders_parts = []
        rumer_receivers_parts = []
        rumer_edge_type_parts = []
        rumer_n_node_parts = []
        rumer_n_edge_parts = []
        active_orbital_index_parts = []
        active_rumer_senders_parts = []
        active_rumer_receivers_parts = []
        active_rumer_edge_type_parts = []
        active_rumer_n_node_parts = []
        active_rumer_n_edge_parts = []
        local_frame_e1_parts = []
        local_frame_e2_parts = []
        local_frame_e3_parts = []
        targets = []
        top_mass_focus_parts = []

        atom_offset = 0
        static_atom_offset = 0
        orbital_offset = 0
        active_orbital_offset = 0
        for chunk in chunks:
            num_structures = int(chunk.atom_n_node.shape[0])
            num_static_atoms = int(np.asarray(chunk.atom_numbers, dtype=np.int32).shape[0])
            total_chunk_atoms = int(chunk.atom_n_node.sum())
            total_chunk_orbitals = int(chunk.rumer_n_node.sum())

            atom_feature_parts.append(np.asarray(chunk.atom_node_features, dtype=np.float32))
            expanded_atom_to_static_atom_index_parts.append(
                np.tile(
                    np.arange(num_static_atoms, dtype=np.int32) + static_atom_offset,
                    num_structures,
                )
            )
            static_atom_number_parts.append(np.asarray(chunk.atom_numbers, dtype=np.int32))
            static_atom_position_parts.append(np.asarray(chunk.atom_positions, dtype=np.float32))
            atom_pair_feature_parts.append(np.asarray(chunk.atom_pair_features, dtype=np.float32))
            if hasattr(chunk, "lap_evals") and hasattr(chunk, "lap_evecs"):
                lap_eval_parts.append(np.asarray(chunk.lap_evals, dtype=np.float32))
                lap_evec_parts.append(np.asarray(chunk.lap_evecs, dtype=np.float32))
            else:
                chunk_lap_evals, chunk_lap_evecs = self.deriveLapPeFieldsFromPackedChunk(chunk)
                lap_eval_parts.append(chunk_lap_evals)
                lap_evec_parts.append(chunk_lap_evecs)
            atom_senders_parts.append(np.asarray(chunk.atom_senders, dtype=np.int32) + atom_offset)
            atom_receivers_parts.append(np.asarray(chunk.atom_receivers, dtype=np.int32) + atom_offset)
            atom_n_node_parts.append(np.asarray(chunk.atom_n_node, dtype=np.int32))
            atom_n_edge_parts.append(np.asarray(chunk.atom_n_edge, dtype=np.int32))

            orbital_atom_index_parts.append(np.asarray(chunk.orbital_atom_index, dtype=np.int32) + atom_offset)
            orbital_role_parts.append(np.asarray(chunk.orbital_role, dtype=np.int32))
            active_slot_index_parts.append(np.asarray(chunk.active_slot_index, dtype=np.int32))

            rumer_senders_parts.append(np.asarray(chunk.rumer_senders, dtype=np.int32) + orbital_offset)
            rumer_receivers_parts.append(np.asarray(chunk.rumer_receivers, dtype=np.int32) + orbital_offset)
            rumer_edge_type_parts.append(np.asarray(chunk.rumer_edge_type, dtype=np.int32))
            rumer_n_node_parts.append(np.asarray(chunk.rumer_n_node, dtype=np.int32))
            rumer_n_edge_parts.append(np.asarray(chunk.rumer_n_edge, dtype=np.int32))

            if hasattr(chunk, "active_orbital_index") and hasattr(chunk, "active_rumer_n_node"):
                chunk_active_orbital_index = np.asarray(chunk.active_orbital_index, dtype=np.int32)
                chunk_active_rumer_senders = np.asarray(chunk.active_rumer_senders, dtype=np.int32)
                chunk_active_rumer_receivers = np.asarray(chunk.active_rumer_receivers, dtype=np.int32)
                chunk_active_rumer_edge_type = np.asarray(chunk.active_rumer_edge_type, dtype=np.int32)
                chunk_active_rumer_n_node = np.asarray(chunk.active_rumer_n_node, dtype=np.int32)
                chunk_active_rumer_n_edge = np.asarray(chunk.active_rumer_n_edge, dtype=np.int32)
            else:
                (
                    chunk_active_orbital_index,
                    chunk_active_rumer_senders,
                    chunk_active_rumer_receivers,
                    chunk_active_rumer_edge_type,
                    chunk_active_rumer_n_node,
                    chunk_active_rumer_n_edge,
                ) = self.deriveActiveFieldsFromPackedChunk(chunk)

            active_orbital_index_parts.append(chunk_active_orbital_index + orbital_offset)
            active_rumer_senders_parts.append(chunk_active_rumer_senders + active_orbital_offset)
            active_rumer_receivers_parts.append(chunk_active_rumer_receivers + active_orbital_offset)
            active_rumer_edge_type_parts.append(chunk_active_rumer_edge_type)
            active_rumer_n_node_parts.append(chunk_active_rumer_n_node)
            active_rumer_n_edge_parts.append(chunk_active_rumer_n_edge)

            local_frame_e1_parts.append(np.asarray(chunk.local_frame_e1, dtype=np.float32))
            local_frame_e2_parts.append(np.asarray(chunk.local_frame_e2, dtype=np.float32))
            local_frame_e3_parts.append(np.asarray(chunk.local_frame_e3, dtype=np.float32))
            targets.append(np.asarray(chunk.targets, dtype=np.float32))
            top_mass_focus_parts.append(self.chunkTopMassFocusMask(chunk))

            atom_offset += total_chunk_atoms
            static_atom_offset += num_static_atoms
            orbital_offset += total_chunk_orbitals
            active_orbital_offset += int(np.asarray(chunk_active_rumer_n_node, dtype=np.int32).sum())

        atom_feature = self.concatenateOrEmpty(atom_feature_parts, (3,), np.float32)
        expanded_atom_to_static_atom_index = self.concatenateOrEmpty(
            expanded_atom_to_static_atom_index_parts,
            tuple(),
            np.int32,
        )
        static_atom_numbers = self.concatenateOrEmpty(static_atom_number_parts, tuple(), np.int32)
        static_atom_positions = self.concatenateOrEmpty(static_atom_position_parts, (3,), np.float32)
        atom_pair_feature = self.concatenateOrEmpty(atom_pair_feature_parts, (2,), np.float32)
        lap_dim = 0 if len(lap_eval_parts) == 0 else int(lap_eval_parts[0].shape[1])
        lap_evals = self.concatenateOrEmpty(lap_eval_parts, (lap_dim,), np.float32)
        lap_evecs = self.concatenateOrEmpty(lap_evec_parts, (lap_dim,), np.float32)
        atom_senders = self.concatenateOrEmpty(atom_senders_parts, tuple(), np.int32)
        atom_receivers = self.concatenateOrEmpty(atom_receivers_parts, tuple(), np.int32)
        atom_n_node = self.concatenateOrEmpty(atom_n_node_parts, tuple(), np.int32)
        atom_n_edge = self.concatenateOrEmpty(atom_n_edge_parts, tuple(), np.int32)
        orbital_atom_index = self.concatenateOrEmpty(orbital_atom_index_parts, (2,), np.int32)
        orbital_role = self.concatenateOrEmpty(orbital_role_parts, tuple(), np.int32)
        active_slot_index = self.concatenateOrEmpty(active_slot_index_parts, tuple(), np.int32)
        rumer_senders = self.concatenateOrEmpty(rumer_senders_parts, tuple(), np.int32)
        rumer_receivers = self.concatenateOrEmpty(rumer_receivers_parts, tuple(), np.int32)
        rumer_edge_type = self.concatenateOrEmpty(rumer_edge_type_parts, tuple(), np.int32)
        rumer_n_node = self.concatenateOrEmpty(rumer_n_node_parts, tuple(), np.int32)
        rumer_n_edge = self.concatenateOrEmpty(rumer_n_edge_parts, tuple(), np.int32)
        active_orbital_index = self.concatenateOrEmpty(active_orbital_index_parts, tuple(), np.int32)
        active_rumer_senders = self.concatenateOrEmpty(active_rumer_senders_parts, tuple(), np.int32)
        active_rumer_receivers = self.concatenateOrEmpty(active_rumer_receivers_parts, tuple(), np.int32)
        active_rumer_edge_type = self.concatenateOrEmpty(active_rumer_edge_type_parts, tuple(), np.int32)
        active_rumer_n_node = self.concatenateOrEmpty(active_rumer_n_node_parts, tuple(), np.int32)
        active_rumer_n_edge = self.concatenateOrEmpty(active_rumer_n_edge_parts, tuple(), np.int32)
        local_frame_e1 = self.concatenateOrEmpty(local_frame_e1_parts, (3,), np.float32)
        local_frame_e2 = self.concatenateOrEmpty(local_frame_e2_parts, (3,), np.float32)
        local_frame_e3 = self.concatenateOrEmpty(local_frame_e3_parts, (3,), np.float32)
        target = self.concatenateOrEmpty(targets, tuple(), np.float32)
        top_mass_focus = (
            np.asarray(top_mass_focus_mask, dtype=np.float32)
            if top_mass_focus_mask is not None
            else self.concatenateOrEmpty(top_mass_focus_parts, tuple(), np.float32)
        )
        if num_structures_per_molecule is None:
            num_structures_per_molecule = np.asarray(
                [int(chunk.atom_n_node.shape[0]) for chunk in chunks],
                dtype=np.int32,
            )
        else:
            num_structures_per_molecule = np.asarray(num_structures_per_molecule, dtype=np.int32)
        if dataset_index_per_molecule is None:
            dataset_index_per_molecule = np.asarray(
                [0 for _ in range(int(num_structures_per_molecule.shape[0]))],
                dtype=np.int32,
            )
        else:
            dataset_index_per_molecule = np.asarray(dataset_index_per_molecule, dtype=np.int32)
        sample_mask = np.ones_like(target, dtype=np.float32)

        if fixed_bucket_spec is not None:
            real_graph_count = int(target.shape[0])
            real_static_atoms = int(static_atom_numbers.shape[0])
            real_atoms = int(atom_feature.shape[0])
            real_atom_edges = int(atom_senders.shape[0])
            real_orbitals = int(orbital_role.shape[0])
            real_rumer_edges = int(rumer_senders.shape[0])
            real_active_orbitals = int(active_orbital_index.shape[0])
            real_active_edges = int(active_rumer_senders.shape[0])

            graph_pad = int(fixed_bucket_spec.total_graphs) - real_graph_count
            atom_pad = int(fixed_bucket_spec.total_atoms) - real_atoms
            static_atom_pad = int(fixed_bucket_spec.total_static_atoms) - real_static_atoms
            atom_edge_pad = int(fixed_bucket_spec.total_atom_edges) - real_atom_edges
            orbital_pad = int(fixed_bucket_spec.total_orbitals) - real_orbitals
            rumer_edge_pad = int(fixed_bucket_spec.total_rumer_edges) - real_rumer_edges
            active_orbital_pad = int(fixed_bucket_spec.total_active_orbitals) - real_active_orbitals
            active_edge_pad = int(fixed_bucket_spec.total_active_rumer_edges) - real_active_edges
            if min(
                graph_pad,
                atom_pad,
                static_atom_pad,
                atom_edge_pad,
                orbital_pad,
                rumer_edge_pad,
                active_orbital_pad,
                active_edge_pad,
            ) < 0:
                raise ValueError("Fixed bucket target is smaller than the real batch.")

            padded_structure_count = real_graph_count + graph_pad
            sample_mask = self.padVector(sample_mask, padded_structure_count, 0.0, np.float32)
            target = self.padVector(target, padded_structure_count, 0.0, np.float32)
            top_mass_focus = self.padVector(top_mass_focus, padded_structure_count, 0.0, np.float32)

            atom_feature = self.padMatrix(atom_feature, fixed_bucket_spec.total_atoms, (3,), 0.0, np.float32)
            lap_evals = self.padMatrix(lap_evals, fixed_bucket_spec.total_atoms, (lap_dim,), 0.0, np.float32)
            lap_evecs = self.padMatrix(lap_evecs, fixed_bucket_spec.total_atoms, (lap_dim,), 0.0, np.float32)
            atom_pair_feature = self.padMatrix(
                atom_pair_feature,
                fixed_bucket_spec.total_atom_edges,
                (2,),
                0.0,
                np.float32,
            )
            static_atom_numbers = self.padVector(
                static_atom_numbers,
                fixed_bucket_spec.total_static_atoms,
                0,
                np.int32,
            )
            static_atom_positions = self.padMatrix(
                static_atom_positions,
                fixed_bucket_spec.total_static_atoms,
                (3,),
                0.0,
                np.float32,
            )
            local_frame_e1 = self.padMatrix(local_frame_e1, fixed_bucket_spec.total_static_atoms, (3,), 0.0, np.float32)
            local_frame_e2 = self.padMatrix(local_frame_e2, fixed_bucket_spec.total_static_atoms, (3,), 0.0, np.float32)
            local_frame_e3 = self.padMatrix(local_frame_e3, fixed_bucket_spec.total_static_atoms, (3,), 0.0, np.float32)

            dummy_static_atom_index = real_static_atoms
            expanded_atom_to_static_atom_index = self.padVector(
                expanded_atom_to_static_atom_index,
                fixed_bucket_spec.total_atoms,
                dummy_static_atom_index,
                np.int32,
            )

            dummy_atom_index = real_atoms
            atom_senders = self.padVector(atom_senders, fixed_bucket_spec.total_atom_edges, dummy_atom_index, np.int32)
            atom_receivers = self.padVector(
                atom_receivers,
                fixed_bucket_spec.total_atom_edges,
                dummy_atom_index,
                np.int32,
            )

            dummy_orbital_index = real_orbitals
            orbital_atom_padding = np.full((orbital_pad, 2), dummy_static_atom_index, dtype=np.int32)
            if orbital_pad > 0:
                orbital_atom_index = np.concatenate([orbital_atom_index, orbital_atom_padding], axis=0)
            orbital_role = self.padVector(orbital_role, fixed_bucket_spec.total_orbitals, 0, np.int32)
            active_slot_index = self.padVector(active_slot_index, fixed_bucket_spec.total_orbitals, -1, np.int32)
            rumer_senders = self.padVector(rumer_senders, fixed_bucket_spec.total_rumer_edges, dummy_orbital_index, np.int32)
            rumer_receivers = self.padVector(
                rumer_receivers,
                fixed_bucket_spec.total_rumer_edges,
                dummy_orbital_index,
                np.int32,
            )
            rumer_edge_type = self.padVector(rumer_edge_type, fixed_bucket_spec.total_rumer_edges, 0, np.int32)

            dummy_active_orbital_index = real_active_orbitals
            active_orbital_index = self.padVector(
                active_orbital_index,
                fixed_bucket_spec.total_active_orbitals,
                dummy_orbital_index,
                np.int32,
            )
            active_rumer_senders = self.padVector(
                active_rumer_senders,
                fixed_bucket_spec.total_active_rumer_edges,
                dummy_active_orbital_index,
                np.int32,
            )
            active_rumer_receivers = self.padVector(
                active_rumer_receivers,
                fixed_bucket_spec.total_active_rumer_edges,
                dummy_active_orbital_index,
                np.int32,
            )
            active_rumer_edge_type = self.padVector(
                active_rumer_edge_type,
                fixed_bucket_spec.total_active_rumer_edges,
                0,
                np.int32,
            )

            atom_n_node = self.padVector(atom_n_node, padded_structure_count, 0, np.int32)
            atom_n_edge = self.padVector(atom_n_edge, padded_structure_count, 0, np.int32)
            rumer_n_node = self.padVector(rumer_n_node, padded_structure_count, 0, np.int32)
            rumer_n_edge = self.padVector(rumer_n_edge, padded_structure_count, 0, np.int32)
            active_rumer_n_node = self.padVector(active_rumer_n_node, padded_structure_count, 0, np.int32)
            active_rumer_n_edge = self.padVector(active_rumer_n_edge, padded_structure_count, 0, np.int32)
            if graph_pad > 0:
                atom_n_node[real_graph_count] = atom_pad
                atom_n_edge[real_graph_count] = atom_edge_pad
                rumer_n_node[real_graph_count] = orbital_pad
                rumer_n_edge[real_graph_count] = rumer_edge_pad
                active_rumer_n_node[real_graph_count] = active_orbital_pad
                active_rumer_n_edge[real_graph_count] = active_edge_pad

            structure_count_target = (
                int(molecule_batch_size_target)
                if molecule_batch_size_target is not None
                else int(len(chunks))
            )
            num_structures_per_molecule = self.padVector(
                num_structures_per_molecule,
                structure_count_target,
                0,
                np.int32,
            )
            dataset_index_per_molecule = self.padVector(
                dataset_index_per_molecule,
                structure_count_target,
                -1,
                np.int32,
            )
            if num_structures_per_molecule.shape[0] > 0:
                num_structures_per_molecule[-1] = int(num_structures_per_molecule[-1]) + graph_pad

        atom_graph = jraph.GraphsTuple(
            nodes={"features": jnp.asarray(atom_feature, dtype=jnp.float32)},
            edges={"pair": jnp.asarray(atom_pair_feature, dtype=jnp.float32)},
            senders=jnp.asarray(atom_senders, dtype=jnp.int32),
            receivers=jnp.asarray(atom_receivers, dtype=jnp.int32),
            n_node=jnp.asarray(atom_n_node, dtype=jnp.int32),
            n_edge=jnp.asarray(atom_n_edge, dtype=jnp.int32),
            globals=None,
        )
        rumer_graph = jraph.GraphsTuple(
            nodes=jnp.zeros((int(orbital_role.shape[0]), orbital_feature_dim), dtype=jnp.float32),
            edges={"edge_type": jnp.asarray(rumer_edge_type, dtype=jnp.int32)},
            senders=jnp.asarray(rumer_senders, dtype=jnp.int32),
            receivers=jnp.asarray(rumer_receivers, dtype=jnp.int32),
            n_node=jnp.asarray(rumer_n_node, dtype=jnp.int32),
            n_edge=jnp.asarray(rumer_n_edge, dtype=jnp.int32),
            globals=None,
        )
        active_rumer_graph = jraph.GraphsTuple(
            nodes=jnp.zeros((int(active_orbital_index.shape[0]), orbital_feature_dim), dtype=jnp.float32),
            edges={"edge_type": jnp.asarray(active_rumer_edge_type, dtype=jnp.int32)},
            senders=jnp.asarray(active_rumer_senders, dtype=jnp.int32),
            receivers=jnp.asarray(active_rumer_receivers, dtype=jnp.int32),
            n_node=jnp.asarray(active_rumer_n_node, dtype=jnp.int32),
            n_edge=jnp.asarray(active_rumer_n_edge, dtype=jnp.int32),
            globals=None,
        )
        return UnifiedBatch(
            atom_graph=atom_graph,
            rumer_graph=rumer_graph,
            active_rumer_graph=active_rumer_graph,
            orbital_atom_index=jnp.asarray(orbital_atom_index, dtype=jnp.int32),
            orbital_role=jnp.asarray(orbital_role, dtype=jnp.int32),
            active_slot_index=jnp.asarray(active_slot_index, dtype=jnp.int32),
            active_orbital_index=jnp.asarray(active_orbital_index, dtype=jnp.int32),
            lap_evals=jnp.asarray(lap_evals, dtype=jnp.float32),
            lap_evecs=jnp.asarray(lap_evecs, dtype=jnp.float32),
            expanded_atom_to_static_atom_index=jnp.asarray(
                expanded_atom_to_static_atom_index,
                dtype=jnp.int32,
            ),
            static_atom_numbers=jnp.asarray(static_atom_numbers, dtype=jnp.int32),
            static_atom_positions=jnp.asarray(static_atom_positions, dtype=jnp.float32),
            num_atoms_per_graph=jnp.asarray(atom_n_node, dtype=jnp.int32),
            num_orbitals_per_graph=jnp.asarray(rumer_n_node, dtype=jnp.int32),
            num_structures_per_molecule=jnp.asarray(num_structures_per_molecule, dtype=jnp.int32),
            dataset_index_per_molecule=jnp.asarray(dataset_index_per_molecule, dtype=jnp.int32),
            num_molecules_in_batch=(
                int(num_molecules_in_batch)
                if num_molecules_in_batch is not None
                else int(len(chunks))
            ),
            local_frame_e1=jnp.asarray(local_frame_e1, dtype=jnp.float32),
            local_frame_e2=jnp.asarray(local_frame_e2, dtype=jnp.float32),
            local_frame_e3=jnp.asarray(local_frame_e3, dtype=jnp.float32),
            top_mass_focus_mask=jnp.asarray(top_mass_focus, dtype=jnp.float32),
            targets=jnp.asarray(target, dtype=jnp.float32),
            sample_mask=jnp.asarray(sample_mask, dtype=jnp.float32),
        )


class GrainPipeline:
    """
    Build Grain iterators that emit UnifiedBatch items.
    """

    def __init__(
        self,
        adapter: GraphPackingAdapter,
        orbital_feature_dim: int,
        dataset_ids: Sequence[str] | None = None,
        iterator_log_interval: int = 1,
    ):
        self.adapter = adapter
        self.orbital_feature_dim = orbital_feature_dim
        self.dataset_ids = list(dataset_ids) if dataset_ids is not None else ["default"]
        self.dataset_index_by_id = {
            dataset_id: index for index, dataset_id in enumerate(self.dataset_ids)
        }
        self.iterator_log_interval = max(1, int(iterator_log_interval))

    def currentRssMb(self) -> float:
        """
        Read current process resident-set size in MiB from /proc.
        """

        try:
            with open(f"/proc/{os.getpid()}/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        value_kb = int(line.split()[1])
                        return float(value_kb) / 1024.0
        except OSError:
            return -1.0
        return -1.0

    def logIteratorStage(
        self,
        split_name: str,
        batch_id: int,
        stage: str,
        extra: str = "",
    ) -> None:
        """
        Emit one lightweight iterator-stage diagnostic line.
        """

        if (batch_id % self.iterator_log_interval) != 0:
            return
        suffix = f" {extra}" if extra else ""
        print(
            f"ITERATOR_STAGE split={split_name} batch={batch_id:05d} stage={stage} "
            f"rss_mb={self.currentRssMb():.1f}{suffix}",
            flush=True,
        )

    def chunkGroupDatasetId(
        self,
        chunk_group: Sequence[PackedMoleculeChunk | PackedChunkReference],
    ) -> str:
        """
        Return the dataset id attached to one whole-molecule chunk group.
        """

        if len(chunk_group) == 0:
            return "default"
        return self.adapter.chunkDatasetId(chunk_group[0])

    def buildDatasetBalancedGroupedBatchIndices(
        self,
        chunk_groups: Sequence[Sequence[PackedMoleculeChunk | PackedChunkReference]],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_key: str | None,
        drop_remainder: bool,
    ) -> List[List[int]]:
        """
        Build molecule-group batches by interleaving dataset-local batches.
        """

        if len(chunk_groups) == 0:
            return []

        dataset_to_indices: dict[str, list[int]] = defaultdict(list)
        for index, chunk_group in enumerate(chunk_groups):
            dataset_to_indices[self.chunkGroupDatasetId(chunk_group)].append(index)

        rng = np.random.default_rng(seed)
        dataset_to_batches: dict[str, list[list[int]]] = {}
        for dataset_id, indices in dataset_to_indices.items():
            subset_groups = [chunk_groups[index] for index in indices]
            if bucket_key is not None:
                local_batches = self.buildGroupedBatchIndices(
                    chunk_groups=subset_groups,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    seed=seed + int(self.dataset_index_by_id.get(dataset_id, 0)),
                    bucket_key=bucket_key,
                    drop_remainder=drop_remainder,
                )
                dataset_to_batches[dataset_id] = [
                    [indices[local_index] for local_index in batch]
                    for batch in local_batches
                ]
                continue

            ordered_local_indices = list(range(len(indices)))
            if shuffle:
                rng.shuffle(ordered_local_indices)
            local_batches: list[list[int]] = []
            for start in range(0, len(ordered_local_indices), batch_size):
                batch = ordered_local_indices[start : start + batch_size]
                if drop_remainder and len(batch) < batch_size:
                    continue
                local_batches.append([indices[local_index] for local_index in batch])
            dataset_to_batches[dataset_id] = local_batches

        dataset_order = [
            dataset_id for dataset_id, batches in dataset_to_batches.items() if len(batches) > 0
        ]
        if shuffle and len(dataset_order) > 1:
            rng.shuffle(dataset_order)
        dataset_cursor = {dataset_id: 0 for dataset_id in dataset_order}
        balanced_batches: list[list[int]] = []
        while True:
            active_datasets = [
                dataset_id
                for dataset_id in dataset_order
                if dataset_cursor[dataset_id] < len(dataset_to_batches[dataset_id])
            ]
            if len(active_datasets) == 0:
                break
            if shuffle and len(active_datasets) > 1:
                rng.shuffle(active_datasets)
            for dataset_id in active_datasets:
                balanced_batches.append(dataset_to_batches[dataset_id][dataset_cursor[dataset_id]])
                dataset_cursor[dataset_id] += 1
        return balanced_batches

    def chunkBucketKey(
        self,
        chunk: ProcessedMoleculeChunk | PackedMoleculeChunk | PackedChunkReference,
        bucket_key: str,
    ) -> Tuple[int, ...]:
        """
        Compute one size key for a single-molecule chunk.
        """

        if isinstance(chunk, PackedMoleculeChunk):
            total_atoms = int(np.asarray(chunk.atom_n_node, dtype=np.int32).sum())
            total_orbitals = int(np.asarray(chunk.rumer_n_node, dtype=np.int32).sum())
            num_structures = int(chunk.atom_n_node.shape[0])
        elif isinstance(chunk, PackedChunkReference):
            total_atoms = int(chunk.total_atoms)
            total_orbitals = int(chunk.total_orbitals)
            num_structures = int(chunk.num_structures)
        else:
            total_atoms = int(chunk.atom_numbers.shape[0]) * len(chunk.structures)
            total_orbitals = sum(int(structure.orbital_role.shape[0]) for structure in chunk.structures)
            num_structures = len(chunk.structures)

        if bucket_key == "num_atoms":
            return (total_atoms, num_structures)
        if bucket_key in {"num_atoms_num_orbitals", "(num_atoms,num_orbitals)"}:
            return (total_atoms, total_orbitals, num_structures)
        raise ValueError(f"Unsupported bucket_key={bucket_key}.")

    def chunkCoarseBucketId(
        self,
        chunk: ProcessedMoleculeChunk | PackedMoleculeChunk | PackedChunkReference,
        bucket_key: str,
    ) -> Tuple[int, ...]:
        """
        Map one chunk to a coarse bucket so batches are formed within tighter size bands.
        """

        raw_key = self.chunkBucketKey(chunk, bucket_key)
        if bucket_key == "num_atoms":
            total_atoms, num_structures = raw_key
            return (
                int(np.ceil(total_atoms / 64.0)),
                int(np.ceil(num_structures / 4.0)),
            )
        if bucket_key in {"num_atoms_num_orbitals", "(num_atoms,num_orbitals)"}:
            total_atoms, total_orbitals, num_structures = raw_key
            return (
                int(np.ceil(total_atoms / 64.0)),
                int(np.ceil(total_orbitals / 32.0)),
                int(np.ceil(num_structures / 4.0)),
            )
        raise ValueError(f"Unsupported bucket_key={bucket_key}.")

    def reorderChunksForBucketing(
        self,
        chunks: Sequence[PackedMoleculeChunk | PackedChunkReference],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_key: str,
        drop_remainder: bool = False,
    ) -> List[PackedMoleculeChunk | PackedChunkReference]:
        """
        Reorder chunks according to bucketed batch construction.
        """

        batch_indices = self.buildBucketedBatchIndices(
            chunks=chunks,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            bucket_key=bucket_key,
            drop_remainder=drop_remainder,
        )
        return [chunks[index] for batch in batch_indices for index in batch]

    def groupBucketKey(
        self,
        chunk_group: Sequence[PackedMoleculeChunk | PackedChunkReference],
        bucket_key: str,
    ) -> Tuple[int, ...]:
        """
        Compute one size key for a whole-molecule chunk group.
        """

        total_structures = 0
        total_atoms = 0
        total_orbitals = 0
        for chunk in chunk_group:
            if isinstance(chunk, PackedMoleculeChunk):
                total_structures += int(chunk.atom_n_node.shape[0])
                total_atoms += int(np.asarray(chunk.atom_n_node, dtype=np.int32).sum())
                total_orbitals += int(np.asarray(chunk.rumer_n_node, dtype=np.int32).sum())
            else:
                total_structures += int(chunk.num_structures)
                total_atoms += int(chunk.total_atoms)
                total_orbitals += int(chunk.total_orbitals)

        if bucket_key == "num_atoms":
            return (total_atoms, total_structures)
        if bucket_key in {"num_atoms_num_orbitals", "(num_atoms,num_orbitals)"}:
            return (total_atoms, total_orbitals, total_structures)
        raise ValueError(f"Unsupported bucket_key={bucket_key}.")

    def groupCoarseBucketId(
        self,
        chunk_group: Sequence[PackedMoleculeChunk | PackedChunkReference],
        bucket_key: str,
    ) -> Tuple[int, ...]:
        """
        Map one whole-molecule chunk group to a coarse bucket id.
        """

        raw_key = self.groupBucketKey(chunk_group, bucket_key)
        if bucket_key == "num_atoms":
            total_atoms, total_structures = raw_key
            return (
                int(np.ceil(total_atoms / 64.0)),
                int(np.ceil(total_structures / 4.0)),
            )
        if bucket_key in {"num_atoms_num_orbitals", "(num_atoms,num_orbitals)"}:
            total_atoms, total_orbitals, total_structures = raw_key
            return (
                int(np.ceil(total_atoms / 64.0)),
                int(np.ceil(total_orbitals / 32.0)),
                int(np.ceil(total_structures / 4.0)),
            )
        raise ValueError(f"Unsupported bucket_key={bucket_key}.")

    def buildGroupedBatchIndices(
        self,
        chunk_groups: Sequence[Sequence[PackedMoleculeChunk | PackedChunkReference]],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_key: str,
        drop_remainder: bool,
    ) -> List[List[int]]:
        """
        Build bucketed batches over whole-molecule chunk groups.
        """

        if len(chunk_groups) == 0:
            return []

        indices = np.arange(len(chunk_groups), dtype=np.int32)
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(indices)

        bucket_to_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index in indices.tolist():
            bucket_to_indices[self.groupCoarseBucketId(chunk_groups[index], bucket_key)].append(index)

        batches: list[list[int]] = []
        for coarse_bucket_id in sorted(bucket_to_indices):
            group_indices = bucket_to_indices[coarse_bucket_id]
            group_indices = sorted(
                group_indices,
                key=lambda index: self.groupBucketKey(chunk_groups[index], bucket_key),
            )
            for start in range(0, len(group_indices), batch_size):
                batch_indices = group_indices[start : start + batch_size]
                if drop_remainder and len(batch_indices) < batch_size:
                    continue
                batches.append(batch_indices)
        if shuffle and len(batches) > 1:
            rng.shuffle(batches)
        return batches

    def buildBucketedBatchIndices(
        self,
        chunks: Sequence[PackedMoleculeChunk | PackedChunkReference],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_key: str,
        drop_remainder: bool,
    ) -> List[List[int]]:
        """
        Build batches within coarse size buckets to reduce shape diversity.
        """

        if len(chunks) == 0:
            return []

        indices = np.arange(len(chunks), dtype=np.int32)
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(indices)

        bucket_to_indices: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index in indices.tolist():
            bucket_to_indices[self.chunkCoarseBucketId(chunks[index], bucket_key)].append(index)

        batches: list[list[int]] = []
        for coarse_bucket_id in sorted(bucket_to_indices):
            bucket_indices = bucket_to_indices[coarse_bucket_id]
            bucket_indices = sorted(
                bucket_indices,
                key=lambda index: self.chunkBucketKey(chunks[index], bucket_key),
            )
            for start in range(0, len(bucket_indices), batch_size):
                batch_indices = bucket_indices[start : start + batch_size]
                if drop_remainder and len(batch_indices) < batch_size:
                    continue
                batches.append(batch_indices)
        if shuffle and len(batches) > 1:
            rng.shuffle(batches)
        return batches

    def createIterator(
        self,
        sample_groups: List[PackedMoleculeChunk | PackedChunkReference],
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool = False,
        bucket_key: str | None = None,
        fixed_bucket_config: dict[str, int] | None = None,
        split_name: str = "unknown",
    ) -> Iterator[UnifiedBatch]:
        """
        Create one iterator over batched single-molecule chunks.
        """

        ordered_groups = sample_groups
        bucketed_batch_indices: list[list[int]] | None = None
        if bucket_key is not None:
            bucketed_batch_indices = self.buildBucketedBatchIndices(
                chunks=sample_groups,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                bucket_key=bucket_key,
                drop_remainder=drop_remainder,
            )

        if bucketed_batch_indices is None:
            dataset = grain.MapDataset.source(list(range(len(ordered_groups))))
            if shuffle:
                dataset = dataset.shuffle(seed=seed)
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            iterator = iter(dataset.to_iter_dataset())
            batch_index_iterable = (
                [int(index) for index in list(chunk_index_batch)]
                for chunk_index_batch in iterator
            )
        else:
            batch_index_iterable = bucketed_batch_indices

        for batch_id, batch_indices in enumerate(batch_index_iterable):
            self.logIteratorStage(
                split_name=split_name,
                batch_id=batch_id,
                stage="indices_ready",
                extra=f"num_chunks={len(batch_indices)}",
            )
            chunk_batch = []
            for index in batch_indices:
                chunk_record = ordered_groups[index]
                if isinstance(chunk_record, PackedChunkReference):
                    chunk_batch.append(self.adapter.loadPackedChunk(chunk_record))
                else:
                    chunk_batch.append(chunk_record)
            total_structures = int(sum(int(chunk.atom_n_node.shape[0]) for chunk in chunk_batch))
            self.logIteratorStage(
                split_name=split_name,
                batch_id=batch_id,
                stage="chunks_loaded",
                extra=f"num_chunks={len(chunk_batch)} total_structures={total_structures}",
            )
            fixed_bucket_spec = None
            if fixed_bucket_config is not None:
                fixed_bucket_spec = self.adapter.buildFixedBucketSpec(
                    chunks=chunk_batch,
                    graph_step=int(fixed_bucket_config["graph_step"]),
                    static_atom_step=int(fixed_bucket_config["static_atom_step"]),
                    atom_step=int(fixed_bucket_config["atom_step"]),
                    atom_edge_step=int(fixed_bucket_config["atom_edge_step"]),
                    orbital_step=int(fixed_bucket_config["orbital_step"]),
                    rumer_edge_step=int(fixed_bucket_config["rumer_edge_step"]),
                    active_orbital_step=int(fixed_bucket_config["active_orbital_step"]),
                    active_edge_step=int(fixed_bucket_config["active_edge_step"]),
                )
            packed_batch = self.adapter.packPackedChunkBatch(
                chunk_batch,
                orbital_feature_dim=self.orbital_feature_dim,
                fixed_bucket_spec=fixed_bucket_spec,
                molecule_batch_size_target=batch_size if fixed_bucket_spec is not None else None,
                dataset_index_per_molecule=[
                    int(self.dataset_index_by_id.get(self.adapter.chunkDatasetId(chunk), 0))
                    for chunk in chunk_batch
                ],
            )
            self.logIteratorStage(
                split_name=split_name,
                batch_id=batch_id,
                stage="batch_packed",
                extra=(
                    f"num_graphs={int(packed_batch.num_atoms_per_graph.shape[0])} "
                    f"total_atoms={int(packed_batch.num_atoms_per_graph.sum())} "
                    f"total_orbitals={int(packed_batch.num_orbitals_per_graph.sum())} "
                    f"total_active_orbitals={int(packed_batch.active_rumer_graph.n_node.sum())} "
                    f"real_graphs={int(np.asarray(packed_batch.sample_mask, dtype=np.float32).sum())}"
                ),
            )
            yield packed_batch

    def loadChunkGroup(
        self,
        chunk_group: Sequence[PackedMoleculeChunk | PackedChunkReference],
    ) -> List[PackedMoleculeChunk]:
        """
        Materialize one whole-molecule chunk group from in-memory chunks or refs.
        """

        loaded_chunks: list[PackedMoleculeChunk] = []
        for chunk_record in chunk_group:
            if isinstance(chunk_record, PackedChunkReference):
                loaded_chunks.append(self.adapter.loadPackedChunk(chunk_record))
            else:
                loaded_chunks.append(chunk_record)
        return loaded_chunks

    def annotateChunkGroupWithTopMassFocus(
        self,
        chunk_group: Sequence[PackedMoleculeChunk],
        focus_cumulative_mass: float,
    ) -> tuple[list[PackedMoleculeChunk], np.ndarray]:
        """
        Attach one molecule-global top-mass mask to every chunk in a chunk group.
        """

        chunk_targets = [np.asarray(chunk.targets, dtype=np.float32) for chunk in chunk_group]
        concatenated_target = np.concatenate(chunk_targets, axis=0)
        concatenated_focus = computeTopMassFocusMask(
            target=concatenated_target,
            num_structures_per_molecule=[int(concatenated_target.shape[0])],
            cumulative_mass=focus_cumulative_mass,
        )

        annotated_chunks: list[PackedMoleculeChunk] = []
        offset = 0
        for chunk in chunk_group:
            count = int(chunk.atom_n_node.shape[0])
            chunk_focus = concatenated_focus[offset : offset + count]
            annotated_chunks.append(
                self.adapter.clonePackedChunkWithFocusMask(
                    chunk=chunk,
                    top_mass_focus_mask=chunk_focus,
                )
            )
            offset += count
        return annotated_chunks, concatenated_focus.astype(np.float32, copy=False)

    def selectChunkGroupStructures(
        self,
        chunk_group: Sequence[PackedMoleculeChunk],
        selected_structure_index: np.ndarray,
    ) -> List[PackedMoleculeChunk]:
        """
        Subset a whole-molecule chunk group to the requested global structure ids.
        """

        selected_structure_index = np.asarray(selected_structure_index, dtype=np.int32)
        selected_chunks: list[PackedMoleculeChunk] = []
        offset = 0
        for chunk in chunk_group:
            count = int(chunk.atom_n_node.shape[0])
            in_chunk = (
                (selected_structure_index >= offset)
                & (selected_structure_index < (offset + count))
            )
            if np.any(in_chunk):
                local_index = selected_structure_index[in_chunk] - offset
                selected_chunks.append(
                    self.adapter.subsetPackedChunk(
                        chunk=chunk,
                        selected_structure_index=local_index,
                        top_mass_focus_mask=self.adapter.chunkTopMassFocusMask(chunk),
                    )
                )
            offset += count
        return selected_chunks

    def createMoleculeIterator(
        self,
        molecule_groups: List[List[PackedMoleculeChunk | PackedChunkReference]],
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool = False,
        bucket_key: str | None = None,
        fixed_bucket_config: dict[str, int] | None = None,
        split_name: str = "unknown",
        focus_cumulative_mass: float = 0.98,
        top_mass_sample_strategy: str = "full_molecule",
        max_tail_samples_per_molecule: int | None = None,
        mixed_tail_top_fraction: float = 0.5,
        dataset_sampling_strategy: str = "natural",
    ) -> Iterator[UnifiedBatch]:
        """
        Create one iterator that batches whole molecules, optionally with top-mass subsampling.
        """

        ordered_groups = molecule_groups
        bucketed_batch_indices: list[list[int]] | None = None
        if bucket_key is not None:
            bucketed_batch_indices = self.buildGroupedBatchIndices(
                chunk_groups=molecule_groups,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                bucket_key=bucket_key,
                drop_remainder=drop_remainder,
            )

        if dataset_sampling_strategy == "balanced":
            batch_index_iterable = self.buildDatasetBalancedGroupedBatchIndices(
                chunk_groups=molecule_groups,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                bucket_key=bucket_key,
                drop_remainder=drop_remainder,
            )
        elif bucketed_batch_indices is None:
            dataset = grain.MapDataset.source(list(range(len(ordered_groups))))
            if shuffle:
                dataset = dataset.shuffle(seed=seed)
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            iterator = iter(dataset.to_iter_dataset())
            batch_index_iterable = (
                [int(index) for index in list(group_index_batch)]
                for group_index_batch in iterator
            )
        else:
            batch_index_iterable = bucketed_batch_indices

        rng = np.random.default_rng(seed)
        for batch_id, batch_indices in enumerate(batch_index_iterable):
            self.logIteratorStage(
                split_name=split_name,
                batch_id=batch_id,
                stage="indices_ready",
                extra=f"num_molecules={len(batch_indices)}",
            )

            chunk_batch: list[PackedMoleculeChunk] = []
            num_structures_per_molecule: list[int] = []
            for molecule_index in batch_indices:
                loaded_group = self.loadChunkGroup(ordered_groups[molecule_index])
                focused_group, full_focus_mask = self.annotateChunkGroupWithTopMassFocus(
                    chunk_group=loaded_group,
                    focus_cumulative_mass=focus_cumulative_mass,
                )
                full_target = np.concatenate(
                    [np.asarray(chunk.targets, dtype=np.float32) for chunk in focused_group],
                    axis=0,
                )
                if top_mass_sample_strategy == "full_molecule":
                    selected_group = focused_group
                    selected_count = int(full_target.shape[0])
                else:
                    selected_index = selectTopMassStructureIndices(
                        target=full_target,
                        focus_mask=full_focus_mask,
                        strategy=top_mass_sample_strategy,
                        max_tail_samples=max_tail_samples_per_molecule,
                        rng=rng,
                        mixed_top_fraction=mixed_tail_top_fraction,
                    )
                    selected_group = self.selectChunkGroupStructures(
                        chunk_group=focused_group,
                        selected_structure_index=selected_index,
                    )
                    selected_count = int(selected_index.shape[0])
                chunk_batch.extend(selected_group)
                num_structures_per_molecule.append(selected_count)

            total_structures = int(sum(int(chunk.atom_n_node.shape[0]) for chunk in chunk_batch))
            self.logIteratorStage(
                split_name=split_name,
                batch_id=batch_id,
                stage="chunks_loaded",
                extra=f"num_chunks={len(chunk_batch)} total_structures={total_structures}",
            )

            fixed_bucket_spec = None
            if fixed_bucket_config is not None:
                fixed_bucket_spec = self.adapter.buildFixedBucketSpec(
                    chunks=chunk_batch,
                    graph_step=int(fixed_bucket_config["graph_step"]),
                    static_atom_step=int(fixed_bucket_config["static_atom_step"]),
                    atom_step=int(fixed_bucket_config["atom_step"]),
                    atom_edge_step=int(fixed_bucket_config["atom_edge_step"]),
                    orbital_step=int(fixed_bucket_config["orbital_step"]),
                    rumer_edge_step=int(fixed_bucket_config["rumer_edge_step"]),
                    active_orbital_step=int(fixed_bucket_config["active_orbital_step"]),
                    active_edge_step=int(fixed_bucket_config["active_edge_step"]),
                )

            packed_batch = self.adapter.packPackedChunkBatch(
                chunk_batch,
                orbital_feature_dim=self.orbital_feature_dim,
                fixed_bucket_spec=fixed_bucket_spec,
                molecule_batch_size_target=batch_size if fixed_bucket_spec is not None else None,
                num_structures_per_molecule=num_structures_per_molecule,
                num_molecules_in_batch=len(batch_indices),
                dataset_index_per_molecule=[
                    int(
                        self.dataset_index_by_id.get(
                            self.chunkGroupDatasetId(ordered_groups[molecule_index]),
                            0,
                        )
                    )
                    for molecule_index in batch_indices
                ],
            )
            self.logIteratorStage(
                split_name=split_name,
                batch_id=batch_id,
                stage="batch_packed",
                extra=(
                    f"num_graphs={int(packed_batch.num_atoms_per_graph.shape[0])} "
                    f"total_atoms={int(packed_batch.num_atoms_per_graph.sum())} "
                    f"total_orbitals={int(packed_batch.num_orbitals_per_graph.sum())} "
                    f"total_active_orbitals={int(packed_batch.active_rumer_graph.n_node.sum())} "
                    f"real_graphs={int(np.asarray(packed_batch.sample_mask, dtype=np.float32).sum())}"
                ),
            )
            yield packed_batch

    def createSplitIterators(
        self,
        split_sample_groups: Dict[str, List[PackedMoleculeChunk | PackedChunkReference]],
        batch_size: int,
        seed: int,
        train_drop_remainder: bool = True,
        train_bucket_key: str | None = "num_atoms_num_orbitals",
    ) -> Dict[str, Iterable[UnifiedBatch]]:
        """
        Create split iterators for train/val/test.
        """

        return {
            "train": self.createIterator(
                split_sample_groups["train"],
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                drop_remainder=train_drop_remainder,
                bucket_key=train_bucket_key,
                split_name="train",
            ),
            "val": self.createIterator(
                split_sample_groups["val"],
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                drop_remainder=False,
                bucket_key=None,
                split_name="val",
            ),
            "test": self.createIterator(
                split_sample_groups["test"],
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                drop_remainder=False,
                bucket_key=None,
                split_name="test",
            ),
        }
