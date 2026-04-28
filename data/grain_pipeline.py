"""Grain + Jraph pipeline for unified E3VB samples."""

from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import jax.numpy as jnp
import jraph
import numpy as np
from grain import python as grain

from data.lap_pe import lapPeFromEdges, repeatLapEvalsPerNode
from data.schema import PackedMoleculeChunk, ProcessedMoleculeChunk, UnifiedBatch, UnifiedSample


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

    def packBatch(
        self,
        samples: List[UnifiedSample],
        orbital_feature_dim: int,
        num_structures_per_molecule: Sequence[int] | None = None,
        local_frame_e1: np.ndarray | None = None,
        local_frame_e2: np.ndarray | None = None,
        local_frame_e3: np.ndarray | None = None,
    ) -> UnifiedBatch:
        """
        Pack one list of UnifiedSample objects into UnifiedBatch.
        """

        if num_structures_per_molecule is None:
            num_structures_per_molecule = [1 for _ in samples]
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
            num_atoms_per_graph=metadata["num_atoms_per_graph"],
            num_orbitals_per_graph=metadata["num_orbitals_per_graph"],
            num_structures_per_molecule=jnp.asarray(num_structures_per_molecule, dtype=jnp.int32),
            num_molecules_in_batch=int(len(num_structures_per_molecule)),
            local_frame_e1=jnp.asarray(local_frame_e1, dtype=jnp.float32),
            local_frame_e2=jnp.asarray(local_frame_e2, dtype=jnp.float32),
            local_frame_e3=jnp.asarray(local_frame_e3, dtype=jnp.float32),
            targets=metadata["targets"],
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
        )

    def packPackedChunkBatch(self, chunks: List[PackedMoleculeChunk], orbital_feature_dim: int) -> UnifiedBatch:
        """
        Combine offline-packed single-molecule chunks into one runtime UnifiedBatch.
        """

        atom_feature_parts = []
        atom_number_parts = []
        atom_position_parts = []
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

        atom_offset = 0
        orbital_offset = 0
        active_orbital_offset = 0
        for chunk in chunks:
            num_structures = int(chunk.atom_n_node.shape[0])
            total_chunk_atoms = int(chunk.atom_n_node.sum())
            total_chunk_orbitals = int(chunk.rumer_n_node.sum())

            atom_feature_parts.append(np.asarray(chunk.atom_node_features, dtype=np.float32))
            atom_number_parts.append(np.tile(np.asarray(chunk.atom_numbers, dtype=np.int32), num_structures))
            atom_position_parts.append(np.tile(np.asarray(chunk.atom_positions, dtype=np.float32), (num_structures, 1)))
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

            local_frame_e1_parts.append(np.tile(np.asarray(chunk.local_frame_e1, dtype=np.float32), (num_structures, 1)))
            local_frame_e2_parts.append(np.tile(np.asarray(chunk.local_frame_e2, dtype=np.float32), (num_structures, 1)))
            local_frame_e3_parts.append(np.tile(np.asarray(chunk.local_frame_e3, dtype=np.float32), (num_structures, 1)))
            targets.append(np.asarray(chunk.targets, dtype=np.float32))

            atom_offset += total_chunk_atoms
            orbital_offset += total_chunk_orbitals
            active_orbital_offset += int(np.asarray(chunk_active_rumer_n_node, dtype=np.int32).sum())

        atom_graph = jraph.GraphsTuple(
            nodes={
                "features": jnp.asarray(self.concatenateOrEmpty(atom_feature_parts, (3,), np.float32), dtype=jnp.float32),
                "numbers": jnp.asarray(self.concatenateOrEmpty(atom_number_parts, tuple(), np.int32), dtype=jnp.int32),
                "positions": jnp.asarray(self.concatenateOrEmpty(atom_position_parts, (3,), np.float32), dtype=jnp.float32),
            },
            edges={
                "pair": jnp.asarray(self.concatenateOrEmpty(atom_pair_feature_parts, (2,), np.float32), dtype=jnp.float32),
            },
            senders=jnp.asarray(self.concatenateOrEmpty(atom_senders_parts, tuple(), np.int32), dtype=jnp.int32),
            receivers=jnp.asarray(self.concatenateOrEmpty(atom_receivers_parts, tuple(), np.int32), dtype=jnp.int32),
            n_node=jnp.asarray(self.concatenateOrEmpty(atom_n_node_parts, tuple(), np.int32), dtype=jnp.int32),
            n_edge=jnp.asarray(self.concatenateOrEmpty(atom_n_edge_parts, tuple(), np.int32), dtype=jnp.int32),
            globals=None,
        )

        total_orbitals = int(sum(int(np.asarray(chunk.rumer_n_node, dtype=np.int32).sum()) for chunk in chunks))
        rumer_graph = jraph.GraphsTuple(
            nodes=jnp.zeros((total_orbitals, orbital_feature_dim), dtype=jnp.float32),
            edges={
                "edge_type": jnp.asarray(
                    self.concatenateOrEmpty(rumer_edge_type_parts, tuple(), np.int32),
                    dtype=jnp.int32,
                ),
            },
            senders=jnp.asarray(self.concatenateOrEmpty(rumer_senders_parts, tuple(), np.int32), dtype=jnp.int32),
            receivers=jnp.asarray(self.concatenateOrEmpty(rumer_receivers_parts, tuple(), np.int32), dtype=jnp.int32),
            n_node=jnp.asarray(self.concatenateOrEmpty(rumer_n_node_parts, tuple(), np.int32), dtype=jnp.int32),
            n_edge=jnp.asarray(self.concatenateOrEmpty(rumer_n_edge_parts, tuple(), np.int32), dtype=jnp.int32),
            globals=None,
        )

        total_active_orbitals = int(
            self.concatenateOrEmpty(active_rumer_n_node_parts, tuple(), np.int32).sum()
        )
        active_rumer_graph = jraph.GraphsTuple(
            nodes=jnp.zeros((total_active_orbitals, orbital_feature_dim), dtype=jnp.float32),
            edges={
                "edge_type": jnp.asarray(
                    self.concatenateOrEmpty(active_rumer_edge_type_parts, tuple(), np.int32),
                    dtype=jnp.int32,
                ),
            },
            senders=jnp.asarray(
                self.concatenateOrEmpty(active_rumer_senders_parts, tuple(), np.int32),
                dtype=jnp.int32,
            ),
            receivers=jnp.asarray(
                self.concatenateOrEmpty(active_rumer_receivers_parts, tuple(), np.int32),
                dtype=jnp.int32,
            ),
            n_node=jnp.asarray(
                self.concatenateOrEmpty(active_rumer_n_node_parts, tuple(), np.int32),
                dtype=jnp.int32,
            ),
            n_edge=jnp.asarray(
                self.concatenateOrEmpty(active_rumer_n_edge_parts, tuple(), np.int32),
                dtype=jnp.int32,
            ),
            globals=None,
        )

        return UnifiedBatch(
            atom_graph=atom_graph,
            rumer_graph=rumer_graph,
            active_rumer_graph=active_rumer_graph,
            orbital_atom_index=jnp.asarray(
                self.concatenateOrEmpty(orbital_atom_index_parts, (2,), np.int32),
                dtype=jnp.int32,
            ),
            orbital_role=jnp.asarray(self.concatenateOrEmpty(orbital_role_parts, tuple(), np.int32), dtype=jnp.int32),
            active_slot_index=jnp.asarray(
                self.concatenateOrEmpty(active_slot_index_parts, tuple(), np.int32),
                dtype=jnp.int32,
            ),
            active_orbital_index=jnp.asarray(
                self.concatenateOrEmpty(active_orbital_index_parts, tuple(), np.int32),
                dtype=jnp.int32,
            ),
            lap_evals=jnp.asarray(
                self.concatenateOrEmpty(lap_eval_parts, (0 if len(lap_eval_parts) == 0 else lap_eval_parts[0].shape[1],), np.float32),
                dtype=jnp.float32,
            ),
            lap_evecs=jnp.asarray(
                self.concatenateOrEmpty(lap_evec_parts, (0 if len(lap_evec_parts) == 0 else lap_evec_parts[0].shape[1],), np.float32),
                dtype=jnp.float32,
            ),
            num_atoms_per_graph=jnp.asarray(self.concatenateOrEmpty(atom_n_node_parts, tuple(), np.int32), dtype=jnp.int32),
            num_orbitals_per_graph=jnp.asarray(
                self.concatenateOrEmpty(rumer_n_node_parts, tuple(), np.int32),
                dtype=jnp.int32,
            ),
            num_structures_per_molecule=jnp.asarray(
                [int(chunk.atom_n_node.shape[0]) for chunk in chunks],
                dtype=jnp.int32,
            ),
            num_molecules_in_batch=int(len(chunks)),
            local_frame_e1=jnp.asarray(self.concatenateOrEmpty(local_frame_e1_parts, (3,), np.float32), dtype=jnp.float32),
            local_frame_e2=jnp.asarray(self.concatenateOrEmpty(local_frame_e2_parts, (3,), np.float32), dtype=jnp.float32),
            local_frame_e3=jnp.asarray(self.concatenateOrEmpty(local_frame_e3_parts, (3,), np.float32), dtype=jnp.float32),
            targets=jnp.asarray(self.concatenateOrEmpty(targets, tuple(), np.float32), dtype=jnp.float32),
        )


class GrainPipeline:
    """
    Build Grain iterators that emit UnifiedBatch items.
    """

    def __init__(self, adapter: GraphPackingAdapter, orbital_feature_dim: int):
        self.adapter = adapter
        self.orbital_feature_dim = orbital_feature_dim

    def chunkBucketKey(
        self,
        chunk: ProcessedMoleculeChunk | PackedMoleculeChunk,
        bucket_key: str,
    ) -> Tuple[int, ...]:
        """
        Compute one size key for a single-molecule chunk.
        """

        if isinstance(chunk, PackedMoleculeChunk):
            total_atoms = int(np.asarray(chunk.atom_n_node, dtype=np.int32).sum())
            total_orbitals = int(np.asarray(chunk.rumer_n_node, dtype=np.int32).sum())
            num_structures = int(chunk.atom_n_node.shape[0])
        else:
            total_atoms = int(chunk.atom_numbers.shape[0]) * len(chunk.structures)
            total_orbitals = sum(int(structure.orbital_role.shape[0]) for structure in chunk.structures)
            num_structures = len(chunk.structures)

        if bucket_key == "num_atoms":
            return (total_atoms, num_structures)
        if bucket_key in {"num_atoms_num_orbitals", "(num_atoms,num_orbitals)"}:
            return (total_atoms, total_orbitals, num_structures)
        raise ValueError(f"Unsupported bucket_key={bucket_key}.")

    def reorderChunksForBucketing(
        self,
        chunks: Sequence[PackedMoleculeChunk],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_key: str,
    ) -> List[ProcessedMoleculeChunk]:
        """
        Reorder chunks so consecutive batches contain similar single-molecule sizes.
        """

        if len(chunks) == 0:
            return []

        indices = np.arange(len(chunks), dtype=np.int32)
        rng = np.random.default_rng(seed)
        if shuffle:
            rng.shuffle(indices)

        ordered_indices = sorted(
            indices.tolist(),
            key=lambda index: self.chunkBucketKey(chunks[index], bucket_key),
        )
        batches = [
            ordered_indices[start : start + batch_size]
            for start in range(0, len(ordered_indices), batch_size)
        ]
        if shuffle and len(batches) > 1:
            rng.shuffle(batches)
        return [chunks[index] for batch in batches for index in batch]

    def createIterator(
        self,
        sample_groups: List[PackedMoleculeChunk],
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_remainder: bool = False,
        bucket_key: str | None = None,
    ) -> Iterator[UnifiedBatch]:
        """
        Create one iterator over batched single-molecule chunks.
        """

        ordered_groups = sample_groups
        if bucket_key is not None:
            ordered_groups = self.reorderChunksForBucketing(
                chunks=sample_groups,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                bucket_key=bucket_key,
            )

        dataset = grain.MapDataset.source(list(range(len(ordered_groups))))
        if shuffle and bucket_key is None:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        iterator = iter(dataset.to_iter_dataset())

        for chunk_index_batch in iterator:
            batch_indices = [int(index) for index in list(chunk_index_batch)]
            chunk_batch = [ordered_groups[index] for index in batch_indices]
            yield self.adapter.packPackedChunkBatch(
                chunk_batch,
                orbital_feature_dim=self.orbital_feature_dim,
            )

    def createSplitIterators(
        self,
        split_sample_groups: Dict[str, List[PackedMoleculeChunk]],
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
            ),
            "val": self.createIterator(
                split_sample_groups["val"],
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                drop_remainder=False,
                bucket_key=None,
            ),
            "test": self.createIterator(
                split_sample_groups["test"],
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                drop_remainder=False,
                bucket_key=None,
            ),
        }
