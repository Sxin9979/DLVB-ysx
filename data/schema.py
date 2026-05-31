"""Data schema objects for the end-to-end JAX E3VB pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import jraph
import numpy as np


@dataclass
class DatasetSourceConfig:
    """
    Describe one logical dataset that contributes molecules to unified training.

    Arguments:
    - dataset_id: Stable dataset name used for split accounting and metrics.
    - xmo_dir: Directory containing ``.xmo`` files for this dataset.
    - split: Optional per-dataset train/val/test split ratios. When omitted, the
      global data split from the top-level config is used.
    """

    dataset_id: str
    xmo_dir: str
    split: Optional[tuple[float, float, float]] = None


@dataclass
class UnifiedSample:
    """
    Store one VB-structure sample with both atom and orbital views.

    Purpose:
    - Hold atom-level graph inputs.
    - Hold orbital/Rumer static topology metadata.
    - Hold molecule-normalized regression target.

    Arguments:
    - dataset_id: Source dataset identifier for balanced sampling / reporting.
    - molecule_id: Molecule identifier string.
    - vb_index: VB-structure index inside the molecule.
    - atom_numbers: Atomic numbers, shape [num_atoms].
    - atom_positions: Cartesian coordinates, shape [num_atoms, 3].
    - atom_node_features: Atom feature tensor, shape [num_atoms, 3].
      The 3 channels follow the existing project semantics:
      [atomic_number, active_electron_count, inactive_electron_count].
    - atom_senders: Atom-graph senders, shape [num_atom_edges].
    - atom_receivers: Atom-graph receivers, shape [num_atom_edges].
    - atom_pair_features: Atom edge pair features, shape [num_atom_edges, 2].
      The 2 channels follow existing semantics:
      [active_electrons_on_edge, inactive_electrons_on_edge].
    - lap_evals: Laplacian PE eigenvalues for this atom graph, shape [lap_pe_k].
    - lap_evecs: Laplacian PE eigenvectors for this atom graph, shape [num_atoms, lap_pe_k].
    - orbital_atom_index: Orbital-to-atom map, shape [num_orbitals, 2].
      For core/active orbitals the two indices are the same atom.
      For inactive bonding orbitals the two indices are the bonded atoms.
    - orbital_role: Orbital role id, shape [num_orbitals].
      Role ids follow current definition:
      0=inactive_core, 1=inactive_bonding, 2=active.
    - active_slot_index: Local slot id for active orbitals, shape [num_orbitals].
      Non-active orbitals use -1.
    - rumer_senders: Rumer-graph senders, shape [num_rumer_edges].
    - rumer_receivers: Rumer-graph receivers, shape [num_rumer_edges].
    - rumer_edge_type: Rumer edge type id, shape [num_rumer_edges].
      Edge types follow current definition:
      0=core self-loop, 1=inactive bonding self-loop, 2=active pairing edge.
    - active_orbital_index: Full-orbital indices of active orbitals, shape [num_active_orbitals].
    - active_rumer_senders: Active-only Rumer senders, shape [num_active_rumer_edges].
    - active_rumer_receivers: Active-only Rumer receivers, shape [num_active_rumer_edges].
    - active_rumer_edge_type: Active-only Rumer edge type ids, shape [num_active_rumer_edges].
    - target: Molecule-normalized label y_true / y_max.
    - target_max: Molecule-level y_max used in normalization.
    - top_mass_focus: Whether this structure belongs to the molecule-local
      cumulative-mass focus set used by the top-mass objective.
    """

    dataset_id: str
    molecule_id: str
    vb_index: int
    atom_numbers: np.ndarray
    atom_positions: np.ndarray
    atom_node_features: np.ndarray
    atom_senders: np.ndarray
    atom_receivers: np.ndarray
    atom_pair_features: np.ndarray
    lap_evals: np.ndarray
    lap_evecs: np.ndarray
    orbital_atom_index: np.ndarray
    orbital_role: np.ndarray
    active_slot_index: np.ndarray
    rumer_senders: np.ndarray
    rumer_receivers: np.ndarray
    rumer_edge_type: np.ndarray
    active_orbital_index: np.ndarray
    active_rumer_senders: np.ndarray
    active_rumer_receivers: np.ndarray
    active_rumer_edge_type: np.ndarray
    target: float
    target_max: float
    top_mass_focus: float = 0.0


@dataclass
class ProcessedStructureSample:
    """
    Store one processed VB-structure sample with molecule-static fields removed.

    Arguments:
    - vb_index: VB-structure index inside the molecule.
    - atom_node_features: Atom feature tensor, shape [num_atoms, 3].
    - atom_senders: Atom-graph senders, shape [num_atom_edges].
    - atom_receivers: Atom-graph receivers, shape [num_atom_edges].
    - atom_pair_features: Atom edge pair features, shape [num_atom_edges, 2].
    - lap_evals: Laplacian PE eigenvalues for this graph, shape [lap_pe_k].
    - lap_evecs: Laplacian PE eigenvectors for this graph, shape [num_atoms, lap_pe_k].
    - orbital_atom_index: Orbital-to-atom map, shape [num_orbitals, 2].
    - orbital_role: Orbital role id, shape [num_orbitals].
    - active_slot_index: Local slot id for active orbitals, shape [num_orbitals].
    - rumer_senders: Rumer-graph senders, shape [num_rumer_edges].
    - rumer_receivers: Rumer-graph receivers, shape [num_rumer_edges].
    - rumer_edge_type: Rumer edge type id, shape [num_rumer_edges].
    - active_orbital_index: Full-orbital indices of active orbitals, shape [num_active_orbitals].
    - active_rumer_senders: Active-only Rumer-graph senders, shape [num_active_rumer_edges].
    - active_rumer_receivers: Active-only Rumer-graph receivers, shape [num_active_rumer_edges].
    - active_rumer_edge_type: Active-only Rumer edge type ids, shape [num_active_rumer_edges].
    - target: Molecule-normalized label y_true / y_max.
    - target_max: Molecule-level y_max used in normalization.
    - top_mass_focus: Whether this structure belongs to the molecule-local
      cumulative-mass focus set used by the top-mass objective.
    """

    vb_index: int
    atom_node_features: np.ndarray
    atom_senders: np.ndarray
    atom_receivers: np.ndarray
    atom_pair_features: np.ndarray
    lap_evals: np.ndarray
    lap_evecs: np.ndarray
    orbital_atom_index: np.ndarray
    orbital_role: np.ndarray
    active_slot_index: np.ndarray
    rumer_senders: np.ndarray
    rumer_receivers: np.ndarray
    rumer_edge_type: np.ndarray
    active_orbital_index: np.ndarray
    active_rumer_senders: np.ndarray
    active_rumer_receivers: np.ndarray
    active_rumer_edge_type: np.ndarray
    target: float
    target_max: float
    top_mass_focus: float = 0.0


@dataclass
class ProcessedMolecule:
    """
    Store one processed molecule with static geometry cached once.

    Arguments:
    - dataset_id: Source dataset identifier for this molecule.
    - molecule_id: Molecule identifier string.
    - atom_numbers: Atomic numbers, shape [num_atoms].
    - atom_positions: Cartesian coordinates, shape [num_atoms, 3].
    - local_frame_e1: Cached local frame axis e1, shape [num_atoms, 3].
    - local_frame_e2: Cached local frame axis e2, shape [num_atoms, 3].
    - local_frame_e3: Cached local frame axis e3, shape [num_atoms, 3].
    - structures: Structure-specific samples belonging to this molecule.
    """

    dataset_id: str
    molecule_id: str
    atom_numbers: np.ndarray
    atom_positions: np.ndarray
    local_frame_e1: np.ndarray
    local_frame_e2: np.ndarray
    local_frame_e3: np.ndarray
    structures: List[ProcessedStructureSample]


@dataclass
class ProcessedMoleculeChunk:
    """
    Store one single-molecule chunk used by training/evaluation loaders.

    Arguments:
    - dataset_id: Source dataset identifier for the parent molecule.
    - molecule_id: Parent molecule identifier.
    - chunk_index: Chunk index inside the parent molecule.
    - num_chunks: Total number of chunks for the parent molecule.
    - atom_numbers: Atomic numbers, shape [num_atoms].
    - atom_positions: Cartesian coordinates, shape [num_atoms, 3].
    - local_frame_e1: Cached local frame axis e1, shape [num_atoms, 3].
    - local_frame_e2: Cached local frame axis e2, shape [num_atoms, 3].
    - local_frame_e3: Cached local frame axis e3, shape [num_atoms, 3].
    - structures: Structure-specific samples contained in this chunk.
    """

    dataset_id: str
    molecule_id: str
    chunk_index: int
    num_chunks: int
    atom_numbers: np.ndarray
    atom_positions: np.ndarray
    local_frame_e1: np.ndarray
    local_frame_e2: np.ndarray
    local_frame_e3: np.ndarray
    structures: List[ProcessedStructureSample]


@dataclass
class ProcessedDatasetCache:
    """
    Store a fully processed dataset payload that can be reused across runs.

    Arguments:
    - dataset_ids: Ordered dataset identifiers included in this cache.
    - split_ids: Molecule ids for each train/val/test split.
    - split_ids_by_dataset: Molecule ids for each split grouped by dataset id.
    - molecules: Mapping from molecule id to processed molecule object.
    """

    split_ids: Dict[str, List[str]]
    molecules: Dict[str, ProcessedMolecule]
    dataset_ids: Optional[List[str]] = None
    split_ids_by_dataset: Optional[Dict[str, Dict[str, List[str]]]] = None


@dataclass
class PackedMoleculeChunk:
    """
    Store one offline-packed single-molecule chunk for direct training loads.

    Arguments:
    - dataset_id: Source dataset identifier for the parent molecule.
    - molecule_id: Parent molecule identifier.
    - chunk_index: Chunk index inside the parent molecule.
    - num_chunks: Total number of chunks for the parent molecule.
    - atom_numbers: Atomic numbers for one molecule geometry, shape [num_atoms].
    - atom_positions: Cartesian coordinates for one molecule geometry, shape [num_atoms, 3].
    - local_frame_e1: Cached local frame axis e1, shape [num_atoms, 3].
    - local_frame_e2: Cached local frame axis e2, shape [num_atoms, 3].
    - local_frame_e3: Cached local frame axis e3, shape [num_atoms, 3].
    - atom_node_features: Packed atom node features for all structures in the chunk,
      shape [sum(num_atoms_per_graph), 3].
    - atom_senders: Packed atom senders with chunk-local offsets, shape [sum(num_atom_edges)].
    - atom_receivers: Packed atom receivers with chunk-local offsets, shape [sum(num_atom_edges)].
    - atom_pair_features: Packed atom edge features, shape [sum(num_atom_edges), 2].
    - lap_evals: Packed node-aligned repeated Laplacian eigenvalues,
      shape [sum(num_atoms_per_graph), lap_pe_k].
    - lap_evecs: Packed Laplacian eigenvectors, shape [sum(num_atoms_per_graph), lap_pe_k].
    - atom_n_node: Atom counts for each structure graph, shape [num_structures].
    - atom_n_edge: Atom edge counts for each structure graph, shape [num_structures].
    - orbital_atom_index: Packed orbital-to-atom map with chunk-local atom offsets,
      shape [sum(num_orbitals_per_graph), 2].
    - orbital_role: Packed orbital role ids, shape [sum(num_orbitals_per_graph)].
    - active_slot_index: Packed active slot ids, shape [sum(num_orbitals_per_graph)].
    - rumer_senders: Packed Rumer senders with chunk-local orbital offsets.
    - rumer_receivers: Packed Rumer receivers with chunk-local orbital offsets.
    - rumer_edge_type: Packed Rumer edge type ids.
    - rumer_n_node: Orbital counts for each structure graph, shape [num_structures].
    - rumer_n_edge: Rumer edge counts for each structure graph, shape [num_structures].
    - active_orbital_index: Packed full-orbital indices for active orbitals in each structure,
      shape [sum(num_active_orbitals_per_graph)].
    - active_rumer_senders: Packed active-only Rumer senders with active-orbital offsets.
    - active_rumer_receivers: Packed active-only Rumer receivers with active-orbital offsets.
    - active_rumer_edge_type: Packed active-only Rumer edge types.
    - active_rumer_n_node: Active orbital counts for each structure graph, shape [num_structures].
    - active_rumer_n_edge: Active-only Rumer edge counts for each structure graph, shape [num_structures].
    - targets: Packed normalized targets for all structures in the chunk, shape [num_structures].
    - top_mass_focus_mask: Packed binary focus-set mask for all structures in the
      chunk, shape [num_structures]. Older caches may omit this field.
    """

    molecule_id: str
    chunk_index: int
    num_chunks: int
    atom_numbers: np.ndarray
    atom_positions: np.ndarray
    local_frame_e1: np.ndarray
    local_frame_e2: np.ndarray
    local_frame_e3: np.ndarray
    atom_node_features: np.ndarray
    atom_senders: np.ndarray
    atom_receivers: np.ndarray
    atom_pair_features: np.ndarray
    lap_evals: np.ndarray
    lap_evecs: np.ndarray
    atom_n_node: np.ndarray
    atom_n_edge: np.ndarray
    orbital_atom_index: np.ndarray
    orbital_role: np.ndarray
    active_slot_index: np.ndarray
    rumer_senders: np.ndarray
    rumer_receivers: np.ndarray
    rumer_edge_type: np.ndarray
    rumer_n_node: np.ndarray
    rumer_n_edge: np.ndarray
    active_orbital_index: np.ndarray
    active_rumer_senders: np.ndarray
    active_rumer_receivers: np.ndarray
    active_rumer_edge_type: np.ndarray
    active_rumer_n_node: np.ndarray
    active_rumer_n_edge: np.ndarray
    targets: np.ndarray
    top_mass_focus_mask: Optional[np.ndarray] = None
    dataset_id: str = "default"


@dataclass
class PackedChunkReference:
    """
    Store one lightweight reference to an offline-packed single-molecule chunk.

    Arguments:
    - dataset_id: Source dataset identifier for the parent molecule.
    - molecule_id: Parent molecule identifier.
    - chunk_index: Chunk index inside the parent molecule.
    - num_chunks: Total number of chunks for the parent molecule.
    - chunk_path: On-disk pickle path for this chunk payload.
    - num_structures: Number of VB structures stored in this chunk.
    - num_focus_structures: Number of top-mass focus structures stored in this chunk.
    - static_num_atoms: Number of molecule-static atoms stored once for this chunk.
    - total_atoms: Total packed atom nodes across all structure graphs in this chunk.
    - total_atom_edges: Total packed atom edges across all structure graphs in this chunk.
    - total_orbitals: Total packed orbital nodes across all structure graphs in this chunk.
    - total_rumer_edges: Total packed Rumer edges across all structure graphs in this chunk.
    - total_active_orbitals: Total packed active orbital nodes across all structure graphs in this chunk.
    - total_active_rumer_edges: Total packed active-only Rumer edges across all structure graphs in this chunk.
    """

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
    hdf5_dataset_path: Optional[str] = None


@dataclass
class FixedBucketSpec:
    """
    Store coarse fixed padding targets for one runtime batch.

    Arguments:
    - total_graphs: Padded number of per-structure graphs.
    - total_static_atoms: Padded number of molecule-static atoms.
    - total_atoms: Padded number of expanded atom nodes.
    - total_atom_edges: Padded number of atom edges.
    - total_orbitals: Padded number of full orbital nodes.
    - total_rumer_edges: Padded number of full Rumer edges.
    - total_active_orbitals: Padded number of active orbital nodes.
    - total_active_rumer_edges: Padded number of active-only Rumer edges.
    """

    total_graphs: int
    total_static_atoms: int
    total_atoms: int
    total_atom_edges: int
    total_orbitals: int
    total_rumer_edges: int
    total_active_orbitals: int
    total_active_rumer_edges: int


@dataclass
class PackedDatasetCache:
    """
    Store offline-packed split chunks for direct train/eval loading.

    Arguments:
    - dataset_ids: Ordered dataset identifiers included in this cache.
    - split_ids: Molecule ids for each train/val/test split.
    - split_ids_by_dataset: Molecule ids for each split grouped by dataset id.
    - split_structure_counts: Structure counts for each split.
    - split_structure_counts_by_dataset: Structure counts for each split grouped
      by dataset id.
    - packed_chunks_by_view: Optional legacy in-memory packed chunk mapping with
      one extra top-level view key such as ``full`` or ``focus_only``.
    - packed_chunk_refs_by_view: Optional streaming chunk references keyed by
      view then split.
    - split_structure_counts_by_view: Optional structure counts keyed by
      view then split.
    - split_structure_counts_by_dataset_by_view: Optional dataset-level
      structure counts keyed by view then split then dataset id.
    - packed_chunks: Optional legacy in-memory mapping from split name to packed
      single-molecule chunks.
    - packed_chunk_refs: Optional mapping from split name to lightweight chunk
      references used for streaming loads.
    """

    split_ids: Dict[str, List[str]]
    split_structure_counts: Dict[str, int]
    packed_chunks_by_view: Optional[Dict[str, Dict[str, List[PackedMoleculeChunk]]]] = None
    packed_chunk_refs_by_view: Optional[Dict[str, Dict[str, List[PackedChunkReference]]]] = None
    split_structure_counts_by_view: Optional[Dict[str, Dict[str, int]]] = None
    split_structure_counts_by_dataset_by_view: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None
    packed_chunks: Optional[Dict[str, List[PackedMoleculeChunk]]] = None
    packed_chunk_refs: Optional[Dict[str, List[PackedChunkReference]]] = None
    dataset_ids: Optional[List[str]] = None
    split_ids_by_dataset: Optional[Dict[str, Dict[str, List[str]]]] = None
    split_structure_counts_by_dataset: Optional[Dict[str, Dict[str, int]]] = None

    def splitChunks(
        self,
        split_name: str,
        view_name: str = "full",
    ) -> List[PackedMoleculeChunk | PackedChunkReference]:
        """
        Return chunk records for one split, preferring lightweight references.
        """

        packed_chunk_refs_by_view = getattr(self, "packed_chunk_refs_by_view", None)
        if packed_chunk_refs_by_view is not None:
            return list(packed_chunk_refs_by_view[view_name][split_name])
        packed_chunks_by_view = getattr(self, "packed_chunks_by_view", None)
        if packed_chunks_by_view is not None:
            return list(packed_chunks_by_view[view_name][split_name])
        packed_chunk_refs = getattr(self, "packed_chunk_refs", None)
        if packed_chunk_refs is not None:
            return list(packed_chunk_refs[split_name])
        packed_chunks = getattr(self, "packed_chunks", None)
        if packed_chunks is not None:
            return list(packed_chunks[split_name])
        raise ValueError("PackedDatasetCache does not contain any packed chunk records.")

    def splitStructureCounts(self, view_name: str = "full") -> Dict[str, int]:
        """
        Return structure counts for one selected packed view.
        """

        split_structure_counts_by_view = getattr(self, "split_structure_counts_by_view", None)
        if split_structure_counts_by_view is not None:
            return dict(split_structure_counts_by_view[view_name])
        return dict(self.split_structure_counts)

    def splitStructureCountsByDataset(
        self,
        view_name: str = "full",
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Return dataset-level structure counts for one selected packed view.
        """

        split_structure_counts_by_dataset_by_view = getattr(
            self,
            "split_structure_counts_by_dataset_by_view",
            None,
        )
        if split_structure_counts_by_dataset_by_view is not None:
            return {
                split_name: dict(dataset_counts)
                for split_name, dataset_counts in split_structure_counts_by_dataset_by_view[view_name].items()
            }
        if self.split_structure_counts_by_dataset is None:
            return None
        return {
            split_name: dict(dataset_counts)
            for split_name, dataset_counts in self.split_structure_counts_by_dataset.items()
        }


@jax.tree_util.register_pytree_node_class
@dataclass
class UnifiedBatch:
    """
    Store one batched training item used by the end-to-end model.

    Purpose:
    - Hold batched atom-level and orbital-level Jraph graphs.
    - Hold orbital metadata required for atom-to-orbital mapping.
    - Hold graph-level normalized regression targets.

    Arguments:
    - atom_graph: Batched atom graph as jraph.GraphsTuple.
      nodes:
        features [total_atoms, 3]
      edges:
        pair [total_atom_edges, 2]
    - rumer_graph: Batched Rumer graph as jraph.GraphsTuple.
      nodes:
        placeholders [total_orbitals, orbital_feature_dim]
      edges:
        edge_type [total_rumer_edges]
    - active_rumer_graph: Batched active-only Rumer graph as jraph.GraphsTuple.
      nodes:
        placeholders [total_active_orbitals, orbital_feature_dim]
      edges:
        edge_type [total_active_rumer_edges]
    - orbital_atom_index: Batched orbital-to-atom map, shape [total_orbitals, 2].
    - orbital_role: Batched orbital role ids, shape [total_orbitals].
    - active_slot_index: Batched active slot ids, shape [total_orbitals].
    - active_orbital_index: Indices selecting active orbitals from the full orbital tensor,
      shape [total_active_orbitals].
    - lap_evals: Batched node-aligned repeated Laplacian eigenvalues,
      shape [total_atoms, lap_pe_k].
    - lap_evecs: Batched Laplacian eigenvectors, shape [total_atoms, lap_pe_k].
    - expanded_atom_to_static_atom_index: Gather indices mapping each expanded
      atom-graph node to one molecule-static atom row, shape [total_atoms].
    - static_atom_numbers: Molecule-static atomic numbers stored once per chunk,
      shape [total_static_atoms].
    - static_atom_positions: Molecule-static coordinates stored once per chunk,
      shape [total_static_atoms, 3].
    - num_atoms_per_graph: Atom node count for each graph, shape [batch_size].
    - num_orbitals_per_graph: Orbital node count for each graph, shape [batch_size].
    - num_structures_per_molecule: Structure counts for each molecule in the batch.
    - dataset_index_per_molecule: Dataset indices aligned with
      ``num_structures_per_molecule``. Padded molecule slots use ``-1``.
    - num_molecules_in_batch: Number of molecules in the current batch.
    - local_frame_e1: Molecule-static cached local frame axis e1, shape [total_static_atoms, 3].
    - local_frame_e2: Molecule-static cached local frame axis e2, shape [total_static_atoms, 3].
    - local_frame_e3: Molecule-static cached local frame axis e3, shape [total_static_atoms, 3].
    - top_mass_focus_mask: Binary mask selecting structures that belong to the
      molecule-local cumulative-mass focus set, shape [batch_size].
    - targets: Normalized targets, shape [batch_size].
    - sample_mask: Mask selecting real structure graphs, shape [batch_size].
    """

    atom_graph: jraph.GraphsTuple
    rumer_graph: jraph.GraphsTuple
    active_rumer_graph: jraph.GraphsTuple
    orbital_atom_index: jnp.ndarray
    orbital_role: jnp.ndarray
    active_slot_index: jnp.ndarray
    active_orbital_index: jnp.ndarray
    lap_evals: jnp.ndarray
    lap_evecs: jnp.ndarray
    expanded_atom_to_static_atom_index: jnp.ndarray
    static_atom_numbers: jnp.ndarray
    static_atom_positions: jnp.ndarray
    num_atoms_per_graph: jnp.ndarray
    num_orbitals_per_graph: jnp.ndarray
    num_structures_per_molecule: jnp.ndarray
    dataset_index_per_molecule: jnp.ndarray
    num_molecules_in_batch: int
    local_frame_e1: jnp.ndarray
    local_frame_e2: jnp.ndarray
    local_frame_e3: jnp.ndarray
    top_mass_focus_mask: jnp.ndarray
    targets: jnp.ndarray
    sample_mask: jnp.ndarray

    def tree_flatten(self):
        """
        Register UnifiedBatch as a JAX pytree so jitted steps can accept it directly.
        """

        children = (
            self.atom_graph,
            self.rumer_graph,
            self.active_rumer_graph,
            self.orbital_atom_index,
            self.orbital_role,
            self.active_slot_index,
            self.active_orbital_index,
            self.lap_evals,
            self.lap_evecs,
            self.expanded_atom_to_static_atom_index,
            self.static_atom_numbers,
            self.static_atom_positions,
            self.num_atoms_per_graph,
            self.num_orbitals_per_graph,
            self.num_structures_per_molecule,
            self.dataset_index_per_molecule,
            self.local_frame_e1,
            self.local_frame_e2,
            self.local_frame_e3,
            self.top_mass_focus_mask,
            self.targets,
            self.sample_mask,
        )
        aux_data = {"num_molecules_in_batch": self.num_molecules_in_batch}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Restore UnifiedBatch from pytree children and auxiliary metadata.
        """

        return cls(
            atom_graph=children[0],
            rumer_graph=children[1],
            active_rumer_graph=children[2],
            orbital_atom_index=children[3],
            orbital_role=children[4],
            active_slot_index=children[5],
            active_orbital_index=children[6],
            lap_evals=children[7],
            lap_evecs=children[8],
            expanded_atom_to_static_atom_index=children[9],
            static_atom_numbers=children[10],
            static_atom_positions=children[11],
            num_atoms_per_graph=children[12],
            num_orbitals_per_graph=children[13],
            num_structures_per_molecule=children[14],
            dataset_index_per_molecule=children[15],
            num_molecules_in_batch=aux_data["num_molecules_in_batch"],
            local_frame_e1=children[16],
            local_frame_e2=children[17],
            local_frame_e3=children[18],
            top_mass_focus_mask=children[19],
            targets=children[20],
            sample_mask=children[21],
        )
