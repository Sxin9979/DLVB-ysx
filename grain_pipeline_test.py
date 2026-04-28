"""Tests for chunked single-molecule Grain batching behavior."""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.grain_pipeline import GrainPipeline, GraphPackingAdapter
from data.schema import ProcessedMoleculeChunk, ProcessedStructureSample


def makeStructure(sample_id: int, num_atoms: int, num_orbitals: int) -> ProcessedStructureSample:
    """
    Create one lightweight processed structure sample for batching tests.
    """

    atom_node_features = np.zeros((num_atoms, 3), dtype=np.float32)

    if num_atoms > 1:
        atom_senders = np.arange(num_atoms - 1, dtype=np.int32)
        atom_receivers = np.arange(1, num_atoms, dtype=np.int32)
        atom_pair_features = np.zeros((num_atoms - 1, 2), dtype=np.float32)
    else:
        atom_senders = np.zeros((0,), dtype=np.int32)
        atom_receivers = np.zeros((0,), dtype=np.int32)
        atom_pair_features = np.zeros((0, 2), dtype=np.float32)
    lap_dim = 2
    lap_evals = np.linspace(0.1, 0.2, lap_dim, dtype=np.float32)
    lap_evecs = np.zeros((num_atoms, lap_dim), dtype=np.float32)
    if num_atoms > 0:
        lap_evecs[:, 0] = 1.0

    orbital_atom_index = np.zeros((num_orbitals, 2), dtype=np.int32)
    if num_atoms > 0 and num_orbitals > 0:
        orbital_atom_index[:, 0] = np.arange(num_orbitals, dtype=np.int32) % num_atoms
        orbital_atom_index[:, 1] = orbital_atom_index[:, 0]
    orbital_role = np.zeros((num_orbitals,), dtype=np.int32)
    active_slot_index = -np.ones((num_orbitals,), dtype=np.int32)
    if num_orbitals > 0:
        orbital_role[-1] = 2
        active_slot_index[-1] = 0

    if num_orbitals > 1:
        rumer_senders = np.arange(num_orbitals - 1, dtype=np.int32)
        rumer_receivers = np.arange(1, num_orbitals, dtype=np.int32)
        rumer_edge_type = np.zeros((num_orbitals - 1,), dtype=np.int32)
    else:
        rumer_senders = np.zeros((0,), dtype=np.int32)
        rumer_receivers = np.zeros((0,), dtype=np.int32)
        rumer_edge_type = np.zeros((0,), dtype=np.int32)

    if num_orbitals > 0:
        active_orbital_index = np.asarray([num_orbitals - 1], dtype=np.int32)
    else:
        active_orbital_index = np.zeros((0,), dtype=np.int32)
    active_rumer_senders = np.zeros((0,), dtype=np.int32)
    active_rumer_receivers = np.zeros((0,), dtype=np.int32)
    active_rumer_edge_type = np.zeros((0,), dtype=np.int32)

    return ProcessedStructureSample(
        vb_index=sample_id,
        atom_node_features=atom_node_features,
        atom_senders=atom_senders,
        atom_receivers=atom_receivers,
        atom_pair_features=atom_pair_features,
        lap_evals=lap_evals,
        lap_evecs=lap_evecs,
        orbital_atom_index=orbital_atom_index,
        orbital_role=orbital_role,
        active_slot_index=active_slot_index,
        rumer_senders=rumer_senders,
        rumer_receivers=rumer_receivers,
        rumer_edge_type=rumer_edge_type,
        active_orbital_index=active_orbital_index,
        active_rumer_senders=active_rumer_senders,
        active_rumer_receivers=active_rumer_receivers,
        active_rumer_edge_type=active_rumer_edge_type,
        target=float(sample_id),
        target_max=1.0,
    )


def makeChunk(chunk_id: int, num_atoms: int, num_orbitals: int, num_structures: int) -> ProcessedMoleculeChunk:
    """
    Create one single-molecule chunk for batching tests.
    """

    atom_numbers = np.arange(1, num_atoms + 1, dtype=np.int32)
    atom_positions = np.zeros((num_atoms, 3), dtype=np.float32)
    local_frame_e1 = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (num_atoms, 1))
    local_frame_e2 = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (num_atoms, 1))
    local_frame_e3 = np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (num_atoms, 1))
    structures = [
        makeStructure(sample_id=100 * chunk_id + structure_id, num_atoms=num_atoms, num_orbitals=num_orbitals)
        for structure_id in range(num_structures)
    ]
    return ProcessedMoleculeChunk(
        molecule_id=f"mol_{chunk_id}",
        chunk_index=0,
        num_chunks=1,
        atom_numbers=atom_numbers,
        atom_positions=atom_positions,
        local_frame_e1=local_frame_e1,
        local_frame_e2=local_frame_e2,
        local_frame_e3=local_frame_e3,
        structures=structures,
    )


def testReorderChunksForBucketingSortsByCompositeSize() -> None:
    """
    Chunk bucket sorting should follow total (num_atoms, num_orbitals).
    """

    chunks = [
        makeChunk(chunk_id=10, num_atoms=5, num_orbitals=2, num_structures=1),
        makeChunk(chunk_id=11, num_atoms=1, num_orbitals=9, num_structures=1),
        makeChunk(chunk_id=12, num_atoms=4, num_orbitals=1, num_structures=1),
        makeChunk(chunk_id=13, num_atoms=1, num_orbitals=1, num_structures=2),
    ]
    adapter = GraphPackingAdapter(lap_pe_k=2)
    packed_chunks = [adapter.packChunk(chunk) for chunk in chunks]
    pipeline = GrainPipeline(adapter=adapter, orbital_feature_dim=8)

    ordered = pipeline.reorderChunksForBucketing(
        chunks=packed_chunks,
        batch_size=2,
        shuffle=False,
        seed=0,
        bucket_key="num_atoms_num_orbitals",
    )

    assert [chunk.molecule_id for chunk in ordered] == ["mol_11", "mol_13", "mol_12", "mol_10"]


def testCreateIteratorDropsFinalShortTrainChunkBatch() -> None:
    """
    drop_remainder should remove the final incomplete chunk batch.
    """

    adapter = GraphPackingAdapter(lap_pe_k=2)
    chunks = [
        adapter.packChunk(makeChunk(chunk_id=index, num_atoms=index + 1, num_orbitals=2, num_structures=1))
        for index in range(5)
    ]
    pipeline = GrainPipeline(adapter=adapter, orbital_feature_dim=8)

    batches = list(
        pipeline.createIterator(
            sample_groups=chunks,
            batch_size=2,
            shuffle=False,
            seed=0,
            drop_remainder=True,
            bucket_key="num_atoms",
        )
    )

    assert len(batches) == 2
    assert sum(int(batch.num_molecules_in_batch) for batch in batches) == 4


def testCreateSplitIteratorsKeepsEqualChunkCountAndCachedFrames() -> None:
    """
    Each batch should contain a fixed number of single-molecule chunks and cached frames.
    """

    adapter = GraphPackingAdapter(lap_pe_k=2)
    pipeline = GrainPipeline(adapter=adapter, orbital_feature_dim=8)
    split_sample_groups = {
        "train": [
            adapter.packChunk(makeChunk(chunk_id=0, num_atoms=1, num_orbitals=4, num_structures=1)),
            adapter.packChunk(makeChunk(chunk_id=1, num_atoms=2, num_orbitals=2, num_structures=2)),
            adapter.packChunk(makeChunk(chunk_id=2, num_atoms=3, num_orbitals=3, num_structures=3)),
            adapter.packChunk(makeChunk(chunk_id=3, num_atoms=4, num_orbitals=1, num_structures=1)),
        ],
        "val": [
            adapter.packChunk(makeChunk(chunk_id=10, num_atoms=1, num_orbitals=1, num_structures=1)),
            adapter.packChunk(makeChunk(chunk_id=11, num_atoms=2, num_orbitals=1, num_structures=1)),
            adapter.packChunk(makeChunk(chunk_id=12, num_atoms=3, num_orbitals=1, num_structures=1)),
        ],
        "test": [
            adapter.packChunk(makeChunk(chunk_id=20, num_atoms=1, num_orbitals=1, num_structures=1)),
            adapter.packChunk(makeChunk(chunk_id=21, num_atoms=2, num_orbitals=1, num_structures=1)),
        ],
    }

    split_iterators = pipeline.createSplitIterators(
        split_sample_groups=split_sample_groups,
        batch_size=2,
        seed=0,
        train_drop_remainder=True,
        train_bucket_key="num_atoms",
    )

    train_batches = list(split_iterators["train"])
    val_batches = list(split_iterators["val"])

    assert len(train_batches) == 2
    assert len(val_batches) == 2
    for batch in train_batches:
        assert int(batch.num_molecules_in_batch) == 2
        assert np.asarray(batch.num_structures_per_molecule).shape[0] == 2
        assert batch.local_frame_e1.shape[-1] == 3
