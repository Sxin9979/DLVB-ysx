"""Unified data pipeline exports for the end-to-end E3VB project."""

from data.processor import UnifiedDataConfig, UnifiedSampleProcessor, XmoDatasetProcessor
from data.schema import (
    DatasetSourceConfig,
    PackedChunkReference,
    PackedDatasetCache,
    PackedMoleculeChunk,
    ProcessedDatasetCache,
    ProcessedMolecule,
    ProcessedMoleculeChunk,
    ProcessedStructureSample,
    UnifiedBatch,
    UnifiedSample,
)
from data.xmo_builder import XmoFeatureBuilder
from data.xmi_str_parser import XmiStrParser
from data.xmo_parser import ParsedVBStructure, ParsedXMOMolecule, XmoParser

__all__ = [
    "GrainPipeline",
    "GraphPackingAdapter",
    "DatasetSourceConfig",
    "PackedChunkReference",
    "PackedDatasetCache",
    "PackedMoleculeChunk",
    "ProcessedDatasetCache",
    "ProcessedMolecule",
    "ProcessedMoleculeChunk",
    "ProcessedStructureSample",
    "UnifiedBatch",
    "UnifiedDataConfig",
    "UnifiedSample",
    "UnifiedSampleProcessor",
    "XmoDatasetProcessor",
    "XmiStrParser",
    "XmoFeatureBuilder",
    "ParsedVBStructure",
    "ParsedXMOMolecule",
    "XmoParser",
]


def __getattr__(name: str):
    """Lazily import Grain/Jraph-dependent pipeline objects on demand."""

    if name in {"GrainPipeline", "GraphPackingAdapter"}:
        from data.grain_pipeline import GrainPipeline, GraphPackingAdapter

        return {"GrainPipeline": GrainPipeline, "GraphPackingAdapter": GraphPackingAdapter}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
