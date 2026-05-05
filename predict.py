"""Inference entry for E3VB structure-weight prediction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from flax import nnx
import jax.numpy as jnp
import numpy as np
import yaml

from data.grain_pipeline import GraphPackingAdapter
from data.schema import UnifiedSample
from data.xmi_str_parser import XmiStrParser
from data.xmo_builder import XmoFeatureBuilder
from data.xmo_parser import ParsedXMOMolecule, XmoParser
from model.end_to_end import EndToEndE3VBModel
from model.orbital_projection import LocalOrbitalProjector
from utils import ConfigFactory, NNXCheckpointManager
from utils.top_mass import buildPredictedTopMassMask


@dataclass
class PredictionRecord:
    """Store one structure-level prediction row."""

    vb_index: int
    structure_string: str
    prediction_raw: float
    prediction_nonnegative: float
    prediction_norm_by_pred_max: float
    prediction_norm_by_full_sum: float
    prediction_norm_by_selected_sum: float
    predicted_global_rank: int
    predicted_top_mass_focus: bool


class Predictor:
    """Run checkpoint-backed structure-weight prediction for one molecule."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        structure_batch_size: int | None = None,
    ):
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.structure_batch_size = structure_batch_size
        self.config = self.loadConfig()
        self.adapter = GraphPackingAdapter(lap_pe_k=self.config.model.atom.lap_pe_k)
        self.builder = XmoFeatureBuilder(lap_pe_k=self.config.model.atom.lap_pe_k)
        self.checkpoint = NNXCheckpointManager()
        self.model = self.buildModel()

    def loadConfig(self):
        """Load typed experiment config from YAML."""

        with open(self.config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        factory = ConfigFactory(payload)
        return factory.createExperimentConfig()

    def buildModel(self) -> EndToEndE3VBModel:
        """Instantiate model and restore checkpoint state."""

        rngs = nnx.Rngs(self.config.training.seed)
        model = EndToEndE3VBModel(config=self.config.model, rngs=rngs)
        self.checkpoint.load(model=model, path=self.checkpoint_path)
        return model

    def parseInput(
        self,
        xmo_path: str | None,
        xmi_path: str | None,
        str_path: str | None,
    ) -> ParsedXMOMolecule:
        """Parse one prediction input from either ``.xmo`` or ``xmi+str``."""

        if xmo_path is not None:
            return XmoParser(xmo_path).parse()
        if (xmi_path is None) or (str_path is None):
            raise ValueError("Either provide --xmo_path, or provide both --xmi_path and --str_path.")
        return XmiStrParser(xmi_path=xmi_path, str_path=str_path).parse()

    def buildSamples(self, parsed: ParsedXMOMolecule) -> tuple[list[UnifiedSample], list[str]]:
        """Convert one parsed molecule into structure-level unified samples."""

        built = self.builder.build(parsed)
        projector = LocalOrbitalProjector()
        molecule_local_frame_e1, molecule_local_frame_e2, molecule_local_frame_e3 = projector.buildLocalFrames(
            jnp.asarray(built.atom_positions, dtype=jnp.float32),
            jnp.asarray([built.atom_numbers.shape[0]], dtype=jnp.int32),
        )
        local_frame_e1 = np.asarray(molecule_local_frame_e1, dtype=np.float32)
        local_frame_e2 = np.asarray(molecule_local_frame_e2, dtype=np.float32)
        local_frame_e3 = np.asarray(molecule_local_frame_e3, dtype=np.float32)

        samples: list[UnifiedSample] = []
        structure_strings: list[str] = []
        structure_lookup = {structure.vb_index: structure for structure in parsed.structures}
        for structure_sample in built.structures:
            structure_strings.append(structure_lookup[structure_sample.vb_index].structure_string)
            samples.append(
                UnifiedSample(
                    molecule_id=built.molecule_id,
                    vb_index=structure_sample.vb_index,
                    atom_numbers=built.atom_numbers.copy(),
                    atom_positions=built.atom_positions.copy(),
                    atom_node_features=structure_sample.atom_node_features.copy(),
                    atom_senders=structure_sample.atom_senders.copy(),
                    atom_receivers=structure_sample.atom_receivers.copy(),
                    atom_pair_features=structure_sample.atom_pair_features.copy(),
                    lap_evals=structure_sample.lap_evals.copy(),
                    lap_evecs=structure_sample.lap_evecs.copy(),
                    orbital_atom_index=structure_sample.orbital_atom_index.copy(),
                    orbital_role=structure_sample.orbital_role.copy(),
                    active_slot_index=structure_sample.active_slot_index.copy(),
                    rumer_senders=structure_sample.rumer_senders.copy(),
                    rumer_receivers=structure_sample.rumer_receivers.copy(),
                    rumer_edge_type=structure_sample.rumer_edge_type.copy(),
                    active_orbital_index=structure_sample.active_orbital_index.copy(),
                    active_rumer_senders=structure_sample.active_rumer_senders.copy(),
                    active_rumer_receivers=structure_sample.active_rumer_receivers.copy(),
                    active_rumer_edge_type=structure_sample.active_rumer_edge_type.copy(),
                    target=0.0,
                    target_max=1.0,
                    top_mass_focus=0.0,
                )
            )

        self.molecule_local_frame_e1 = np.tile(local_frame_e1, (len(samples), 1))
        self.molecule_local_frame_e2 = np.tile(local_frame_e2, (len(samples), 1))
        self.molecule_local_frame_e3 = np.tile(local_frame_e3, (len(samples), 1))
        return samples, structure_strings

    def resolveSelectedIndices(
        self,
        samples: Sequence[UnifiedSample],
        vb_indices: Sequence[int] | None,
    ) -> np.ndarray:
        """Return sample indices for the requested VB structures."""

        if vb_indices is None:
            return np.arange(len(samples), dtype=np.int32)

        requested = [int(vb_index) for vb_index in vb_indices]
        index_by_vb_index = {int(sample.vb_index): index for index, sample in enumerate(samples)}
        missing = [vb_index for vb_index in requested if vb_index not in index_by_vb_index]
        if missing:
            raise ValueError(f"Requested vb_indices were not found in the input molecule: {missing}")
        return np.asarray([index_by_vb_index[vb_index] for vb_index in requested], dtype=np.int32)

    def iterateBatches(self, samples: Sequence[UnifiedSample]):
        """Yield batched samples for one molecule."""

        batch_size = self.structure_batch_size
        if batch_size is None:
            batch_size = len(samples)

        start = 0
        while start < len(samples):
            end = min(len(samples), start + int(batch_size))
            batch_samples = list(samples[start:end])
            atom_start = start * int(batch_samples[0].atom_numbers.shape[0])
            atom_end = end * int(batch_samples[0].atom_numbers.shape[0])
            batch = self.adapter.packBatch(
                samples=batch_samples,
                orbital_feature_dim=self.config.model.orbital.orbital_feature_dim,
                num_structures_per_molecule=[len(batch_samples)],
                local_frame_e1=self.molecule_local_frame_e1[atom_start:atom_end],
                local_frame_e2=self.molecule_local_frame_e2[atom_start:atom_end],
                local_frame_e3=self.molecule_local_frame_e3[atom_start:atom_end],
            )
            yield start, end, batch
            start = end

    def predictFromParsed(
        self,
        parsed: ParsedXMOMolecule,
        vb_indices: Sequence[int] | None = None,
    ) -> list[PredictionRecord]:
        """Run one full-molecule prediction and return ordered rows."""

        samples, structure_strings = self.buildSamples(parsed)
        selected_indices = self.resolveSelectedIndices(samples=samples, vb_indices=vb_indices)
        predictions = np.zeros((len(samples),), dtype=np.float32)

        for start, end, batch in self.iterateBatches(samples):
            batch_prediction = np.asarray(self.model(batch), dtype=np.float32)
            predictions[start:end] = batch_prediction

        nonnegative_prediction = np.maximum(predictions, 0.0)
        pred_max = float(np.max(predictions)) if len(predictions) > 0 else 1.0
        pred_scale = pred_max if abs(pred_max) > 1.0e-12 else 1.0
        full_sum = float(np.sum(nonnegative_prediction))
        full_scale = full_sum if full_sum > 1.0e-12 else 1.0
        selected_nonnegative_prediction = nonnegative_prediction[selected_indices]
        selected_sum = float(np.sum(selected_nonnegative_prediction))
        selected_scale = selected_sum if selected_sum > 1.0e-12 else 1.0
        predicted_focus_mask = buildPredictedTopMassMask(
            prediction=nonnegative_prediction,
            num_structures_per_molecule=[len(samples)],
            cumulative_mass=float(self.config.training.focus_cumulative_mass),
        ) > 0.5
        ranking_order = np.argsort(predictions)[::-1]
        global_rank = np.empty((len(samples),), dtype=np.int32)
        global_rank[ranking_order] = np.arange(1, len(samples) + 1, dtype=np.int32)

        return [
            PredictionRecord(
                vb_index=samples[int(index)].vb_index,
                structure_string=structure_strings[int(index)],
                prediction_raw=float(predictions[int(index)]),
                prediction_nonnegative=float(nonnegative_prediction[int(index)]),
                prediction_norm_by_pred_max=float(predictions[int(index)] / pred_scale),
                prediction_norm_by_full_sum=float(nonnegative_prediction[int(index)] / full_scale),
                prediction_norm_by_selected_sum=float(
                    nonnegative_prediction[int(index)] / selected_scale
                ),
                predicted_global_rank=int(global_rank[int(index)]),
                predicted_top_mass_focus=bool(predicted_focus_mask[int(index)]),
            )
            for index in selected_indices.tolist()
        ]

    def predict(
        self,
        xmo_path: str | None = None,
        xmi_path: str | None = None,
        str_path: str | None = None,
        vb_indices: Sequence[int] | None = None,
    ) -> tuple[ParsedXMOMolecule, list[PredictionRecord]]:
        """Parse input, run prediction, and return results."""

        parsed = self.parseInput(xmo_path=xmo_path, xmi_path=xmi_path, str_path=str_path)
        return parsed, self.predictFromParsed(parsed, vb_indices=vb_indices)


def parseVbIndices(argument: str | None) -> list[int] | None:
    """Parse one comma-separated vb index string."""

    if argument is None:
        return None
    value = argument.strip()
    if value == "":
        return None
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def writePredictionTable(
    output_path: str | None,
    molecule_id: str,
    records: Sequence[PredictionRecord],
) -> None:
    """Write one TSV table if output path is requested."""

    if output_path is None:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(
            "molecule_id\tvb_index\tprediction_raw\tprediction_nonnegative\t"
            "prediction_norm_by_pred_max\tprediction_norm_by_full_sum\t"
            "prediction_norm_by_selected_sum\tpredicted_global_rank\t"
            "predicted_top_mass_focus\tstructure_string\n"
        )
        for record in records:
            handle.write(
                f"{molecule_id}\t{record.vb_index}\t"
                f"{record.prediction_raw:.10f}\t"
                f"{record.prediction_nonnegative:.10f}\t"
                f"{record.prediction_norm_by_pred_max:.10f}\t"
                f"{record.prediction_norm_by_full_sum:.10f}\t"
                f"{record.prediction_norm_by_selected_sum:.10f}\t"
                f"{record.predicted_global_rank}\t"
                f"{int(record.predicted_top_mass_focus)}\t"
                f"{record.structure_string}\n"
            )


def printPredictionSummary(
    molecule_id: str,
    records: Sequence[PredictionRecord],
    top_k: int,
) -> None:
    """Print a compact ranked summary to stdout."""

    ranked = sorted(records, key=lambda record: record.prediction_raw, reverse=True)
    print(
        f"PREDICT molecule_id={molecule_id} num_structures={len(records)} top_k={min(top_k, len(records))}",
        flush=True,
    )
    for rank, record in enumerate(ranked[:top_k], start=1):
        print(
            f"TOP rank={rank:03d} vb_index={record.vb_index:05d} "
            f"pred_raw={record.prediction_raw:.10f} "
            f"pred_w_full={record.prediction_norm_by_full_sum:.10f} "
            f"pred_w_sel={record.prediction_norm_by_selected_sum:.10f} "
            f"global_rank={record.predicted_global_rank:05d} "
            f"pred_focus={int(record.predicted_top_mass_focus)} "
            f"structure=\"{record.structure_string}\"",
            flush=True,
        )


def parseArgs() -> argparse.Namespace:
    """Parse CLI arguments for structure-weight prediction."""

    parser = argparse.ArgumentParser(description="E3VB structure-weight prediction entry")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Experiment YAML used to build model config.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint path to load into the end-to-end model.",
    )
    parser.add_argument(
        "--xmo_path",
        type=str,
        default=None,
        help="Optional .xmo input path. If set, xmi/str are not required.",
    )
    parser.add_argument(
        "--xmi_path",
        type=str,
        default=None,
        help="Optional .xmi input path for paired xmi+str inference.",
    )
    parser.add_argument(
        "--str_path",
        type=str,
        default=None,
        help="Optional .str input path for paired xmi+str inference.",
    )
    parser.add_argument(
        "--structure_batch_size",
        type=int,
        default=None,
        help="Optional number of VB structures to score per inference batch.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="How many top-ranked structures to print to stdout.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional TSV output path for all structure predictions.",
    )
    parser.add_argument(
        "--vb_indices",
        type=str,
        default=None,
        help="Optional comma-separated vb indices to score as candidate structures only.",
    )
    return parser.parse_args()


def main() -> None:
    """Run CLI prediction."""

    arguments = parseArgs()
    predictor = Predictor(
        config_path=arguments.config_path,
        checkpoint_path=arguments.checkpoint_path,
        structure_batch_size=arguments.structure_batch_size,
    )
    parsed, records = predictor.predict(
        xmo_path=arguments.xmo_path,
        xmi_path=arguments.xmi_path,
        str_path=arguments.str_path,
        vb_indices=parseVbIndices(arguments.vb_indices),
    )
    printPredictionSummary(
        molecule_id=parsed.molecule_id,
        records=records,
        top_k=max(1, int(arguments.top_k)),
    )
    writePredictionTable(
        output_path=arguments.output_path,
        molecule_id=parsed.molecule_id,
        records=records,
    )


if __name__ == "__main__":
    main()
