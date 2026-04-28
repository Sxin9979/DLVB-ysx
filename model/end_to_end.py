"""End-to-end atom-to-orbital-to-Rumer model definition."""

from dataclasses import dataclass

import flax.nnx as nnx
import jraph
import jax.numpy as jnp

from data.schema import UnifiedBatch
from model.atom_encoder import AtomE3Encoder, AtomEncoderConfig
from model.orbital_projection import AtomToOrbitalProjectionStack, OrbitalProjectionConfig
from model.rumer_encoder import RumerEncoderConfig, RumerGraphEncoder


@dataclass
class EndToEndModelConfig:
    """
    Bundle all submodule configs for end-to-end model creation.

    Arguments:
    - atom: AtomEncoderConfig.
    - orbital: OrbitalProjectionConfig.
    - rumer: RumerEncoderConfig.
    """

    atom: AtomEncoderConfig
    orbital: OrbitalProjectionConfig
    rumer: RumerEncoderConfig


class EndToEndE3VBModel(nnx.Module):
    """
    Full end-to-end architecture:
    atom encoder -> orbital projection + slot matching -> Rumer encoder -> prediction.
    """

    def __init__(self, config: EndToEndModelConfig, rngs: nnx.Rngs):
        """
        Initialize end-to-end model.

        Arguments:
        - config: EndToEndModelConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.atom_encoder = AtomE3Encoder(config=config.atom, rngs=rngs)
        self.projection = AtomToOrbitalProjectionStack(config=config.orbital, rngs=rngs)
        self.rumer_encoder = RumerGraphEncoder(config=config.rumer, rngs=rngs)

    def createRumerGraphWithNodeFeature(
        self,
        rumer_graph: jraph.GraphsTuple,
        orbital_feature,
    ) -> jraph.GraphsTuple:
        """
        Replace placeholder Rumer node tensor by dynamic orbital feature tensor.

        Arguments:
        - rumer_graph: Batched static-topology Rumer graph.
        - orbital_feature: Dynamic orbital node feature, shape [total_orbitals, orbital_feature_dim].

        Returns:
        - Updated Rumer graph with dynamic nodes.
        """

        return rumer_graph._replace(nodes=orbital_feature)

    def gatherActiveOrbitalFeature(
        self,
        orbital_feature: jnp.ndarray,
        active_orbital_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Gather active-orbital node features from the full orbital tensor.
        """

        return orbital_feature[active_orbital_index]

    def slotDiversityPenalty(
        self,
        slot_alpha: jnp.ndarray,
        active_capacity: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Penalize slots on the same atom that collapse onto similar candidate mixtures.
        """

        max_slots = int(slot_alpha.shape[1])
        slot_ids = jnp.arange(max_slots, dtype=jnp.int32)[None, :]
        valid_slot_mask = (slot_ids < active_capacity[:, None]).astype(slot_alpha.dtype)
        pair_mask = valid_slot_mask[:, :, None] * valid_slot_mask[:, None, :]
        pair_mask = pair_mask * (1.0 - jnp.eye(max_slots, dtype=slot_alpha.dtype)[None, :, :])
        similarity = jnp.einsum("ask,atk->ast", slot_alpha, slot_alpha)
        denominator = jnp.maximum(jnp.sum(pair_mask), 1.0)
        return jnp.sum(similarity * pair_mask) / denominator

    def forwardWithAux(self, batch: UnifiedBatch) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Run forward prediction together with auxiliary regularization terms.
        """

        atom_output = self.atom_encoder(
            batch.atom_graph,
            lap_evecs=batch.lap_evecs,
            lap_evals=batch.lap_evals,
        )
        projection_output = self.projection(
            scalar_feature=atom_output["scalar"],
            vector_feature=atom_output["vector"],
            positions=atom_output["positions"],
            num_atoms_per_graph=batch.num_atoms_per_graph,
            orbital_atom_index=batch.orbital_atom_index,
            orbital_role=batch.orbital_role,
            active_slot_index=batch.active_slot_index,
            local_frame_e1=batch.local_frame_e1,
            local_frame_e2=batch.local_frame_e2,
            local_frame_e3=batch.local_frame_e3,
        )
        slot_diversity_penalty = self.slotDiversityPenalty(
            slot_alpha=projection_output["slot_alpha"],
            active_capacity=projection_output["active_capacity"],
        )
        if self.config.rumer.mode == "active_only_low_risk":
            rumer_graph = None
            rumer_graph_q3_flipped = None
            active_rumer_graph = self.createRumerGraphWithNodeFeature(
                rumer_graph=batch.active_rumer_graph,
                orbital_feature=self.gatherActiveOrbitalFeature(
                    orbital_feature=projection_output["orbital_feature"],
                    active_orbital_index=batch.active_orbital_index,
                ),
            )
            active_rumer_graph_q3_flipped = self.createRumerGraphWithNodeFeature(
                rumer_graph=batch.active_rumer_graph,
                orbital_feature=self.gatherActiveOrbitalFeature(
                    orbital_feature=projection_output["orbital_feature_q3_flipped"],
                    active_orbital_index=batch.active_orbital_index,
                ),
            )
        else:
            rumer_graph = self.createRumerGraphWithNodeFeature(
                rumer_graph=batch.rumer_graph,
                orbital_feature=projection_output["orbital_feature"],
            )
            rumer_graph_q3_flipped = self.createRumerGraphWithNodeFeature(
                rumer_graph=batch.rumer_graph,
                orbital_feature=projection_output["orbital_feature_q3_flipped"],
            )
            active_rumer_graph = None
            active_rumer_graph_q3_flipped = None
        prediction = self.rumer_encoder(
            rumer_graph=rumer_graph,
            rumer_graph_q3_flipped=rumer_graph_q3_flipped,
            orbital_role=batch.orbital_role,
            active_rumer_graph=active_rumer_graph,
            active_rumer_graph_q3_flipped=active_rumer_graph_q3_flipped,
            full_orbital_feature=projection_output["orbital_feature"],
            full_orbital_feature_q3_flipped=projection_output["orbital_feature_q3_flipped"],
            num_orbitals_per_graph=batch.num_orbitals_per_graph,
        )
        return prediction, {
            "slot_diversity_penalty": slot_diversity_penalty,
        }

    def __call__(self, batch: UnifiedBatch):
        """
        Run forward prediction for one UnifiedBatch.

        Arguments:
        - batch: UnifiedBatch object.

        Returns:
        - Predicted normalized labels, shape [batch_size].
        """
        prediction, _ = self.forwardWithAux(batch)
        return prediction
