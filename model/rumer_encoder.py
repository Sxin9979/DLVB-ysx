"""Rumer graph encoder based on Jraph message passing."""

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph


@nnx.remat
def runRumerMessageLayer(
    layer: "RumerMessageLayer",
    node_feature: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    edge_feature: jnp.ndarray,
) -> jnp.ndarray:
    """
    Rematerialize one Rumer message layer during backward to reduce activation memory.
    """

    return layer(
        node_feature=node_feature,
        senders=senders,
        receivers=receivers,
        edge_feature=edge_feature,
    )


@dataclass
class RumerEncoderConfig:
    """
    Configure Rumer graph message passing and readout.

    Arguments:
    - orbital_feature_dim: Input orbital node feature size.
    - hidden_dim: Hidden node feature size.
    - edge_embedding_dim: Edge type embedding size.
    - layers: Number of Rumer message layers.
    - mode: ``full`` keeps the original full-graph path, while
      ``active_only_low_risk`` runs message passing only on active orbitals and
      keeps background orbitals in a lightweight bypass readout.
    """

    orbital_feature_dim: int
    hidden_dim: int
    edge_embedding_dim: int
    layers: int
    mode: str = "full"


class RumerMessageLayer(nnx.Module):
    """
    One message passing layer on static Rumer topology.
    """

    def __init__(self, config: RumerEncoderConfig, rngs: nnx.Rngs):
        """
        Initialize one Rumer message layer.

        Arguments:
        - config: RumerEncoderConfig.
        - rngs: Flax NNX random container.
        """

        self.message_linear1 = nnx.Linear(
            2 * config.hidden_dim + config.edge_embedding_dim,
            config.hidden_dim,
            rngs=rngs,
        )
        self.message_linear2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.gate_linear1 = nnx.Linear(
            2 * config.hidden_dim + config.edge_embedding_dim,
            config.hidden_dim,
            rngs=rngs,
        )
        self.gate_linear2 = nnx.Linear(config.hidden_dim, 1, rngs=rngs)
        self.update_linear1 = nnx.Linear(
            2 * config.hidden_dim,
            config.hidden_dim,
            rngs=rngs,
        )
        self.update_linear2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(config.hidden_dim, rngs=rngs)

    def activate(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - tensor: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(tensor)

    def __call__(
        self,
        node_feature: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_feature: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run one Rumer message passing update.

        Arguments:
        - node_feature: Node tensor, shape [total_orbitals, hidden_dim].
        - senders: Edge sender indices, shape [num_edges].
        - receivers: Edge receiver indices, shape [num_edges].
        - edge_feature: Edge embedding tensor, shape [num_edges, edge_embedding_dim].

        Returns:
        - Updated node tensor, shape [total_orbitals, hidden_dim].
        """

        sender_feature = node_feature[senders]
        receiver_feature = node_feature[receivers]
        message_input = jnp.concatenate([sender_feature, receiver_feature, edge_feature], axis=-1)
        raw_message = self.activate(self.message_linear1(message_input))
        raw_message = self.message_linear2(raw_message)
        gate = self.activate(self.gate_linear1(message_input))
        gate = jax.nn.sigmoid(self.gate_linear2(gate))
        message = gate * raw_message

        aggregated = jraph.segment_sum(
            message,
            receivers,
            num_segments=node_feature.shape[0],
        )
        update_input = jnp.concatenate([node_feature, aggregated], axis=-1)
        update = self.activate(self.update_linear1(update_input))
        update = self.update_linear2(update)
        return self.norm(node_feature + update)


class RumerGraphEncoder(nnx.Module):
    """
    Encode orbital features on static Rumer topology and predict graph targets.
    """

    def __init__(self, config: RumerEncoderConfig, rngs: nnx.Rngs):
        """
        Initialize Rumer graph encoder.

        Arguments:
        - config: RumerEncoderConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.input_linear = nnx.Linear(
            config.orbital_feature_dim,
            config.hidden_dim,
            rngs=rngs,
        )
        self.edge_embedding = nnx.Embed(
            num_embeddings=3,
            features=config.edge_embedding_dim,
            rngs=rngs,
        )
        self.layers = [RumerMessageLayer(config=config, rngs=rngs) for _ in range(config.layers)]

        self.background_linear1 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.background_linear2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.active_linear1 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.active_linear2 = nnx.Linear(config.hidden_dim, config.hidden_dim, rngs=rngs)

        self.output_linear1 = nnx.Linear(2 * config.hidden_dim, config.hidden_dim, rngs=rngs)
        self.output_linear2 = nnx.Linear(config.hidden_dim, 1, rngs=rngs)

    def activate(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - tensor: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(tensor)

    def graphIndexFromCounts(self, counts: jnp.ndarray, total_nodes: int) -> jnp.ndarray:
        """
        Expand per-graph node counts to one graph id per node.

        Arguments:
        - counts: Node counts per graph, shape [batch_size].
        - total_nodes: Total node count in the flattened batch.

        Returns:
        - Graph index tensor for each node, shape [total_nodes].
        """

        graph_ids = jnp.arange(counts.shape[0], dtype=jnp.int32)
        return jnp.repeat(
            graph_ids,
            counts,
            total_repeat_length=total_nodes,
        )

    def maskedMeanPool(
        self,
        node_feature: jnp.ndarray,
        graph_index: jnp.ndarray,
        mask: jnp.ndarray,
        num_graphs: int,
    ) -> jnp.ndarray:
        """
        Mean pool selected nodes into graph embeddings.

        Arguments:
        - node_feature: Node tensor, shape [total_nodes, hidden_dim].
        - graph_index: Graph ids for nodes, shape [total_nodes].
        - mask: Node selection mask, shape [total_nodes].
        - num_graphs: Number of graphs in batch.

        Returns:
        - Graph tensor, shape [batch_size, hidden_dim].
        """

        weighted = node_feature * mask[:, None]
        pooled_sum = jraph.segment_sum(weighted, graph_index, num_segments=num_graphs)
        pooled_count = jraph.segment_sum(mask, graph_index, num_segments=num_graphs)
        return pooled_sum / jnp.maximum(pooled_count[:, None], 1.0)

    def encodeNodeFeature(self, rumer_graph: jraph.GraphsTuple) -> jnp.ndarray:
        """
        Encode one Rumer graph into node hidden states by message passing.

        Arguments:
        - rumer_graph: Batched Rumer graph.

        Returns:
        - Node hidden tensor, shape [total_orbitals, hidden_dim].
        """

        node_feature = self.input_linear(rumer_graph.nodes)
        edge_type = rumer_graph.edges["edge_type"]
        edge_feature = self.edge_embedding(edge_type)
        senders = rumer_graph.senders
        receivers = rumer_graph.receivers

        for layer in self.layers:
            node_feature = runRumerMessageLayer(
                layer=layer,
                node_feature=node_feature,
                senders=senders,
                receivers=receivers,
                edge_feature=edge_feature,
            )
        return node_feature

    def buildReadoutInput(
        self,
        node_feature: jnp.ndarray,
        n_node: jnp.ndarray,
        orbital_role: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Build fused graph embedding before final scalar head.

        Arguments:
        - node_feature: Node hidden tensor, shape [total_orbitals, hidden_dim].
        - n_node: Number of orbitals per graph, shape [batch_size].
        - orbital_role: Orbital role ids, shape [total_orbitals].

        Returns:
        - Fused graph embedding, shape [batch_size, 2 * hidden_dim].
        """

        graph_index = self.graphIndexFromCounts(
            counts=n_node,
            total_nodes=node_feature.shape[0],
        )
        num_graphs = int(n_node.shape[0])
        active_mask = (orbital_role == 2).astype(jnp.float32)
        background_mask = (orbital_role != 2).astype(jnp.float32)

        background = self.maskedMeanPool(
            node_feature=node_feature,
            graph_index=graph_index,
            mask=background_mask,
            num_graphs=num_graphs,
        )
        background = self.activate(self.background_linear1(background))
        background = self.background_linear2(background)

        active = self.maskedMeanPool(
            node_feature=node_feature,
            graph_index=graph_index,
            mask=active_mask,
            num_graphs=num_graphs,
        )
        active = self.activate(self.active_linear1(active))
        active = self.active_linear2(active)
        return jnp.concatenate([background, active], axis=-1)

    def buildBackgroundBypass(
        self,
        orbital_feature: jnp.ndarray,
        n_node: jnp.ndarray,
        orbital_role: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Build one lightweight background embedding without Rumer message passing.
        """

        graph_index = self.graphIndexFromCounts(
            counts=n_node,
            total_nodes=orbital_feature.shape[0],
        )
        num_graphs = int(n_node.shape[0])
        background_mask = (orbital_role != 2).astype(jnp.float32)
        encoded_background = self.input_linear(orbital_feature)
        background = self.maskedMeanPool(
            node_feature=encoded_background,
            graph_index=graph_index,
            mask=background_mask,
            num_graphs=num_graphs,
        )
        background = self.activate(self.background_linear1(background))
        return self.background_linear2(background)

    def poolActiveNodeFeature(
        self,
        node_feature: jnp.ndarray,
        n_node: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Mean-pool active-only hidden states into one graph embedding.
        """

        graph_index = self.graphIndexFromCounts(
            counts=n_node,
            total_nodes=node_feature.shape[0],
        )
        num_graphs = int(n_node.shape[0])
        active_mask = jnp.ones((node_feature.shape[0],), dtype=jnp.float32)
        active = self.maskedMeanPool(
            node_feature=node_feature,
            graph_index=graph_index,
            mask=active_mask,
            num_graphs=num_graphs,
        )
        active = self.activate(self.active_linear1(active))
        return self.active_linear2(active)

    def buildActiveLowRiskReadout(
        self,
        full_orbital_feature: jnp.ndarray,
        active_rumer_graph: jraph.GraphsTuple,
        orbital_role: jnp.ndarray,
        num_orbitals_per_graph: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Fuse background bypass embeddings with active-only graph embeddings.
        """

        background = self.buildBackgroundBypass(
            orbital_feature=full_orbital_feature,
            n_node=num_orbitals_per_graph,
            orbital_role=orbital_role,
        )
        active_node_feature = self.encodeNodeFeature(active_rumer_graph)
        active = self.poolActiveNodeFeature(
            node_feature=active_node_feature,
            n_node=active_rumer_graph.n_node,
        )
        return jnp.concatenate([background, active], axis=-1)

    def finalHead(self, fused_embedding: jnp.ndarray) -> jnp.ndarray:
        """
        Predict one scalar per graph from fused readout embedding.

        Arguments:
        - fused_embedding: Readout embedding, shape [batch_size, 2 * hidden_dim].

        Returns:
        - Predicted scalar tensor, shape [batch_size].
        """

        output = self.activate(self.output_linear1(fused_embedding))
        output = self.output_linear2(output).squeeze(-1)
        return output

    def __call__(
        self,
        rumer_graph: jraph.GraphsTuple | None,
        rumer_graph_q3_flipped: jraph.GraphsTuple | None,
        orbital_role: jnp.ndarray,
        active_rumer_graph: jraph.GraphsTuple | None = None,
        active_rumer_graph_q3_flipped: jraph.GraphsTuple | None = None,
        full_orbital_feature: jnp.ndarray | None = None,
        full_orbital_feature_q3_flipped: jnp.ndarray | None = None,
        num_orbitals_per_graph: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Run Rumer encoding and parity-symmetrized prediction.

        Arguments:
        - rumer_graph: Batched Rumer graph.
          nodes: [total_orbitals, orbital_feature_dim]
          edges["edge_type"]: [num_edges]
          senders: [num_edges]
          receivers: [num_edges]
          n_node: [batch_size]
        - rumer_graph_q3_flipped: Same topology graph with q3 sign-flipped
          orbital node features.
        - orbital_role: Orbital roles, shape [total_orbitals].

        Returns:
        - Predicted normalized structure weights, shape [batch_size].
        """

        if self.config.mode == "active_only_low_risk":
            if (
                active_rumer_graph is None
                or active_rumer_graph_q3_flipped is None
                or full_orbital_feature is None
                or full_orbital_feature_q3_flipped is None
                or num_orbitals_per_graph is None
            ):
                raise ValueError(
                    "Active-only low-risk Rumer mode requires active graphs, full orbital features, "
                    "and num_orbitals_per_graph."
                )
            fused = self.buildActiveLowRiskReadout(
                full_orbital_feature=full_orbital_feature,
                active_rumer_graph=active_rumer_graph,
                orbital_role=orbital_role,
                num_orbitals_per_graph=num_orbitals_per_graph,
            )
            fused_q3_flipped = self.buildActiveLowRiskReadout(
                full_orbital_feature=full_orbital_feature_q3_flipped,
                active_rumer_graph=active_rumer_graph_q3_flipped,
                orbital_role=orbital_role,
                num_orbitals_per_graph=num_orbitals_per_graph,
            )
        else:
            if rumer_graph is None or rumer_graph_q3_flipped is None:
                raise ValueError("Full Rumer mode requires both full Rumer graphs.")
            node_feature = self.encodeNodeFeature(rumer_graph)
            node_feature_q3_flipped = self.encodeNodeFeature(rumer_graph_q3_flipped)

            fused = self.buildReadoutInput(
                node_feature=node_feature,
                n_node=rumer_graph.n_node,
                orbital_role=orbital_role,
            )
            fused_q3_flipped = self.buildReadoutInput(
                node_feature=node_feature_q3_flipped,
                n_node=rumer_graph_q3_flipped.n_node,
                orbital_role=orbital_role,
            )
        output = self.finalHead(fused)
        output_q3_flipped = self.finalHead(fused_q3_flipped)
        return 0.5 * (output + output_q3_flipped)
