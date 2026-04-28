"""Atom-level encoder with O(3)-equivariant 0e+1o updates via e3nn-jax tensor products."""

from dataclasses import dataclass
from typing import Optional

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph

from model.norms import IrrepsLayerNorm
from model.signnet import SignNet, SignNetConfig

# Reduce GPU tensor-product equivariance drift by opting into the highest
# available matmul precision globally before the atom encoder traces execute.
jax.config.update("jax_default_matmul_precision", "highest")


@nnx.remat
def runAtomMessageLayer(
    layer: "ScalarVectorEquivariantMessageLayer",
    scalar_feature: jnp.ndarray,
    vector_feature: jnp.ndarray,
    positions: jnp.ndarray,
    senders: jnp.ndarray,
    receivers: jnp.ndarray,
    pair_feature: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rematerialize one atom message layer during backward to reduce activation memory.
    """

    return layer(
        scalar_feature=scalar_feature,
        vector_feature=vector_feature,
        positions=positions,
        senders=senders,
        receivers=receivers,
        pair_feature=pair_feature,
    )


@dataclass
class AtomEncoderConfig:
    """
    Configure atom-level encoder hyper-parameters.

    Purpose:
    - Define scalar channel sizes and geometric message settings.

    Arguments:
    - input_feature_dim: Atom scalar input dimension from dataset.
    - scalar_dim: Number of 0e scalar channels.
    - vector_dim: Number of 1o vector channels.
    - radial_dim: Hidden size of radial MLP.
    - layers: Number of scalar message passing layers.
    - max_atomic_number: Max atomic number of embedding table.
    - radial_min: Minimum radial basis center.
    - radial_max: Maximum radial basis center.
    - radial_basis: Number of Gaussian radial basis functions.
    - lmax: Maximum degree for spherical harmonics used in tensor products.
      Hard requirement in this project: lmax must be 1, producing 0e+1o.
    - lap_pe_k: Number of non-trivial Laplacian eigenvectors used by SignNet.
    - signnet: Optional SignNet enhancement config.
    - edge_content_modulation: Whether edge pair features modulate sender content
      before tensor products, in addition to the existing edge-weight path.
    - edge_content_hidden_dim: Hidden width used by edge-content modulation MLP.
    - edge_content_mode: Content modulation rule. "legacy" reproduces the current
      strong multiplicative path; "stable_residual" applies a bounded residual
      modulation that stays closer to the baseline encoder early in training.
    - edge_content_residual_scale: Maximum residual modulation amplitude used by
      the stable_residual edge-content path.
    """

    input_feature_dim: int
    scalar_dim: int
    vector_dim: int
    radial_dim: int
    layers: int
    max_atomic_number: int
    radial_min: float
    radial_max: float
    radial_basis: int
    lmax: int = 1
    lap_pe_k: int = 0
    signnet: Optional[SignNetConfig] = None
    edge_content_modulation: bool = False
    edge_content_hidden_dim: int = 0
    edge_content_mode: str = "legacy"
    edge_content_residual_scale: float = 0.1


class RadialEdgeConditioner(nnx.Module):
    """
    Build edge-wise radial conditioning weights from distances and pair features.
    """

    def __init__(self, config: AtomEncoderConfig, output_dim: int, rngs: nnx.Rngs):
        """
        Initialize radial MLP.

        Arguments:
        - config: AtomEncoderConfig.
        - output_dim: Output channel dimension.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.output_dim = output_dim
        width = (config.radial_max - config.radial_min) / max(config.radial_basis - 1, 1)
        self.gamma = 1.0 / max(width * width, 1e-8)
        self.linear1 = nnx.Linear(config.radial_basis + 2, config.radial_dim, rngs=rngs)
        self.linear2 = nnx.Linear(config.radial_dim, output_dim, rngs=rngs)

    def activate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - value: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(value)

    def radialBasis(self, distance: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances on Gaussian radial basis.

        Arguments:
        - distance: Edge distance tensor, shape [num_edges].

        Returns:
        - Radial basis tensor, shape [num_edges, radial_basis].
        """

        centers = jnp.linspace(
            self.config.radial_min,
            self.config.radial_max,
            self.config.radial_basis,
            dtype=distance.dtype,
        )
        diff = distance[:, None] - centers[None, :]
        return jnp.exp(-self.gamma * diff * diff)

    def __call__(self, distance: jnp.ndarray, pair_feature: jnp.ndarray) -> jnp.ndarray:
        """
        Build radial conditioning weights.

        Arguments:
        - distance: Edge distance tensor, shape [num_edges].
        - pair_feature: Edge pair feature tensor, shape [num_edges, 2].

        Returns:
        - Edge weight tensor, shape [num_edges, output_dim].
        """

        basis = self.radialBasis(distance)
        context = jnp.concatenate([basis, pair_feature], axis=-1)
        with jax.default_matmul_precision("highest"):
            hidden = self.activate(self.linear1(context))
            return self.linear2(hidden)


class ScalarVectorEquivariantMessageLayer(nnx.Module):
    """
    One tensor-product message layer that updates both 0e scalar and 1o vector channels.
    """

    def __init__(self, config: AtomEncoderConfig, rngs: nnx.Rngs):
        """
        Initialize one scalar-vector equivariant layer.

        Arguments:
        - config: AtomEncoderConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        if int(config.lmax) != 1:
            raise ValueError(
                "AtomEncoderConfig.lmax must be 1 in this project to enforce angular irreps 0e+1o."
            )

        self.scalar_irreps = e3nn.Irreps(f"{config.scalar_dim}x0e")
        self.vector_irreps = e3nn.Irreps(f"{config.vector_dim}x1o")
        self.node_irreps = self.scalar_irreps + self.vector_irreps
        self.angular_irreps = e3nn.Irreps("1x0e+1x1o")
        (
            self.tp_irreps,
            self.tp_scalar_dim,
            self.tp_vector_mul,
        ) = self.inferTensorProductMetadata()
        self.tp_dim = int(self.tp_irreps.dim)
        self.tp_weight_dim = int(self.tp_scalar_dim + self.tp_vector_mul)

        self.edge_conditioner = RadialEdgeConditioner(
            config=config,
            output_dim=self.tp_weight_dim,
            rngs=rngs,
        )
        self.scalar_message_linear = nnx.Linear(self.tp_scalar_dim, config.scalar_dim, rngs=rngs)
        self.update_linear1 = nnx.Linear(2 * config.scalar_dim, config.scalar_dim, rngs=rngs)
        self.update_linear2 = nnx.Linear(config.scalar_dim, config.scalar_dim, rngs=rngs)
        self.norm = IrrepsLayerNorm(self.node_irreps, rngs=rngs)
        self.vector_gate_linear = nnx.Linear(config.scalar_dim, config.vector_dim, rngs=rngs)
        self.vector_message_weight = nnx.Param(
            self.initializeChannelMixWeight(
                rngs=rngs,
                in_channels=max(self.tp_vector_mul, 1),
                out_channels=config.vector_dim,
            )
        )
        self.vector_update_weight = nnx.Param(
            self.initializeChannelMixWeight(
                rngs=rngs,
                in_channels=2 * config.vector_dim,
                out_channels=config.vector_dim,
            )
        )
        self.edge_content_modulation = bool(config.edge_content_modulation)
        self.edge_content_mode = str(config.edge_content_mode)
        self.edge_content_residual_scale = float(config.edge_content_residual_scale)
        if self.edge_content_modulation:
            valid_modes = {
                "legacy",
                "stable_residual",
            }
            if self.edge_content_mode not in valid_modes:
                raise ValueError(
                    "AtomEncoderConfig.edge_content_mode must be one of "
                    f"{sorted(valid_modes)}, got {self.edge_content_mode}."
                )
            edge_hidden_dim = int(
                config.edge_content_hidden_dim if config.edge_content_hidden_dim > 0 else config.radial_dim
            )
            self.edge_content_linear1 = nnx.Linear(2, edge_hidden_dim, rngs=rngs)
            self.edge_scalar_gate_linear = nnx.Linear(edge_hidden_dim, config.scalar_dim, rngs=rngs)
            self.edge_scalar_shift_linear = nnx.Linear(edge_hidden_dim, config.scalar_dim, rngs=rngs)
            self.edge_vector_gate_linear = nnx.Linear(edge_hidden_dim, config.vector_dim, rngs=rngs)
        else:
            self.edge_content_linear1 = None
            self.edge_scalar_gate_linear = None
            self.edge_scalar_shift_linear = None
            self.edge_vector_gate_linear = None

    def initializeChannelMixWeight(
        self,
        rngs: nnx.Rngs,
        in_channels: int,
        out_channels: int,
    ) -> jnp.ndarray:
        """
        Initialize channel-mixing matrix used on vector multiplicity axis.

        Arguments:
        - rngs: Flax NNX random container.
        - in_channels: Input channel count.
        - out_channels: Output channel count.

        Returns:
        - Weight matrix, shape [in_channels, out_channels].
        """

        scale = 1.0 / jnp.sqrt(float(max(in_channels, 1)))
        return jax.random.normal(rngs.params(), shape=(in_channels, out_channels)) * scale

    def activate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - value: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(value)

    def inferTensorProductMetadata(self) -> tuple[e3nn.Irreps, int, int]:
        """
        Infer tensor-product output metadata for 0e+1o message path.

        Returns:
        - tp_irreps: Tensor-product output irreps after filtering to 0e+1o.
        - tp_scalar_dim: Flattened dimension of 0e part.
        - tp_vector_mul: Multiplicity of 1o part.
        """

        sender_state = e3nn.IrrepsArray(
            self.node_irreps,
            jnp.zeros((1, self.node_irreps.dim), dtype=jnp.float32),
        )
        angular_state = e3nn.IrrepsArray(
            self.angular_irreps,
            jnp.zeros((1, self.angular_irreps.dim), dtype=jnp.float32),
        )
        tp = e3nn.tensor_product(
            sender_state,
            angular_state,
            filter_ir_out=[e3nn.Irrep("0e"), e3nn.Irrep("1o")],
        )
        scalar_dim = int(tp.filter(keep="0e").irreps.dim)
        vector_dim = int(tp.filter(keep="1o").irreps.dim // 3)
        return tp.irreps, scalar_dim, vector_dim

    def angularState(self, direction: jnp.ndarray) -> e3nn.IrrepsArray:
        """
        Build spherical harmonics angular tensor.

        Arguments:
        - direction: Unit edge direction vectors, shape [num_edges, 3].

        Returns:
        - Angular irreps tensor.
        """

        return e3nn.spherical_harmonics(
            self.angular_irreps,
            direction,
            normalize=True,
            normalization="component",
        )

    def packNodeState(
        self,
        scalar_feature: jnp.ndarray,
        vector_feature: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        """
        Pack scalar and vector channels into one irreps array.

        Arguments:
        - scalar_feature: Scalar channels, shape [num_nodes, scalar_dim].
        - vector_feature: Vector channels, shape [num_nodes, vector_dim, 3].

        Returns:
        - Node irreps array with irreps scalar+vector.
        """

        packed = jnp.concatenate(
            [scalar_feature, vector_feature.reshape(vector_feature.shape[0], -1)],
            axis=-1,
        )
        return e3nn.IrrepsArray(self.node_irreps, packed)

    def tensorProductMessage(
        self,
        sender_scalar: jnp.ndarray,
        sender_vector: jnp.ndarray,
        angular_state: e3nn.IrrepsArray,
        pair_feature: jnp.ndarray | None = None,
    ) -> e3nn.IrrepsArray:
        """
        Compute 0e+1o tensor-product message before radial conditioning.

        Arguments:
        - sender_scalar: Sender scalar tensor, shape [num_edges, scalar_dim].
        - sender_vector: Sender vector tensor, shape [num_edges, vector_dim, 3].
        - angular_state: Angular irreps tensor for edges.
        - pair_feature: Edge pair feature tensor, shape [num_edges, 2].

        Returns:
        - Tensor-product irreps array with irreps 0e+1o.
        """
        if self.edge_content_modulation:
            if pair_feature is None:
                raise ValueError(
                    "ScalarVectorEquivariantMessageLayer requires pair_feature when edge_content_modulation is enabled."
                )
            edge_hidden = self.activate(self.edge_content_linear1(pair_feature))
            if self.edge_content_mode == "legacy":
                scalar_gate = jax.nn.sigmoid(self.edge_scalar_gate_linear(edge_hidden))
                scalar_shift = self.edge_scalar_shift_linear(edge_hidden)
                vector_gate = jax.nn.sigmoid(self.edge_vector_gate_linear(edge_hidden))
                sender_scalar = sender_scalar * (1.0 + scalar_gate) + scalar_shift
                sender_vector = sender_vector * (1.0 + vector_gate[:, :, None])
            elif self.edge_content_mode == "stable_residual":
                residual_scale = jnp.asarray(
                    self.edge_content_residual_scale,
                    dtype=sender_scalar.dtype,
                )
                scalar_gate = 1.0 + residual_scale * jnp.tanh(
                    self.edge_scalar_gate_linear(edge_hidden)
                )
                scalar_shift = residual_scale * jnp.tanh(
                    self.edge_scalar_shift_linear(edge_hidden)
                )
                vector_gate = 1.0 + residual_scale * jnp.tanh(
                    self.edge_vector_gate_linear(edge_hidden)
                )
                sender_scalar = sender_scalar * scalar_gate + scalar_shift
                sender_vector = sender_vector * vector_gate[:, :, None]
            else:
                raise ValueError(
                    "ScalarVectorEquivariantMessageLayer received unsupported "
                    f"edge_content_mode={self.edge_content_mode}."
                )

        sender_state = self.packNodeState(sender_scalar, sender_vector)
        with jax.default_matmul_precision("highest"):
            return e3nn.tensor_product(
                sender_state,
                angular_state,
                filter_ir_out=[e3nn.Irrep("0e"), e3nn.Irrep("1o")],
            )

    def channelMix(
        self,
        value: jnp.ndarray,
        weight: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Mix vector channels on multiplicity axis while preserving 1o equivariance.

        Arguments:
        - value: Vector tensor, shape [num_nodes, in_channels, 3].
        - weight: Channel-mixing matrix, shape [in_channels, out_channels].

        Returns:
        - Mixed vector tensor, shape [num_nodes, out_channels, 3].
        """

        with jax.default_matmul_precision("highest"):
            return jnp.einsum("nid,io->nod", value, weight)

    def unpackNodeState(self, state: e3nn.IrrepsArray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Unpack one node irreps array into scalar and vector tensors.

        Arguments:
        - state: Irreps array with node irreps scalar+vector.

        Returns:
        - scalar_feature: Shape [num_nodes, scalar_dim].
        - vector_feature: Shape [num_nodes, vector_dim, 3].
        """

        scalar_part = state.filter(keep="0e").array
        vector_part = state.filter(keep="1o").chunks[0]
        return scalar_part, vector_part

    def __call__(
        self,
        scalar_feature: jnp.ndarray,
        vector_feature: jnp.ndarray,
        positions: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        pair_feature: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Run one 0e+1o equivariant update.

        Arguments:
        - scalar_feature: Node scalar channels, shape [num_nodes, scalar_dim].
        - vector_feature: Node vector channels, shape [num_nodes, vector_dim, 3].
        - positions: Atom coordinates, shape [num_nodes, 3].
        - senders: Edge sender indices, shape [num_edges].
        - receivers: Edge receiver indices, shape [num_edges].
        - pair_feature: Edge pair features, shape [num_edges, 2].

        Returns:
        - updated_scalar: Shape [num_nodes, scalar_dim].
        - updated_vector: Shape [num_nodes, vector_dim, 3].
        """

        with jax.default_matmul_precision("highest"):
            edge_vector = positions[receivers] - positions[senders]
            distance = jnp.linalg.norm(edge_vector, axis=-1)
            direction = edge_vector / jnp.maximum(distance[:, None], 1e-8)

            angular_state = self.angularState(direction)
            tp_state = self.tensorProductMessage(
                sender_scalar=scalar_feature[senders],
                sender_vector=vector_feature[senders],
                angular_state=angular_state,
                pair_feature=pair_feature,
            )
            edge_weight = self.edge_conditioner(distance=distance, pair_feature=pair_feature)
            scalar_weight = edge_weight[:, : self.tp_scalar_dim]
            vector_weight = edge_weight[:, self.tp_scalar_dim :]

            scalar_chunk = tp_state.filter(keep="0e").chunks[0]
            vector_chunk = tp_state.filter(keep="1o").chunks[0]
            weighted_scalar = scalar_chunk * scalar_weight[:, :, None]
            weighted_vector = vector_chunk * vector_weight[:, :, None]
            edge_message = jnp.concatenate(
                [
                    weighted_scalar.reshape(weighted_scalar.shape[0], -1),
                    weighted_vector.reshape(weighted_vector.shape[0], -1),
                ],
                axis=-1,
            )

            aggregated_tp = jraph.segment_sum(
                edge_message,
                receivers,
                num_segments=scalar_feature.shape[0],
            )
            aggregated_state = e3nn.IrrepsArray(self.tp_irreps, aggregated_tp)
            aggregated_scalar_raw = aggregated_state.filter(keep="0e").array
            aggregated_vector_raw = aggregated_state.filter(keep="1o").chunks[0]

            aggregated_scalar = self.scalar_message_linear(aggregated_scalar_raw)
            if self.tp_vector_mul > 0:
                aggregated_vector = self.channelMix(
                    aggregated_vector_raw,
                    self.vector_message_weight.value,
                )
            else:
                aggregated_vector = jnp.zeros_like(vector_feature)

            update_input = jnp.concatenate([scalar_feature, aggregated_scalar], axis=-1)
            update = self.activate(self.update_linear1(update_input))
            update = self.update_linear2(update)
            scalar_residual = scalar_feature + update

            vector_context = jnp.concatenate([vector_feature, aggregated_vector], axis=1)
            vector_delta = self.channelMix(vector_context, self.vector_update_weight.value)
            vector_gate = jax.nn.sigmoid(self.vector_gate_linear(scalar_residual))
            vector_residual = vector_feature + vector_gate[:, :, None] * vector_delta

            normalized_state = self.norm(
                self.packNodeState(
                    scalar_feature=scalar_residual,
                    vector_feature=vector_residual,
                )
            )
        updated_scalar, updated_vector = self.unpackNodeState(normalized_state)
        return updated_scalar, updated_vector


class AtomE3Encoder(nnx.Module):
    """
    Atom-level encoder for end-to-end pipeline.

    Structure guarantee:
    - Atom updates keep 0e and 1o channels as formal representation.
    - Angular path is fixed to lmax=1 with angular irreps 0e+1o.
    """

    def __init__(self, config: AtomEncoderConfig, rngs: nnx.Rngs):
        """
        Initialize atom encoder.

        Arguments:
        - config: AtomEncoderConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.number_embedding = nnx.Embed(
            num_embeddings=config.max_atomic_number + 1,
            features=config.scalar_dim,
            rngs=rngs,
        )
        self.signnet = None
        signnet_out_dim = 0
        if (config.signnet is not None) and (int(config.lap_pe_k) > 0):
            self.signnet = SignNet(
                k=int(config.lap_pe_k),
                config=config.signnet,
                rngs=rngs,
            )
            signnet_out_dim = int(config.signnet.out_dim)
        self.feature_linear1 = nnx.Linear(
            config.input_feature_dim + signnet_out_dim,
            config.scalar_dim,
            rngs=rngs,
        )
        self.feature_linear2 = nnx.Linear(config.scalar_dim, config.scalar_dim, rngs=rngs)
        self.layers = [
            ScalarVectorEquivariantMessageLayer(config=config, rngs=rngs)
            for _ in range(config.layers)
        ]

    def activate(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - value: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(value)

    def initialScalar(
        self,
        atom_feature: jnp.ndarray,
        atom_number: jnp.ndarray,
        lap_evecs: jnp.ndarray | None = None,
        lap_evals: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Build initial scalar channels from atom number and input features.

        Arguments:
        - atom_feature: Input atom features, shape [num_nodes, input_feature_dim].
        - atom_number: Atomic numbers, shape [num_nodes].
        - lap_evecs: Node-aligned Laplacian eigenvectors, shape [num_nodes, lap_pe_k].
        - lap_evals: Node-aligned repeated Laplacian eigenvalues, shape [num_nodes, lap_pe_k].

        Returns:
        - Initial scalar channels, shape [num_nodes, scalar_dim].
        """

        clipped = jnp.clip(atom_number, 0, self.config.max_atomic_number)
        number_part = self.number_embedding(clipped)
        enhanced_feature = atom_feature
        if self.signnet is not None:
            if (lap_evecs is None) or (lap_evals is None):
                raise ValueError(
                    "AtomE3Encoder requires lap_evecs and lap_evals when SignNet is enabled."
                )
            signnet_feature = self.signnet(lap_evecs=lap_evecs, lap_evals=lap_evals)
            enhanced_feature = jnp.concatenate([atom_feature, signnet_feature], axis=-1)
        with jax.default_matmul_precision("highest"):
            feature_part = self.activate(self.feature_linear1(enhanced_feature))
            feature_part = self.feature_linear2(feature_part)
        return number_part + feature_part

    def __call__(
        self,
        atom_graph: jraph.GraphsTuple,
        lap_evecs: jnp.ndarray | None = None,
        lap_evals: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """
        Run atom encoder forward pass.

        Arguments:
        - atom_graph: Batched atom graph.
          nodes["features"]: [total_atoms, input_feature_dim]
          nodes["numbers"]: [total_atoms]
          nodes["positions"]: [total_atoms, 3]
          edges["pair"]: [total_edges, 2]
          senders: [total_edges]
          receivers: [total_edges]
        - lap_evecs: Node-aligned Laplacian eigenvectors, shape [total_atoms, lap_pe_k].
        - lap_evals: Node-aligned repeated Laplacian eigenvalues, shape [total_atoms, lap_pe_k].

        Returns:
        - Dictionary:
          scalar: [total_atoms, scalar_dim]
          vector: [total_atoms, vector_dim, 3]
          positions: [total_atoms, 3]
        """

        atom_feature = atom_graph.nodes["features"]
        atom_number = atom_graph.nodes["numbers"]
        positions = atom_graph.nodes["positions"]
        pair_feature = atom_graph.edges["pair"]
        senders = atom_graph.senders
        receivers = atom_graph.receivers

        scalar_feature = self.initialScalar(
            atom_feature=atom_feature,
            atom_number=atom_number,
            lap_evecs=lap_evecs,
            lap_evals=lap_evals,
        )
        vector_feature = jnp.zeros(
            (scalar_feature.shape[0], self.config.vector_dim, 3),
            dtype=scalar_feature.dtype,
        )
        for layer in self.layers:
            scalar_feature, vector_feature = runAtomMessageLayer(
                layer=layer,
                scalar_feature=scalar_feature,
                vector_feature=vector_feature,
                positions=positions,
                senders=senders,
                receivers=receivers,
                pair_feature=pair_feature,
            )

        return {
            "scalar": scalar_feature,
            "vector": vector_feature,
            "positions": positions,
        }
