"""SignNet module for Laplacian positional encoding enhancement."""

from __future__ import annotations

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclass
class SignNetConfig:
    """
    Configure the LapPE + SignNet enhancement branch.

    Arguments:
    - phi_hidden: Hidden width of the per-eigenpair encoder phi.
    - phi_out: Output width of phi for one eigenpair block.
    - phi_layers: Number of layers in phi MLP.
    - rho_hidden: Hidden width of the aggregation encoder rho.
    - out_dim: Final SignNet feature dimension concatenated to atom features.
    - rho_layers: Number of layers in rho MLP.
    """

    phi_hidden: int
    phi_out: int
    phi_layers: int
    rho_hidden: int
    out_dim: int
    rho_layers: int


class SignNetMLP(nnx.Module):
    """
    Small feed-forward network used inside SignNet.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        rngs: nnx.Rngs,
    ):
        """
        Initialize one MLP.

        Arguments:
        - input_dim: Input feature dimension.
        - hidden_dim: Hidden feature dimension.
        - output_dim: Output feature dimension.
        - num_layers: Number of layers.
        - rngs: Flax NNX RNG container.
        """

        if int(num_layers) < 1:
            raise ValueError("SignNetMLP.num_layers must be at least 1.")
        self.num_layers = int(num_layers)
        widths = [input_dim]
        if self.num_layers > 1:
            widths.extend([hidden_dim] * (self.num_layers - 1))
        widths.append(output_dim)
        self.layers = [
            nnx.Linear(widths[index], widths[index + 1], rngs=rngs)
            for index in range(self.num_layers)
        ]

    def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Run the MLP on one tensor with arbitrary leading dimensions.
        """

        hidden = value
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < len(self.layers) - 1:
                hidden = jax.nn.relu(hidden)
        return hidden


class SignNet(nnx.Module):
    """
    JAX + Flax NNX implementation of the old PyTorch SignNet.
    """

    def __init__(
        self,
        k: int,
        config: SignNetConfig,
        rngs: nnx.Rngs,
    ):
        """
        Initialize SignNet.

        Arguments:
        - k: Number of Laplacian eigenvectors used.
        - config: SignNetConfig.
        - rngs: Flax NNX RNG container.
        """

        self.k = int(k)
        self.config = config
        self.phi = SignNetMLP(
            input_dim=2,
            hidden_dim=config.phi_hidden,
            output_dim=config.phi_out,
            num_layers=config.phi_layers,
            rngs=rngs,
        )
        self.rho = SignNetMLP(
            input_dim=max(self.k, 1) * config.phi_out,
            hidden_dim=config.rho_hidden,
            output_dim=config.out_dim,
            num_layers=config.rho_layers,
            rngs=rngs,
        )

    def __call__(self, lap_evecs: jnp.ndarray, lap_evals: jnp.ndarray) -> jnp.ndarray:
        """
        Encode Laplacian eigenvectors/eigenvalues into per-node sign-invariant features.

        Arguments:
        - lap_evecs: Node-aligned eigenvectors, shape [num_nodes, k].
        - lap_evals: Node-aligned repeated eigenvalues, shape [num_nodes, k].

        Returns:
        - SignNet feature tensor, shape [num_nodes, out_dim].
        """

        if self.k <= 0:
            return jnp.zeros(
                (lap_evecs.shape[0], self.config.out_dim),
                dtype=lap_evecs.dtype,
            )

        x_pos = jnp.stack([lap_evecs, lap_evals], axis=-1)
        x_neg = jnp.stack([-lap_evecs, lap_evals], axis=-1)

        flat_pos = x_pos.reshape((-1, 2))
        flat_neg = x_neg.reshape((-1, 2))
        encoded = self.phi(flat_pos) + self.phi(flat_neg)
        encoded = encoded.reshape((lap_evecs.shape[0], self.k * self.config.phi_out))
        return self.rho(encoded)
