"""Irreps-aware normalization modules for JAX/Flax NNX."""

from __future__ import annotations

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax.numpy as jnp


class IrrepsLayerNorm(nnx.Module):
    """
    Irreps-safe LayerNorm/RMSNorm matching the old PyTorch E3nnVB behavior.

    Rules:
    - Scalars (l=0): standard LayerNorm over the whole scalar block.
    - Non-scalars (l>0): RMS normalization only, without mean subtraction.
    - Affine scale is applied per copy; scalar bias is only applied to l=0 blocks.
    """

    def __init__(
        self,
        irreps: str | e3nn.Irreps,
        rngs: nnx.Rngs | None = None,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        self.irreps = e3nn.Irreps(irreps)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.blocks = []

        cursor = 0
        scale_index = []
        scalar_component_index = []
        scalar_bias_index = []
        copy_cursor = 0
        scalar_copy_cursor = 0

        for mul, ir in self.irreps:
            multiplicity = int(mul)
            component_dim = int(ir.dim)
            block_dim = multiplicity * component_dim
            block_slice = slice(cursor, cursor + block_dim)
            angular_degree = int(ir.l)
            self.blocks.append((block_slice, multiplicity, component_dim, angular_degree))

            scale_index.append(
                jnp.repeat(
                    jnp.arange(copy_cursor, copy_cursor + multiplicity, dtype=jnp.int32),
                    component_dim,
                )
            )

            if angular_degree == 0:
                scalar_component_index.append(
                    jnp.arange(cursor, cursor + block_dim, dtype=jnp.int32)
                )
                scalar_bias_index.append(
                    jnp.repeat(
                        jnp.arange(
                            scalar_copy_cursor,
                            scalar_copy_cursor + multiplicity,
                            dtype=jnp.int32,
                        ),
                        component_dim,
                    )
                )
                scalar_copy_cursor += multiplicity

            copy_cursor += multiplicity
            cursor += block_dim

        self.dim = cursor
        self.scale_index = tuple(
            map(
                int,
                (
                    jnp.concatenate(scale_index, axis=0).tolist()
                    if scale_index
                    else []
                ),
            )
        )
        self.scalar_component_index = tuple(
            map(
                int,
                (
                    jnp.concatenate(scalar_component_index, axis=0).tolist()
                    if scalar_component_index
                    else []
                ),
            )
        )
        self.scalar_bias_index = tuple(
            map(
                int,
                (
                    jnp.concatenate(scalar_bias_index, axis=0).tolist()
                    if scalar_bias_index
                    else []
                ),
            )
        )
        self.num_copies = int(copy_cursor)
        self.num_scalar_copies = int(scalar_copy_cursor)

        if self.affine:
            self.scale_per_copy = nnx.Param(jnp.ones((self.num_copies,), dtype=jnp.float32))
            if self.num_scalar_copies > 0:
                self.bias_scalar = nnx.Param(
                    jnp.zeros((self.num_scalar_copies,), dtype=jnp.float32)
                )
            else:
                self.bias_scalar = None
        else:
            self.scale_per_copy = None
            self.bias_scalar = None

    def normalizePacked(self, packed: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize one packed irreps tensor of shape [batch, dim].
        """

        normalized_blocks = []
        for block_slice, multiplicity, component_dim, angular_degree in self.blocks:
            block = packed[:, block_slice].reshape(-1, multiplicity, component_dim)
            if angular_degree == 0:
                mean = jnp.mean(block, axis=(1, 2), keepdims=True)
                variance = jnp.mean(jnp.square(block - mean), axis=(1, 2), keepdims=True)
                block = (block - mean) / jnp.sqrt(variance + self.eps)
            else:
                mean_square = jnp.mean(jnp.square(block), axis=(1, 2), keepdims=True)
                block = block / jnp.sqrt(mean_square + self.eps)
            normalized_blocks.append(block.reshape(block.shape[0], multiplicity * component_dim))

        output = jnp.concatenate(normalized_blocks, axis=-1)

        if not self.affine:
            return output

        scale_index = jnp.asarray(self.scale_index, dtype=jnp.int32)
        scale = self.scale_per_copy.value[scale_index][None, :]
        output = output * scale

        if self.bias_scalar is None or not self.scalar_component_index:
            return output

        bias_full = jnp.zeros((self.dim,), dtype=output.dtype)
        scalar_bias_index = jnp.asarray(self.scalar_bias_index, dtype=jnp.int32)
        scalar_component_index = jnp.asarray(self.scalar_component_index, dtype=jnp.int32)
        scalar_bias = self.bias_scalar.value[scalar_bias_index]
        bias_full = bias_full.at[scalar_component_index].set(scalar_bias)
        return output + bias_full[None, :]

    def __call__(self, value: e3nn.IrrepsArray | jnp.ndarray) -> e3nn.IrrepsArray | jnp.ndarray:
        """
        Normalize an IrrepsArray or a raw packed array with matching dimension.
        """

        if isinstance(value, e3nn.IrrepsArray):
            normalized = self.normalizePacked(value.array)
            return e3nn.IrrepsArray(value.irreps, normalized)

        normalized = self.normalizePacked(value)
        return normalized
