"""Orbital projection, slot construction, and slot matching modules."""

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph


@dataclass
class OrbitalProjectionConfig:
    """
    Configure orbital projection and active slot matching.

    Arguments:
    - scalar_dim: Atom scalar channel size.
    - vector_dim: Atom 1o vector channel size.
    - slot_dim: Orbital slot candidate feature size.
    - slot_query_dim: Active slot query feature size.
    - slot_embedding_dim: Learnable slot id embedding size.
    - max_active_slots: Maximum active slots per atom.
    - orbital_feature_dim: Final orbital node feature size.
    """

    scalar_dim: int
    vector_dim: int
    slot_dim: int
    slot_query_dim: int
    slot_embedding_dim: int
    max_active_slots: int
    orbital_feature_dim: int


class LocalOrbitalProjector(nnx.Module):
    """
    Project vector channels onto local atom frames.
    """

    def __init__(self):
        """
        Initialize local frame projector.
        """

    def normalizeVector(self, value: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize vectors along the last axis.

        Arguments:
        - value: Input tensor, shape [..., 3].

        Returns:
        - Normalized tensor, shape [..., 3].
        """

        norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
        return value / jnp.maximum(norm, 1e-8)

    def chooseReferenceAxis(self, direction: jnp.ndarray) -> jnp.ndarray:
        """
        Choose least-aligned Cartesian axis against one direction vector.

        Arguments:
        - direction: One unit vector, shape [3].

        Returns:
        - Reference axis, shape [3].
        """

        axes = jnp.eye(3, dtype=direction.dtype)
        score = jnp.abs(axes @ direction)
        index = int(jnp.argmin(score).item())
        return axes[index]

    def orthogonalDirection(self, direction: jnp.ndarray) -> jnp.ndarray:
        """
        Build one normalized vector orthogonal to input direction.

        Arguments:
        - direction: One unit vector, shape [3].

        Returns:
        - Orthogonal unit vector, shape [3].
        """

        reference = self.chooseReferenceAxis(direction)
        orthogonal = reference - jnp.dot(reference, direction) * direction
        return self.normalizeVector(orthogonal)

    def buildFrameForOneGraph(self, positions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build local frames for one molecular graph.

        Arguments:
        - positions: Atom coordinates for one graph, shape [num_atoms, 3].

        Returns:
        - e1: First local axis, shape [num_atoms, 3].
        - e2: Second local axis, shape [num_atoms, 3].
        - e3: Third local axis, shape [num_atoms, 3].
        """

        num_atoms = positions.shape[0]
        if num_atoms == 0:
            empty = jnp.zeros((0, 3), dtype=positions.dtype)
            return empty, empty, empty

        distance = jnp.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=-1,
        )
        distance = distance + jnp.eye(num_atoms, dtype=positions.dtype) * 1e6
        nearest = jnp.argsort(distance, axis=-1)

        e1_list = []
        e2_list = []
        e3_list = []
        default_e1 = jnp.asarray([1.0, 0.0, 0.0], dtype=positions.dtype)
        default_e2 = jnp.asarray([0.0, 1.0, 0.0], dtype=positions.dtype)
        default_e3 = jnp.asarray([0.0, 0.0, 1.0], dtype=positions.dtype)

        for atom_id in range(num_atoms):
            neighbor_first = int(nearest[atom_id, 0].item())
            r_first = positions[neighbor_first] - positions[atom_id]
            e1 = self.normalizeVector(r_first)

            if num_atoms >= 3:
                neighbor_second = int(nearest[atom_id, 1].item())
                r_second = positions[neighbor_second] - positions[atom_id]
                projected = r_second - jnp.dot(r_second, e1) * e1
                if float(jnp.linalg.norm(projected)) < 1e-8:
                    e2 = self.orthogonalDirection(e1)
                else:
                    e2 = self.normalizeVector(projected)
            elif num_atoms == 2:
                e2 = self.orthogonalDirection(e1)
            else:
                e1 = default_e1
                e2 = default_e2
                e3 = default_e3
                e1_list.append(e1)
                e2_list.append(e2)
                e3_list.append(e3)
                continue

            e3 = self.normalizeVector(jnp.cross(e1, e2))
            e2 = self.normalizeVector(jnp.cross(e3, e1))
            e1_list.append(e1)
            e2_list.append(e2)
            e3_list.append(e3)

        return (
            jnp.stack(e1_list, axis=0),
            jnp.stack(e2_list, axis=0),
            jnp.stack(e3_list, axis=0),
        )

    def buildLocalFrames(
        self,
        positions: jnp.ndarray,
        num_atoms_per_graph: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Build local frames for a batched atom tensor.

        Arguments:
        - positions: Batched coordinates, shape [total_atoms, 3].
        - num_atoms_per_graph: Atom counts, shape [batch_size].

        Returns:
        - e1: Shape [total_atoms, 3].
        - e2: Shape [total_atoms, 3].
        - e3: Shape [total_atoms, 3].
        """

        e1_parts = []
        e2_parts = []
        e3_parts = []
        cursor = 0
        for count in num_atoms_per_graph.tolist():
            block = positions[cursor : cursor + int(count)]
            e1, e2, e3 = self.buildFrameForOneGraph(block)
            e1_parts.append(e1)
            e2_parts.append(e2)
            e3_parts.append(e3)
            cursor += int(count)
        return (
            jnp.concatenate(e1_parts, axis=0),
            jnp.concatenate(e2_parts, axis=0),
            jnp.concatenate(e3_parts, axis=0),
        )

    def projectVectorChannels(
        self,
        vector_feature: jnp.ndarray,
        e1: jnp.ndarray,
        e2: jnp.ndarray,
        e3: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Project vector channels onto local axes.

        Arguments:
        - vector_feature: Atom 1o channels, shape [total_atoms, vector_dim, 3].
        - e1: First local axis, shape [total_atoms, 3].
        - e2: Second local axis, shape [total_atoms, 3].
        - e3: Third local axis, shape [total_atoms, 3].

        Returns:
        - q1: Projection on e1, shape [total_atoms, vector_dim].
        - q2: Projection on e2, shape [total_atoms, vector_dim].
        - q3: Projection on e3, shape [total_atoms, vector_dim].
        """

        q1 = jnp.einsum("ncd,nd->nc", vector_feature, e1)
        q2 = jnp.einsum("ncd,nd->nc", vector_feature, e2)
        q3 = jnp.einsum("ncd,nd->nc", vector_feature, e3)
        return q1, q2, q3

    def __call__(
        self,
        positions: jnp.ndarray,
        vector_feature: jnp.ndarray,
        num_atoms_per_graph: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """
        Compute local frame projections for batched atoms.

        Arguments:
        - positions: Batched coordinates, shape [total_atoms, 3].
        - vector_feature: Batched 1o channels, shape [total_atoms, vector_dim, 3].
        - num_atoms_per_graph: Atom counts per graph, shape [batch_size].

        Returns:
        - Dictionary with keys e1, e2, e3, q1, q2, q3.
        """

        e1, e2, e3 = self.buildLocalFrames(positions, num_atoms_per_graph)
        q1, q2, q3 = self.projectVectorChannels(vector_feature, e1, e2, e3)
        return {"e1": e1, "e2": e2, "e3": e3, "q1": q1, "q2": q2, "q3": q3}


class OrbitalSlotConstructor(nnx.Module):
    """
    Build local orbital candidate features per atom.

    Output candidate order is fixed:
    - candidate 0: s-like
    - candidate 1: p-like branch 1
    - candidate 2: p-like branch 2
    - candidate 3: p-like branch 3
    """

    def __init__(self, config: OrbitalProjectionConfig, rngs: nnx.Rngs):
        """
        Initialize slot constructor.

        Arguments:
        - config: OrbitalProjectionConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.s_linear1 = nnx.Linear(config.scalar_dim, config.slot_dim, rngs=rngs)
        self.s_linear2 = nnx.Linear(config.slot_dim, config.slot_dim, rngs=rngs)
        p_input_dim = config.scalar_dim + config.vector_dim
        self.p1_linear1 = nnx.Linear(p_input_dim, config.slot_dim, rngs=rngs)
        self.p1_linear2 = nnx.Linear(config.slot_dim, config.slot_dim, rngs=rngs)
        self.p2_linear1 = nnx.Linear(p_input_dim, config.slot_dim, rngs=rngs)
        self.p2_linear2 = nnx.Linear(config.slot_dim, config.slot_dim, rngs=rngs)
        self.p3_linear1 = nnx.Linear(p_input_dim, config.slot_dim, rngs=rngs)
        self.p3_linear2 = nnx.Linear(config.slot_dim, config.slot_dim, rngs=rngs)

    def activate(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - tensor: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(tensor)

    def branchFeature(
        self,
        scalar_feature: jnp.ndarray,
        linear1: nnx.Linear,
        linear2: nnx.Linear,
    ) -> jnp.ndarray:
        """
        Build one candidate branch from invariant scalar feature.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].
        - linear1: First linear layer.
        - linear2: Second linear layer.

        Returns:
        - Candidate feature tensor, shape [total_atoms, slot_dim].
        """

        value = self.activate(linear1(scalar_feature))
        value = linear2(value)
        return value

    def __call__(
        self,
        scalar_feature: jnp.ndarray,
        q1: jnp.ndarray,
        q2: jnp.ndarray,
        q3: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """
        Construct local orbital slot candidates per atom.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].
        - q1: Vector projections on local axis e1, shape [total_atoms, vector_dim].
        - q2: Vector projections on local axis e2, shape [total_atoms, vector_dim].
        - q3: Vector projections on local axis e3, shape [total_atoms, vector_dim].

        Returns:
        - Dictionary:
          candidates: [total_atoms, 4, slot_dim]
        """

        p1_input = jnp.concatenate([scalar_feature, q1], axis=-1)
        p2_input = jnp.concatenate([scalar_feature, q2], axis=-1)
        p3_input = jnp.concatenate([scalar_feature, q3], axis=-1)

        phi_s = self.branchFeature(scalar_feature, self.s_linear1, self.s_linear2)
        phi_p1 = self.branchFeature(p1_input, self.p1_linear1, self.p1_linear2)
        phi_p2 = self.branchFeature(p2_input, self.p2_linear1, self.p2_linear2)
        phi_p3 = self.branchFeature(p3_input, self.p3_linear1, self.p3_linear2)
        candidates = jnp.stack([phi_s, phi_p1, phi_p2, phi_p3], axis=1)
        return {"candidates": candidates}


class ActiveSlotMatcher(nnx.Module):
    """
    Match active slots against local orbital candidates.

    This module follows the required equations:
    - u_{i,k} = MLP([h_i^{0e}, Emb(k)])
    - a_{i,k,m} = MLP([u_{i,k}, phi_{i,m}])
    - alpha_{i,k,m} = softmax_m(a_{i,k,m})
    - x_{i,k}^{active} = sum_m alpha_{i,k,m} * phi_{i,m}
    """

    def __init__(self, config: OrbitalProjectionConfig, rngs: nnx.Rngs):
        """
        Initialize active slot matcher.

        Arguments:
        - config: OrbitalProjectionConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.slot_embedding = nnx.Embed(
            num_embeddings=config.max_active_slots,
            features=config.slot_embedding_dim,
            rngs=rngs,
        )
        self.query_linear1 = nnx.Linear(
            config.scalar_dim + config.slot_embedding_dim,
            config.slot_query_dim,
            rngs=rngs,
        )
        self.query_linear2 = nnx.Linear(
            config.slot_query_dim,
            config.slot_query_dim,
            rngs=rngs,
        )
        self.score_linear1 = nnx.Linear(
            config.slot_query_dim + config.slot_dim,
            config.slot_query_dim,
            rngs=rngs,
        )
        self.score_linear2 = nnx.Linear(config.slot_query_dim, 1, rngs=rngs)

    def activate(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - tensor: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(tensor)

    def buildSlotQuery(self, scalar_feature: jnp.ndarray) -> jnp.ndarray:
        """
        Build all slot queries for every atom.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].

        Returns:
        - Slot query tensor, shape [total_atoms, max_active_slots, slot_query_dim].
        """

        total_atoms = scalar_feature.shape[0]
        slot_ids = jnp.arange(self.config.max_active_slots, dtype=jnp.int32)
        slot_embedding = self.slot_embedding(slot_ids)
        scalar_expand = jnp.repeat(
            scalar_feature[:, None, :],
            repeats=self.config.max_active_slots,
            axis=1,
        )
        slot_expand = jnp.repeat(
            slot_embedding[None, :, :],
            repeats=total_atoms,
            axis=0,
        )
        query_input = jnp.concatenate([scalar_expand, slot_expand], axis=-1)
        query = self.activate(self.query_linear1(query_input))
        query = self.query_linear2(query)
        return query

    def matchCandidates(
        self,
        query: jnp.ndarray,
        candidates: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Match slot queries and local orbital candidates.

        Arguments:
        - query: Slot query tensor, shape [total_atoms, max_active_slots, slot_query_dim].
        - candidates: Local orbital candidates, shape [total_atoms, 4, slot_dim].

        Returns:
        - slot_feature: Slot-matched active features, shape [total_atoms, max_active_slots, slot_dim].
        - alpha: Soft matching weights, shape [total_atoms, max_active_slots, 4].
        """

        total_atoms = candidates.shape[0]
        query_expand = jnp.repeat(query[:, :, None, :], repeats=4, axis=2)
        candidate_expand = jnp.repeat(
            candidates[:, None, :, :],
            repeats=self.config.max_active_slots,
            axis=1,
        )
        score_input = jnp.concatenate([query_expand, candidate_expand], axis=-1)
        score_hidden = self.activate(self.score_linear1(score_input))
        score = self.score_linear2(score_hidden).squeeze(-1)
        alpha = jax.nn.softmax(score, axis=-1)
        slot_feature = jnp.sum(alpha[..., None] * candidate_expand, axis=2)
        return slot_feature, alpha

    def __call__(
        self,
        scalar_feature: jnp.ndarray,
        candidates: jnp.ndarray,
        active_slot_capacity: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """
        Build slot-matched active features for each atom.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].
        - candidates: Local candidates, shape [total_atoms, 4, slot_dim].
        - active_slot_capacity: Number of active slots on each atom, shape [total_atoms].

        Returns:
        - Dictionary:
          slot_feature: [total_atoms, max_active_slots, slot_dim]
          alpha: [total_atoms, max_active_slots, 4]
        """

        query = self.buildSlotQuery(scalar_feature)
        slot_feature, alpha = self.matchCandidates(query, candidates)
        slot_id = jnp.arange(self.config.max_active_slots)[None, :]
        mask = (slot_id < active_slot_capacity[:, None]).astype(slot_feature.dtype)
        masked_slot_feature = slot_feature * mask[:, :, None]
        return {"slot_feature": masked_slot_feature, "alpha": alpha}


class AtomToOrbitalMapper(nnx.Module):
    """
    Map atom features into orbital node features.

    Mapping rules keep current scientific semantics:
    - core orbital (role=0): single-atom mapping
    - inactive bonding orbital (role=1): average of two atom features
    - active orbital (role=2): slot-matched feature at the owner atom and slot id
    """

    def __init__(self, config: OrbitalProjectionConfig, rngs: nnx.Rngs):
        """
        Initialize atom-to-orbital mapper.

        Arguments:
        - config: OrbitalProjectionConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.atom_base_linear1 = nnx.Linear(
            config.scalar_dim + 3 * config.vector_dim,
            config.orbital_feature_dim,
            rngs=rngs,
        )
        self.atom_base_linear2 = nnx.Linear(config.orbital_feature_dim, config.orbital_feature_dim, rngs=rngs)
        self.active_linear1 = nnx.Linear(config.slot_dim, config.orbital_feature_dim, rngs=rngs)
        self.active_linear2 = nnx.Linear(config.orbital_feature_dim, config.orbital_feature_dim, rngs=rngs)

    def activate(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """
        Apply SiLU activation.

        Arguments:
        - tensor: Input tensor.

        Returns:
        - Activated tensor.
        """

        return jax.nn.silu(tensor)

    def projectAtomBase(
        self,
        scalar_feature: jnp.ndarray,
        local_vector_feature: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Project atom scalar+local-vector channels to base orbital feature space.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].
        - local_vector_feature: Concatenated local projections [q1,q2,q3],
          shape [total_atoms, 3 * vector_dim].

        Returns:
        - Base atom feature, shape [total_atoms, orbital_feature_dim].
        """

        atom_base_input = jnp.concatenate([scalar_feature, local_vector_feature], axis=-1)
        value = self.activate(self.atom_base_linear1(atom_base_input))
        return self.atom_base_linear2(value)

    def projectActiveSlot(self, slot_feature: jnp.ndarray) -> jnp.ndarray:
        """
        Project slot-matched active feature to orbital feature space.

        Arguments:
        - slot_feature: Slot feature, shape [total_orbitals, slot_dim].

        Returns:
        - Projected active orbital feature, shape [total_orbitals, orbital_feature_dim].
        """

        value = self.activate(self.active_linear1(slot_feature))
        return self.active_linear2(value)

    def __call__(
        self,
        scalar_feature: jnp.ndarray,
        local_vector_feature: jnp.ndarray,
        slot_feature: jnp.ndarray,
        orbital_atom_index: jnp.ndarray,
        orbital_role: jnp.ndarray,
        active_slot_index: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Build orbital node features for a batched Rumer graph.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].
        - local_vector_feature: Concatenated local vector projections,
          shape [total_atoms, 3 * vector_dim].
        - slot_feature: Atom slot features, shape [total_atoms, max_active_slots, slot_dim].
        - orbital_atom_index: Orbital-to-atom map, shape [total_orbitals, 2].
        - orbital_role: Orbital roles, shape [total_orbitals].
        - active_slot_index: Local active slot id for each orbital, shape [total_orbitals].

        Returns:
        - Orbital node features, shape [total_orbitals, orbital_feature_dim].
        """

        base_atom_feature = self.projectAtomBase(
            scalar_feature=scalar_feature,
            local_vector_feature=local_vector_feature,
        )
        first_atom = orbital_atom_index[:, 0]
        second_atom = orbital_atom_index[:, 1]

        core_feature = base_atom_feature[first_atom]
        inactive_feature = 0.5 * (
            base_atom_feature[first_atom] + base_atom_feature[second_atom]
        )

        safe_slot = jnp.clip(active_slot_index, 0, self.config.max_active_slots - 1)
        active_raw = slot_feature[first_atom, safe_slot]
        active_feature = self.projectActiveSlot(active_raw)

        core_mask = (orbital_role == 0)[:, None].astype(base_atom_feature.dtype)
        inactive_mask = (orbital_role == 1)[:, None].astype(base_atom_feature.dtype)
        active_mask = (orbital_role == 2)[:, None].astype(base_atom_feature.dtype)

        return (
            core_mask * core_feature
            + inactive_mask * inactive_feature
            + active_mask * active_feature
        )


class AtomToOrbitalProjectionStack(nnx.Module):
    """
    End-to-end projection stack from atom channels to orbital node features.
    """

    def __init__(self, config: OrbitalProjectionConfig, rngs: nnx.Rngs):
        """
        Initialize projection stack.

        Arguments:
        - config: OrbitalProjectionConfig.
        - rngs: Flax NNX random container.
        """

        self.config = config
        self.local_projector = LocalOrbitalProjector()
        self.slot_constructor = OrbitalSlotConstructor(config=config, rngs=rngs)
        self.slot_matcher = ActiveSlotMatcher(config=config, rngs=rngs)
        self.mapper = AtomToOrbitalMapper(config=config, rngs=rngs)

    def buildOrbitalFeatureFromProjections(
        self,
        scalar_feature: jnp.ndarray,
        q1: jnp.ndarray,
        q2: jnp.ndarray,
        q3: jnp.ndarray,
        active_capacity: jnp.ndarray,
        orbital_atom_index: jnp.ndarray,
        orbital_role: jnp.ndarray,
        active_slot_index: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build one orbital feature tensor from one set of local projections.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].
        - q1: Local projection on e1, shape [total_atoms, vector_dim].
        - q2: Local projection on e2, shape [total_atoms, vector_dim].
        - q3: Local projection on e3, shape [total_atoms, vector_dim].
        - active_capacity: Active slot capacity per atom, shape [total_atoms].
        - orbital_atom_index: Orbital-to-atom map, shape [total_orbitals, 2].
        - orbital_role: Orbital role ids, shape [total_orbitals].
        - active_slot_index: Active slot ids per orbital, shape [total_orbitals].

        Returns:
        - orbital_feature: [total_orbitals, orbital_feature_dim]
        - slot_alpha: [total_atoms, max_active_slots, 4]
        """

        local_vector_feature = jnp.concatenate([q1, q2, q3], axis=-1)
        candidates = self.slot_constructor(
            scalar_feature=scalar_feature,
            q1=q1,
            q2=q2,
            q3=q3,
        )
        matched = self.slot_matcher(
            scalar_feature=scalar_feature,
            candidates=candidates["candidates"],
            active_slot_capacity=active_capacity,
        )
        orbital_feature = self.mapper(
            scalar_feature=scalar_feature,
            local_vector_feature=local_vector_feature,
            slot_feature=matched["slot_feature"],
            orbital_atom_index=orbital_atom_index,
            orbital_role=orbital_role,
            active_slot_index=active_slot_index,
        )
        return orbital_feature, matched["alpha"]

    def countActiveSlotsPerAtom(
        self,
        num_atoms: int,
        orbital_atom_index: jnp.ndarray,
        orbital_role: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Count active slots per atom from static orbital metadata.

        Arguments:
        - num_atoms: Total number of atoms in current batch.
        - orbital_atom_index: Orbital-to-atom map, shape [total_orbitals, 2].
        - orbital_role: Orbital role ids, shape [total_orbitals].

        Returns:
        - Active slot capacity per atom, shape [total_atoms].
        """

        active_mask = (orbital_role == 2).astype(jnp.int32)
        active_owner = orbital_atom_index[:, 0]
        return jraph.segment_sum(active_mask, active_owner, num_segments=num_atoms)

    def __call__(
        self,
        scalar_feature: jnp.ndarray,
        vector_feature: jnp.ndarray,
        positions: jnp.ndarray,
        num_atoms_per_graph: jnp.ndarray,
        orbital_atom_index: jnp.ndarray,
        orbital_role: jnp.ndarray,
        active_slot_index: jnp.ndarray,
        local_frame_e1: jnp.ndarray | None = None,
        local_frame_e2: jnp.ndarray | None = None,
        local_frame_e3: jnp.ndarray | None = None,
    ) -> dict[str, jnp.ndarray]:
        """
        Run full atom-to-orbital projection forward.

        Arguments:
        - scalar_feature: Atom scalar channels, shape [total_atoms, scalar_dim].
        - vector_feature: Atom vector channels, shape [total_atoms, vector_dim, 3].
        - positions: Atom coordinates, shape [total_atoms, 3].
        - num_atoms_per_graph: Atom count tensor, shape [batch_size].
        - orbital_atom_index: Orbital-to-atom map, shape [total_orbitals, 2].
        - orbital_role: Orbital role ids, shape [total_orbitals].
        - active_slot_index: Active slot ids per orbital, shape [total_orbitals].
        - local_frame_e1: Optional cached local axis e1, shape [total_atoms, 3].
        - local_frame_e2: Optional cached local axis e2, shape [total_atoms, 3].
        - local_frame_e3: Optional cached local axis e3, shape [total_atoms, 3].

        Returns:
        - Dictionary:
          orbital_feature: [total_orbitals, orbital_feature_dim]
          orbital_feature_q3_flipped: [total_orbitals, orbital_feature_dim]
          slot_alpha: [total_atoms, max_active_slots, 4]
          active_capacity: [total_atoms]
        """

        if (
            (local_frame_e1 is not None)
            and (local_frame_e2 is not None)
            and (local_frame_e3 is not None)
        ):
            q1, q2, q3 = self.local_projector.projectVectorChannels(
                vector_feature=vector_feature,
                e1=local_frame_e1,
                e2=local_frame_e2,
                e3=local_frame_e3,
            )
            local_projection = {
                "e1": local_frame_e1,
                "e2": local_frame_e2,
                "e3": local_frame_e3,
                "q1": q1,
                "q2": q2,
                "q3": q3,
            }
        else:
            local_projection = self.local_projector(
                positions=positions,
                vector_feature=vector_feature,
                num_atoms_per_graph=num_atoms_per_graph,
            )
        active_capacity = self.countActiveSlotsPerAtom(
            num_atoms=scalar_feature.shape[0],
            orbital_atom_index=orbital_atom_index,
            orbital_role=orbital_role,
        )
        orbital_feature, slot_alpha = self.buildOrbitalFeatureFromProjections(
            scalar_feature=scalar_feature,
            q1=local_projection["q1"],
            q2=local_projection["q2"],
            q3=local_projection["q3"],
            active_capacity=active_capacity,
            orbital_atom_index=orbital_atom_index,
            orbital_role=orbital_role,
            active_slot_index=active_slot_index,
        )
        orbital_feature_q3_flipped, _ = self.buildOrbitalFeatureFromProjections(
            scalar_feature=scalar_feature,
            q1=local_projection["q1"],
            q2=local_projection["q2"],
            q3=-local_projection["q3"],
            active_capacity=active_capacity,
            orbital_atom_index=orbital_atom_index,
            orbital_role=orbital_role,
            active_slot_index=active_slot_index,
        )
        return {
            "orbital_feature": orbital_feature,
            "orbital_feature_q3_flipped": orbital_feature_q3_flipped,
            "slot_alpha": slot_alpha,
            "active_capacity": active_capacity,
        }
