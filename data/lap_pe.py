"""Laplacian positional encoding utilities for atom graphs."""

from __future__ import annotations

import numpy as np


def denseAdjacencyFromEdges(
    num_nodes: int,
    senders: np.ndarray,
    receivers: np.ndarray,
) -> np.ndarray:
    """
    Build one symmetric dense adjacency matrix from graph edges.

    Arguments:
    - num_nodes: Number of graph nodes.
    - senders: Edge sender indices, shape [num_edges].
    - receivers: Edge receiver indices, shape [num_edges].

    Returns:
    - Dense symmetric adjacency matrix, shape [num_nodes, num_nodes].
    """

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if num_nodes == 0:
        return adjacency
    if senders.size == 0:
        return adjacency
    adjacency[np.asarray(senders, dtype=np.int32), np.asarray(receivers, dtype=np.int32)] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def lapPeFromAdjacency(
    adjacency: np.ndarray,
    num_nodes: int,
    k: int,
    eps: float = 1.0e-12,
    add_self_loops: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized-Laplacian eigenpairs following the old PyTorch semantics.

    Arguments:
    - adjacency: Dense adjacency matrix, shape [num_nodes, num_nodes].
    - num_nodes: Number of graph nodes.
    - k: Number of non-trivial eigenvectors.
    - eps: Numerical stability epsilon.
    - add_self_loops: Whether to add self-loops before building the Laplacian.

    Returns:
    - evals_k: Non-trivial eigenvalues padded to shape [k].
    - evecs_k: Matching eigenvectors padded to shape [num_nodes, k].
    """

    if k <= 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((num_nodes, 0), dtype=np.float32),
        )

    adjacency = np.asarray(adjacency, dtype=np.float32).copy()
    if add_self_loops:
        np.fill_diagonal(adjacency, 1.0)

    degree = np.sum(adjacency, axis=1)
    degree_inv_sqrt = np.power(degree + float(eps), -0.5, dtype=np.float32)
    d_inv_sqrt = np.diag(degree_inv_sqrt.astype(np.float32))
    identity = np.eye(num_nodes, dtype=np.float32)
    laplacian = identity - d_inv_sqrt @ adjacency @ d_inv_sqrt

    evals, evecs = np.linalg.eigh(laplacian.astype(np.float64))
    nontrivial = np.where(evals > float(eps))[0]
    if nontrivial.size == 0:
        return (
            np.zeros((k,), dtype=np.float32),
            np.zeros((num_nodes, k), dtype=np.float32),
        )

    take = nontrivial[:k]
    evals_k = evals[take].astype(np.float32, copy=False)
    evecs_k = evecs[:, take].astype(np.float32, copy=False)

    if evecs_k.shape[1] < k:
        pad = k - evecs_k.shape[1]
        evecs_k = np.concatenate(
            [evecs_k, np.zeros((num_nodes, pad), dtype=np.float32)],
            axis=1,
        )
        evals_k = np.concatenate(
            [evals_k, np.zeros((pad,), dtype=np.float32)],
            axis=0,
        )

    return evals_k, evecs_k


def lapPeFromEdges(
    num_nodes: int,
    senders: np.ndarray,
    receivers: np.ndarray,
    k: int,
    eps: float = 1.0e-12,
    add_self_loops: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Laplacian PE directly from graph edges.

    Arguments:
    - num_nodes: Number of graph nodes.
    - senders: Edge sender indices.
    - receivers: Edge receiver indices.
    - k: Number of non-trivial eigenvectors.
    - eps: Numerical stability epsilon.
    - add_self_loops: Whether to add self-loops.

    Returns:
    - evals_k: Shape [k].
    - evecs_k: Shape [num_nodes, k].
    """

    adjacency = denseAdjacencyFromEdges(num_nodes=num_nodes, senders=senders, receivers=receivers)
    return lapPeFromAdjacency(
        adjacency=adjacency,
        num_nodes=num_nodes,
        k=k,
        eps=eps,
        add_self_loops=add_self_loops,
    )


def repeatLapEvalsPerNode(num_nodes: int, lap_evals: np.ndarray) -> np.ndarray:
    """
    Repeat one graph-level Laplacian eigenvalue vector to all nodes in the graph.

    Arguments:
    - num_nodes: Number of graph nodes.
    - lap_evals: Eigenvalue vector, shape [k].

    Returns:
    - Repeated eigenvalue tensor, shape [num_nodes, k].
    """

    lap_evals = np.asarray(lap_evals, dtype=np.float32)
    if num_nodes == 0:
        return np.zeros((0, int(lap_evals.shape[0])), dtype=np.float32)
    return np.repeat(lap_evals[None, :], repeats=num_nodes, axis=0).astype(np.float32, copy=False)
