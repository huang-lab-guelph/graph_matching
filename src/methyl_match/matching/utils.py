"""
Utility functions for graph matching.

This module provides helper functions for scoring assignments, computing
confidence scores, and validating matching results.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


def compute_chemical_shift_similarity(
    exp_shifts: np.ndarray,
    struct_shifts: np.ndarray,
    h_tolerance: float = 0.5,
    c_tolerance: float = 1.0,
) -> np.ndarray:
    """
    Compute chemical shift similarity matrix.

    Args:
        exp_shifts: Experimental shifts (n_exp, 2) with [1H, 13C] shifts.
        struct_shifts: Structural/predicted shifts (n_struct, 2).
        h_tolerance: 1H chemical shift tolerance (ppm).
        c_tolerance: 13C chemical shift tolerance (ppm).

    Returns:
        Similarity matrix (n_exp, n_struct) with values in [0, 1].
        1.0 means perfect match, 0.0 means shifts differ by >tolerance.
    """
    n_exp, n_struct = exp_shifts.shape[0], struct_shifts.shape[0]
    similarity = np.zeros((n_exp, n_struct))

    for i in range(n_exp):
        for j in range(n_struct):
            # 1H shift difference
            h_diff = abs(exp_shifts[i, 0] - struct_shifts[j, 0])
            h_sim = max(0.0, 1.0 - h_diff / h_tolerance)

            # 13C shift difference
            c_diff = abs(exp_shifts[i, 1] - struct_shifts[j, 1])
            c_sim = max(0.0, 1.0 - c_diff / c_tolerance)

            # Combined similarity (geometric mean to penalize mismatches)
            similarity[i, j] = np.sqrt(h_sim * c_sim)

    return similarity


def compute_residue_type_similarity(
    exp_types: List[Optional[str]],
    struct_types: List[str],
) -> np.ndarray:
    """
    Compute residue type similarity matrix.

    Args:
        exp_types: List of experimental residue types (may be None if unknown).
        struct_types: List of structural residue types.

    Returns:
        Similarity matrix (n_exp, n_struct) with values in {0, 1}.
        1.0 if types match, 0.0 if they don't.
    """
    n_exp = len(exp_types)
    n_struct = len(struct_types)
    similarity = np.zeros((n_exp, n_struct))

    for i in range(n_exp):
        for j in range(n_struct):
            if exp_types[i] is None:
                # Unknown experimental type - neutral similarity
                similarity[i, j] = 0.5
            elif exp_types[i] == struct_types[j]:
                similarity[i, j] = 1.0
            else:
                similarity[i, j] = 0.0

    return similarity


def compute_topology_consistency(
    assignments: Dict[int, int],
    graph_experimental: nx.Graph,
    graph_structural: nx.Graph,
) -> float:
    """
    Compute topology consistency score for assignments.

    Measures how well the experimental NOE connectivity matches the structural
    distance connectivity under the given assignments.

    Args:
        assignments: Dictionary mapping exp_id -> struct_id.
        graph_experimental: Experimental graph with NOE edges.
        graph_structural: Structural graph with distance edges.

    Returns:
        Topology consistency score in [0, 1]. Higher means better agreement
        between experimental and structural connectivity.
    """
    if len(assignments) < 2:
        return 1.0  # Not enough assignments to compute topology

    # Count consistent and inconsistent edges
    consistent_edges = 0
    total_edges = 0

    exp_ids = list(assignments.keys())
    for i in range(len(exp_ids)):
        for j in range(i + 1, len(exp_ids)):
            exp_i, exp_j = exp_ids[i], exp_ids[j]
            struct_i = assignments[exp_i]
            struct_j = assignments[exp_j]

            # Check if edge exists in experimental graph
            has_exp_edge = graph_experimental.has_edge(exp_i, exp_j)

            # Check if edge exists in structural graph
            has_struct_edge = graph_structural.has_edge(struct_i, struct_j)

            total_edges += 1

            # Consistent if both have edge or both don't have edge
            if has_exp_edge == has_struct_edge:
                consistent_edges += 1

    if total_edges == 0:
        return 1.0

    return consistent_edges / total_edges


def compute_distance_consistency(
    assignments: Dict[int, int],
    graph_experimental: nx.Graph,
    graph_structural: nx.Graph,
    distance_tolerance: float = 2.0,
) -> float:
    """
    Compute distance consistency score for assignments.

    For edges that exist in both graphs under the assignments, check if the
    NOE intensity correlates with structural distance (closer = stronger NOE).

    Args:
        assignments: Dictionary mapping exp_id -> struct_id.
        graph_experimental: Experimental graph with NOE intensities.
        graph_structural: Structural graph with spatial distances.
        distance_tolerance: Tolerance for distance agreement (Angstroms).

    Returns:
        Distance consistency score in [0, 1]. Higher means better correlation
        between NOE intensities and structural distances.
    """
    if len(assignments) < 2:
        return 1.0

    consistent_count = 0
    total_count = 0

    exp_ids = list(assignments.keys())
    for i in range(len(exp_ids)):
        for j in range(i + 1, len(exp_ids)):
            exp_i, exp_j = exp_ids[i], exp_ids[j]
            struct_i = assignments[exp_i]
            struct_j = assignments[exp_j]

            # Only consider edges that exist in both graphs
            if not graph_experimental.has_edge(exp_i, exp_j):
                continue
            if not graph_structural.has_edge(struct_i, struct_j):
                continue

            # Get NOE intensity and structural distance
            exp_edge_data = graph_experimental.get_edge_data(exp_i, exp_j)
            struct_edge_data = graph_structural.get_edge_data(struct_i, struct_j)

            noe_intensity = exp_edge_data.get('weight', 1.0)
            struct_distance = struct_edge_data.get('weight', 0.0)

            # Check consistency: strong NOE should mean short distance
            # Assume NOE intensity ~ 1/distance^6 (approximately)
            # For simplicity, just check if strong NOE pairs with short distance
            total_count += 1

            # High intensity (>0.7) should pair with short distance (<8 Angstroms)
            # Low intensity (<0.3) should pair with long distance (>10 Angstroms)
            if noe_intensity > 0.7 and struct_distance < 8.0:
                consistent_count += 1
            elif noe_intensity < 0.3 and struct_distance > 10.0:
                consistent_count += 1
            elif 0.3 <= noe_intensity <= 0.7:
                # Medium intensity - neutral
                consistent_count += 0.5

    if total_count == 0:
        return 1.0

    return consistent_count / total_count


def compute_confidence_scores(
    assignments: Dict[int, int],
    cost_matrix: np.ndarray,
    graph_experimental: nx.Graph,
    graph_structural: nx.Graph,
    method: str = "combined",
) -> Dict[int, float]:
    """
    Compute confidence scores for assignments.

    Args:
        assignments: Dictionary mapping exp_id -> struct_id.
        cost_matrix: Cost matrix used for matching (n_exp, n_struct).
        graph_experimental: Experimental graph.
        graph_structural: Structural graph.
        method: Method for computing confidence:
                - "cost": Based on assignment cost (lower cost = higher confidence)
                - "gap": Based on cost gap to next-best assignment
                - "topology": Based on topology consistency
                - "combined": Weighted combination of all methods

    Returns:
        Dictionary mapping exp_id -> confidence_score (0.0-1.0).
    """
    confidence = {}

    exp_ids = sorted(assignments.keys())

    for exp_idx, exp_id in enumerate(exp_ids):
        struct_id = assignments[exp_id]

        # Find struct_idx from node list
        struct_nodes = sorted(graph_structural.nodes())
        struct_idx = struct_nodes.index(struct_id)

        # Cost-based confidence
        assigned_cost = cost_matrix[exp_idx, struct_idx]
        min_cost = np.min(cost_matrix[exp_idx, :])
        max_cost = np.max(cost_matrix[exp_idx, :])

        if max_cost > min_cost:
            cost_conf = 1.0 - (assigned_cost - min_cost) / (max_cost - min_cost)
        else:
            cost_conf = 1.0

        # Gap-based confidence (difference to second-best)
        costs = cost_matrix[exp_idx, :].copy()
        costs[struct_idx] = np.inf  # Exclude current assignment
        second_best_cost = np.min(costs)

        if second_best_cost > assigned_cost and assigned_cost > 0:
            gap_conf = min(1.0, (second_best_cost - assigned_cost) / assigned_cost)
        elif assigned_cost == 0 and second_best_cost > 0:
            gap_conf = 1.0  # Perfect assignment
        else:
            gap_conf = 0.0

        # Topology-based confidence (local neighborhood consistency)
        # Check how many neighbors of exp_id are consistently assigned
        neighbors_exp = list(graph_experimental.neighbors(exp_id))
        if neighbors_exp:
            consistent_neighbors = 0
            for neighbor_exp in neighbors_exp:
                if neighbor_exp in assignments:
                    neighbor_struct = assignments[neighbor_exp]
                    # Check if structural counterparts are also neighbors
                    if graph_structural.has_edge(struct_id, neighbor_struct):
                        consistent_neighbors += 1
            topo_conf = consistent_neighbors / len(neighbors_exp)
        else:
            topo_conf = 0.5  # Neutral if no neighbors

        # Combined confidence
        if method == "cost":
            confidence[exp_id] = cost_conf
        elif method == "gap":
            confidence[exp_id] = gap_conf
        elif method == "topology":
            confidence[exp_id] = topo_conf
        elif method == "combined":
            # Weighted combination
            confidence[exp_id] = 0.4 * cost_conf + 0.3 * gap_conf + 0.3 * topo_conf
        else:
            raise ValueError(f"Unknown confidence method: {method}")

    return confidence


def validate_assignment(
    assignments: Dict[int, int],
    graph_experimental: nx.Graph,
    graph_structural: nx.Graph,
) -> Dict[str, float]:
    """
    Validate matching result and compute quality metrics.

    Args:
        assignments: Dictionary mapping exp_id -> struct_id.
        graph_experimental: Experimental graph.
        graph_structural: Structural graph.

    Returns:
        Dictionary of validation metrics:
        - topology_consistency: Fraction of edges consistent between graphs
        - distance_consistency: Correlation between NOE and distance
        - assignment_rate: Fraction of experimental nodes assigned
        - uniqueness: Fraction of unique assignments (should be 1.0)
    """
    metrics = {}

    # Topology consistency
    metrics['topology_consistency'] = compute_topology_consistency(
        assignments, graph_experimental, graph_structural
    )

    # Distance consistency
    metrics['distance_consistency'] = compute_distance_consistency(
        assignments, graph_experimental, graph_structural
    )

    # Assignment rate
    n_exp = graph_experimental.number_of_nodes()
    metrics['assignment_rate'] = len(assignments) / n_exp if n_exp > 0 else 0.0

    # Uniqueness (all assignments should be unique)
    unique_assignments = len(set(assignments.values()))
    metrics['uniqueness'] = unique_assignments / len(assignments) if assignments else 1.0

    return metrics


def create_cost_matrix_from_similarity(
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    """
    Convert similarity matrix to cost matrix.

    Args:
        similarity_matrix: Similarity matrix (n_exp, n_struct) with values in [0, 1].

    Returns:
        Cost matrix with same shape. Higher cost means less similar.
    """
    return 1.0 - similarity_matrix


def create_similarity_matrix_from_cost(
    cost_matrix: np.ndarray,
) -> np.ndarray:
    """
    Convert cost matrix to similarity matrix.

    Args:
        cost_matrix: Cost matrix (n_exp, n_struct).

    Returns:
        Similarity matrix with same shape. Higher similarity means lower cost.
    """
    # Normalize cost matrix to [0, 1]
    min_cost = np.min(cost_matrix)
    max_cost = np.max(cost_matrix)

    if max_cost > min_cost:
        return 1.0 - (cost_matrix - min_cost) / (max_cost - min_cost)
    else:
        return np.ones_like(cost_matrix)
