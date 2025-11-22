"""
Hungarian algorithm for optimal graph matching.

This module implements the Hungarian algorithm (also known as the Kuhn-Munkres
algorithm) for solving the Linear Assignment Problem (LAP). It finds the optimal
one-to-one assignment that minimizes total cost.
"""

import time
from typing import Optional
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

from .base import GraphMatcher, MatchingResult
from .utils import (
    compute_confidence_scores,
    create_cost_matrix_from_similarity,
)


class HungarianMatcher(GraphMatcher):
    """
    Hungarian algorithm for optimal graph matching.

    This algorithm solves the Linear Assignment Problem (LAP) optimally using
    the Hungarian algorithm. It finds the one-to-one assignment that minimizes
    the total cost across all assignments.

    The algorithm runs in O(n³) time and always finds the globally optimal
    solution. However, it only considers node-to-node costs and does not
    directly incorporate graph topology (edge structure).

    For problems where topology is important, consider using QAPMatcher instead.

    Parameters:
    ----------
    confidence_threshold : float, default=0.0
        Minimum confidence score to accept an assignment. Assignments with
        lower confidence are marked as unassigned.
    maximize : bool, default=False
        If True, maximize similarity instead of minimizing cost.
    handle_rectangular : str, default="pad"
        How to handle non-square matrices (different number of exp/struct nodes):
        - "pad": Pad with high-cost dummy nodes
        - "truncate": Only assign min(n_exp, n_struct) nodes

    Attributes:
    ----------
    name : str
        Algorithm name ("HungarianMatcher").
    """

    def __init__(
        self,
        confidence_threshold: float = 0.0,
        maximize: bool = False,
        handle_rectangular: str = "pad",
        **config,
    ):
        """
        Initialize HungarianMatcher.

        Args:
            confidence_threshold: Minimum confidence to accept assignment (0.0-1.0).
            maximize: If True, maximize similarity instead of minimize cost.
            handle_rectangular: How to handle non-square matrices ("pad" or "truncate").
            **config: Additional configuration parameters.
        """
        super().__init__(
            name="HungarianMatcher",
            confidence_threshold=confidence_threshold,
            maximize=maximize,
            handle_rectangular=handle_rectangular,
            **config,
        )
        self.confidence_threshold = confidence_threshold
        self.maximize = maximize
        self.handle_rectangular = handle_rectangular

    def match(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> MatchingResult:
        """
        Perform optimal graph matching using Hungarian algorithm.

        Args:
            graph_experimental: Experimental NMR peak graph.
            graph_structural: Structural methyl graph.

        Returns:
            MatchingResult with optimal assignments and confidence scores.
        """
        start_time = time.time()

        # Validate inputs
        n_exp, n_struct = self._validate_graphs(
            graph_experimental, graph_structural
        )

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(
            graph_experimental, graph_structural
        )

        # Convert to cost matrix (lower is better)
        cost_matrix = create_cost_matrix_from_similarity(similarity_matrix)

        # Handle non-square matrices
        cost_matrix_padded, pad_info = self._handle_rectangular_matrix(
            cost_matrix, n_exp, n_struct
        )

        # Solve using Hungarian algorithm
        if self.maximize:
            # Convert to minimization problem
            max_val = np.max(cost_matrix_padded)
            cost_matrix_for_solver = max_val - cost_matrix_padded
        else:
            cost_matrix_for_solver = cost_matrix_padded

        row_ind, col_ind = linear_sum_assignment(cost_matrix_for_solver)

        # Extract assignments (removing dummy nodes if padded)
        assignments = self._extract_assignments(
            row_ind, col_ind, graph_experimental, graph_structural, pad_info
        )

        # Compute confidence scores
        confidence_scores = compute_confidence_scores(
            assignments,
            cost_matrix,
            graph_experimental,
            graph_structural,
            method="combined",
        )

        # Compute total cost
        total_cost = sum(
            cost_matrix[
                sorted(graph_experimental.nodes()).index(exp_id),
                sorted(graph_structural.nodes()).index(struct_id),
            ]
            for exp_id, struct_id in assignments.items()
        )

        # Filter by confidence threshold
        filtered_assignments = {
            exp_id: struct_id
            for exp_id, struct_id in assignments.items()
            if confidence_scores[exp_id] >= self.confidence_threshold
        }
        filtered_confidence = {
            exp_id: conf
            for exp_id, conf in confidence_scores.items()
            if conf >= self.confidence_threshold
        }
        unassigned = [
            exp_id
            for exp_id in assignments.keys()
            if confidence_scores[exp_id] < self.confidence_threshold
        ]

        # Metadata
        elapsed_time = time.time() - start_time
        metadata = {
            "algorithm": "hungarian",
            "runtime_seconds": elapsed_time,
            "n_experimental": n_exp,
            "n_structural": n_struct,
            "total_cost": total_cost,
            "mean_cost": total_cost / len(assignments) if assignments else 0.0,
            "confidence_threshold": self.confidence_threshold,
            "maximize": self.maximize,
            "padded": pad_info["padded"],
        }

        return MatchingResult(
            assignments=filtered_assignments,
            confidence_scores=filtered_confidence,
            unassigned_peaks=unassigned,
            metadata=metadata,
            cost_matrix=cost_matrix,
            similarity_matrix=similarity_matrix,
        )

    def _build_similarity_matrix(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> np.ndarray:
        """
        Build node-to-node similarity matrix.

        Args:
            graph_experimental: Experimental graph.
            graph_structural: Structural graph.

        Returns:
            Similarity matrix (n_exp, n_struct) with values in [0, 1].
        """
        exp_nodes = sorted(graph_experimental.nodes())
        struct_nodes = sorted(graph_structural.nodes())

        n_exp = len(exp_nodes)
        n_struct = len(struct_nodes)

        similarity = np.zeros((n_exp, n_struct))

        for i, exp_id in enumerate(exp_nodes):
            exp_data = graph_experimental.nodes[exp_id]

            for j, struct_id in enumerate(struct_nodes):
                struct_data = graph_structural.nodes[struct_id]

                # Residue type similarity (primary feature)
                residue_sim = 0.0
                if 'residue_type' in exp_data and 'residue_type' in struct_data:
                    if exp_data['residue_type'] == struct_data['residue_type']:
                        residue_sim = 1.0
                    else:
                        residue_sim = 0.0
                else:
                    residue_sim = 0.5  # Neutral if unknown

                # Node degree similarity (secondary feature)
                exp_degree = graph_experimental.degree(exp_id)
                struct_degree = graph_structural.degree(struct_id)
                max_degree = max(exp_degree, struct_degree)
                if max_degree > 0:
                    degree_sim = 1.0 - abs(exp_degree - struct_degree) / max_degree
                else:
                    degree_sim = 1.0

                # Combined similarity (weighted)
                similarity[i, j] = 0.8 * residue_sim + 0.2 * degree_sim

        return similarity

    def _handle_rectangular_matrix(
        self,
        cost_matrix: np.ndarray,
        n_exp: int,
        n_struct: int,
    ) -> tuple[np.ndarray, dict]:
        """
        Handle rectangular cost matrices.

        Args:
            cost_matrix: Original cost matrix (n_exp, n_struct).
            n_exp: Number of experimental nodes.
            n_struct: Number of structural nodes.

        Returns:
            Tuple of (padded_matrix, padding_info_dict).
        """
        pad_info = {"padded": False, "pad_rows": 0, "pad_cols": 0}

        if n_exp == n_struct:
            return cost_matrix, pad_info

        if self.handle_rectangular == "pad":
            # Pad with high-cost dummy nodes
            max_size = max(n_exp, n_struct)
            padded = np.full((max_size, max_size), fill_value=np.max(cost_matrix) * 2)
            padded[:n_exp, :n_struct] = cost_matrix

            pad_info["padded"] = True
            pad_info["pad_rows"] = max_size - n_exp
            pad_info["pad_cols"] = max_size - n_struct

            return padded, pad_info

        elif self.handle_rectangular == "truncate":
            # Matrix is already rectangular, scipy handles it
            return cost_matrix, pad_info

        else:
            raise ValueError(
                f"Unknown handle_rectangular method: {self.handle_rectangular}"
            )

    def _extract_assignments(
        self,
        row_ind: np.ndarray,
        col_ind: np.ndarray,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
        pad_info: dict,
    ) -> dict[int, int]:
        """
        Extract assignments from Hungarian algorithm output.

        Args:
            row_ind: Row indices from linear_sum_assignment.
            col_ind: Column indices from linear_sum_assignment.
            graph_experimental: Experimental graph.
            graph_structural: Structural graph.
            pad_info: Padding information.

        Returns:
            Dictionary mapping exp_id -> struct_id.
        """
        exp_nodes = sorted(graph_experimental.nodes())
        struct_nodes = sorted(graph_structural.nodes())

        n_exp = len(exp_nodes)
        n_struct = len(struct_nodes)

        assignments = {}

        for row, col in zip(row_ind, col_ind):
            # Skip dummy assignments
            if row >= n_exp or col >= n_struct:
                continue

            exp_id = exp_nodes[row]
            struct_id = struct_nodes[col]

            assignments[exp_id] = struct_id

        return assignments

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HungarianMatcher("
            f"confidence_threshold={self.confidence_threshold}, "
            f"maximize={self.maximize})"
        )
