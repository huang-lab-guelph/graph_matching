"""
Greedy graph matching algorithm.

This module implements a simple greedy matching algorithm that iteratively
selects the most similar node pair until all nodes are assigned.
"""

import time
from typing import Dict, List, Optional, Set
import numpy as np
import networkx as nx

from .base import GraphMatcher, MatchingResult
from .utils import (
    compute_confidence_scores,
    create_cost_matrix_from_similarity,
)


class GreedyMatcher(GraphMatcher):
    """
    Greedy graph matching algorithm.

    This algorithm iteratively selects the node pair with highest similarity
    (or lowest cost) and assigns them, then removes them from consideration.
    This continues until all nodes are assigned or a confidence threshold is met.

    The algorithm is fast (O(n²) to O(n³)) but does not guarantee optimal
    solutions. It works well when the similarity matrix has clear peaks.

    Parameters:
    ----------
    confidence_threshold : float, default=0.0
        Minimum confidence score to accept an assignment. Assignments with
        lower confidence are marked as unassigned.
    use_topology : bool, default=False
        If True, incorporate neighborhood consistency when computing scores.
    max_iterations : Optional[int], default=None
        Maximum number of assignments to make. If None, assigns all nodes.

    Attributes:
    ----------
    name : str
        Algorithm name ("GreedyMatcher").
    """

    def __init__(
        self,
        confidence_threshold: float = 0.0,
        use_topology: bool = False,
        max_iterations: Optional[int] = None,
        **config,
    ):
        """
        Initialize GreedyMatcher.

        Args:
            confidence_threshold: Minimum confidence to accept assignment (0.0-1.0).
            use_topology: Whether to use topology information.
            max_iterations: Maximum number of assignments (None = all nodes).
            **config: Additional configuration parameters.
        """
        super().__init__(
            name="GreedyMatcher",
            confidence_threshold=confidence_threshold,
            use_topology=use_topology,
            max_iterations=max_iterations,
            **config,
        )
        self.confidence_threshold = confidence_threshold
        self.use_topology = use_topology
        self.max_iterations = max_iterations

    def match(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> MatchingResult:
        """
        Perform greedy graph matching.

        Args:
            graph_experimental: Experimental NMR peak graph.
            graph_structural: Structural methyl graph.

        Returns:
            MatchingResult with assignments and confidence scores.
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

        # Perform greedy assignment
        assignments, iteration_log = self._greedy_assign(
            cost_matrix,
            graph_experimental,
            graph_structural,
        )

        # Compute confidence scores
        confidence_scores = compute_confidence_scores(
            assignments,
            cost_matrix,
            graph_experimental,
            graph_structural,
            method="combined",
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
            "algorithm": "greedy",
            "runtime_seconds": elapsed_time,
            "iterations": len(iteration_log),
            "n_experimental": n_exp,
            "n_structural": n_struct,
            "confidence_threshold": self.confidence_threshold,
            "use_topology": self.use_topology,
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

                # Residue type similarity
                residue_sim = 0.0
                if 'residue_type' in exp_data and 'residue_type' in struct_data:
                    if exp_data['residue_type'] == struct_data['residue_type']:
                        residue_sim = 1.0
                    else:
                        residue_sim = 0.0
                else:
                    residue_sim = 0.5  # Neutral if unknown

                # Node degree similarity (graph topology)
                exp_degree = graph_experimental.degree(exp_id)
                struct_degree = graph_structural.degree(struct_id)
                max_degree = max(exp_degree, struct_degree)
                if max_degree > 0:
                    degree_sim = 1.0 - abs(exp_degree - struct_degree) / max_degree
                else:
                    degree_sim = 1.0

                # Combined similarity
                if self.use_topology:
                    similarity[i, j] = 0.6 * residue_sim + 0.4 * degree_sim
                else:
                    similarity[i, j] = residue_sim

        return similarity

    def _greedy_assign(
        self,
        cost_matrix: np.ndarray,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> tuple[Dict[int, int], List[Dict]]:
        """
        Perform greedy assignment.

        Args:
            cost_matrix: Cost matrix (n_exp, n_struct).
            graph_experimental: Experimental graph.
            graph_structural: Structural graph.

        Returns:
            Tuple of (assignments dict, iteration log list).
        """
        exp_nodes = sorted(graph_experimental.nodes())
        struct_nodes = sorted(graph_structural.nodes())

        n_exp = len(exp_nodes)
        n_struct = len(struct_nodes)

        # Make a copy to modify
        cost_matrix = cost_matrix.copy()

        assignments = {}
        assigned_exp: Set[int] = set()
        assigned_struct: Set[int] = set()
        iteration_log = []

        max_iter = self.max_iterations if self.max_iterations else min(n_exp, n_struct)

        for iteration in range(max_iter):
            # Find minimum cost unassigned pair
            min_cost = np.inf
            best_exp_idx = -1
            best_struct_idx = -1

            for i in range(n_exp):
                if i in assigned_exp:
                    continue
                for j in range(n_struct):
                    if j in assigned_struct:
                        continue
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        best_exp_idx = i
                        best_struct_idx = j

            # Check if we found a valid assignment
            if best_exp_idx == -1 or best_struct_idx == -1:
                break

            # Make assignment
            exp_id = exp_nodes[best_exp_idx]
            struct_id = struct_nodes[best_struct_idx]

            assignments[exp_id] = struct_id
            assigned_exp.add(best_exp_idx)
            assigned_struct.add(best_struct_idx)

            iteration_log.append({
                "iteration": iteration,
                "exp_id": exp_id,
                "struct_id": struct_id,
                "cost": min_cost,
            })

        return assignments, iteration_log

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GreedyMatcher("
            f"confidence_threshold={self.confidence_threshold}, "
            f"use_topology={self.use_topology})"
        )
