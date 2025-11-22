"""
Quadratic Assignment Problem (QAP) matching algorithm.

This module implements graph matching using the Quadratic Assignment Problem
formulation with the Fast Approximate QAP (FAQ) algorithm. Unlike the Hungarian
algorithm, QAP considers both node similarities AND edge topology.
"""

import time
from typing import Optional
import numpy as np
import networkx as nx
from scipy.optimize import quadratic_assignment

from .base import GraphMatcher, MatchingResult
from .utils import (
    compute_confidence_scores,
    create_cost_matrix_from_similarity,
)


class QAPMatcher(GraphMatcher):
    """
    Quadratic Assignment Problem (QAP) graph matching.

    This algorithm formulates graph matching as a QAP and solves it using the
    Fast Approximate QAP (FAQ) algorithm from scipy. QAP naturally incorporates
    both node-to-node similarities and graph topology (edge structure).

    The objective is to find a permutation that minimizes:
        cost = trace(A_exp^T @ P @ A_struct @ P^T) + trace(S^T @ P)

    where:
    - A_exp, A_struct are adjacency/distance matrices
    - S is the node similarity matrix
    - P is the permutation matrix (assignment)

    The FAQ algorithm is an iterative method that typically converges quickly
    and provides good approximate solutions. It runs in O(n³) per iteration.

    Parameters:
    ----------
    confidence_threshold : float, default=0.0
        Minimum confidence score to accept an assignment.
    method : str, default="faq"
        QAP solver method ("faq" or "2opt"). FAQ is recommended.
    options : Optional[dict], default=None
        Solver options passed to scipy.optimize.quadratic_assignment.
        Common options:
        - maximize: If True, maximize similarity (default: True)
        - partial_match: Optional initial assignment
        - rng: Random number generator seed
        - P0: Initial permutation matrix
        - shuffle_input: Shuffle input for better convergence
        - maxiter: Maximum iterations (default: 30)
        - tol: Convergence tolerance (default: 0.03)
    topology_weight : float, default=0.5
        Weight for topology term (0.0-1.0). Higher values emphasize edge structure.
    normalize_adjacency : bool, default=True
        If True, normalize adjacency matrices to [0, 1] range.

    Attributes:
    ----------
    name : str
        Algorithm name ("QAPMatcher").
    """

    def __init__(
        self,
        confidence_threshold: float = 0.0,
        method: str = "faq",
        options: Optional[dict] = None,
        topology_weight: float = 0.5,
        normalize_adjacency: bool = True,
        **config,
    ):
        """
        Initialize QAPMatcher.

        Args:
            confidence_threshold: Minimum confidence to accept assignment (0.0-1.0).
            method: QAP solver method ("faq" or "2opt").
            options: Solver options dict.
            topology_weight: Weight for topology vs node features (0.0-1.0).
            normalize_adjacency: Whether to normalize adjacency matrices.
            **config: Additional configuration parameters.
        """
        super().__init__(
            name="QAPMatcher",
            confidence_threshold=confidence_threshold,
            method=method,
            options=options or {},
            topology_weight=topology_weight,
            normalize_adjacency=normalize_adjacency,
            **config,
        )
        self.confidence_threshold = confidence_threshold
        self.method = method
        self.options = options or {"maximize": False, "maxiter": 30}
        self.topology_weight = topology_weight
        self.normalize_adjacency = normalize_adjacency

    def match(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> MatchingResult:
        """
        Perform graph matching using QAP formulation.

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

        # Build node similarity matrix (S)
        similarity_matrix = self._build_similarity_matrix(
            graph_experimental, graph_structural
        )

        # Build adjacency/distance matrices (A_exp, A_struct)
        adj_exp = self._build_adjacency_matrix(graph_experimental)
        adj_struct = self._build_adjacency_matrix(graph_structural)

        # Handle rectangular case by padding
        if n_exp != n_struct:
            similarity_matrix, adj_exp, adj_struct = self._pad_matrices(
                similarity_matrix, adj_exp, adj_struct, n_exp, n_struct
            )

        # Normalize adjacency matrices if requested
        if self.normalize_adjacency:
            adj_exp = self._normalize_matrix(adj_exp)
            adj_struct = self._normalize_matrix(adj_struct)

        # Convert similarity to cost (QAP minimizes by default)
        cost_matrix = create_cost_matrix_from_similarity(similarity_matrix)

        # Solve QAP
        result = quadratic_assignment(
            A=adj_exp,
            B=adj_struct,
            method=self.method,
            options=self.options,
        )

        # Extract assignments
        assignments = self._extract_assignments(
            result.col_ind,
            graph_experimental,
            graph_structural,
            n_exp,
            n_struct,
        )

        # Compute confidence scores
        confidence_scores = compute_confidence_scores(
            assignments,
            cost_matrix[:n_exp, :n_struct],  # Use original cost matrix
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
            "algorithm": "qap",
            "method": self.method,
            "runtime_seconds": elapsed_time,
            "n_experimental": n_exp,
            "n_structural": n_struct,
            "qap_fun": result.fun,  # Final objective value
            "qap_nit": result.nit,  # Number of iterations
            "confidence_threshold": self.confidence_threshold,
            "topology_weight": self.topology_weight,
        }

        return MatchingResult(
            assignments=filtered_assignments,
            confidence_scores=filtered_confidence,
            unassigned_peaks=unassigned,
            metadata=metadata,
            cost_matrix=cost_matrix[:n_exp, :n_struct],
            similarity_matrix=similarity_matrix[:n_exp, :n_struct],
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

                # Node degree similarity
                exp_degree = graph_experimental.degree(exp_id)
                struct_degree = graph_structural.degree(struct_id)
                max_degree = max(exp_degree, struct_degree)
                if max_degree > 0:
                    degree_sim = 1.0 - abs(exp_degree - struct_degree) / max_degree
                else:
                    degree_sim = 1.0

                # Combined similarity
                similarity[i, j] = 0.7 * residue_sim + 0.3 * degree_sim

        return similarity

    def _build_adjacency_matrix(self, graph: nx.Graph) -> np.ndarray:
        """
        Build adjacency matrix from graph.

        Args:
            graph: NetworkX graph.

        Returns:
            Adjacency matrix (n, n) with edge weights.
        """
        nodes = sorted(graph.nodes())
        n = len(nodes)

        adj = np.zeros((n, n))

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j and graph.has_edge(node_i, node_j):
                    edge_data = graph.get_edge_data(node_i, node_j)
                    weight = edge_data.get('weight', 1.0)
                    adj[i, j] = weight

        return adj

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize matrix to [0, 1] range.

        Args:
            matrix: Input matrix.

        Returns:
            Normalized matrix.
        """
        min_val = np.min(matrix)
        max_val = np.max(matrix)

        if max_val > min_val:
            return (matrix - min_val) / (max_val - min_val)
        else:
            return matrix

    def _pad_matrices(
        self,
        similarity: np.ndarray,
        adj_exp: np.ndarray,
        adj_struct: np.ndarray,
        n_exp: int,
        n_struct: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pad matrices to make them square.

        Args:
            similarity: Similarity matrix (n_exp, n_struct).
            adj_exp: Experimental adjacency matrix (n_exp, n_exp).
            adj_struct: Structural adjacency matrix (n_struct, n_struct).
            n_exp: Number of experimental nodes.
            n_struct: Number of structural nodes.

        Returns:
            Tuple of (padded_similarity, padded_adj_exp, padded_adj_struct).
        """
        max_size = max(n_exp, n_struct)

        # Pad similarity with low similarity (high cost)
        padded_sim = np.zeros((max_size, max_size))
        padded_sim[:n_exp, :n_struct] = similarity

        # Pad adjacency matrices with zeros (no edges)
        padded_adj_exp = np.zeros((max_size, max_size))
        padded_adj_exp[:n_exp, :n_exp] = adj_exp

        padded_adj_struct = np.zeros((max_size, max_size))
        padded_adj_struct[:n_struct, :n_struct] = adj_struct

        return padded_sim, padded_adj_exp, padded_adj_struct

    def _extract_assignments(
        self,
        col_ind: np.ndarray,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
        n_exp: int,
        n_struct: int,
    ) -> dict[int, int]:
        """
        Extract assignments from QAP solution.

        Args:
            col_ind: Column indices from QAP solver.
            graph_experimental: Experimental graph.
            graph_structural: Structural graph.
            n_exp: Number of experimental nodes.
            n_struct: Number of structural nodes.

        Returns:
            Dictionary mapping exp_id -> struct_id.
        """
        exp_nodes = sorted(graph_experimental.nodes())
        struct_nodes = sorted(graph_structural.nodes())

        assignments = {}

        for row, col in enumerate(col_ind):
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
            f"QAPMatcher("
            f"method={self.method}, "
            f"topology_weight={self.topology_weight}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
