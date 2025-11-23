"""
Spectral graph matching algorithm.

This module implements spectral graph matching using eigendecomposition of
adjacency or Laplacian matrices. It uses the pygmtools library for the
implementation.
"""

import time
from typing import Optional
import numpy as np
import networkx as nx

try:
    import pygmtools as pygm
    pygm.set_backend('numpy')
    PYGMTOOLS_AVAILABLE = True
except ImportError:
    PYGMTOOLS_AVAILABLE = False

from .base import GraphMatcher, MatchingResult
from .utils import (
    compute_confidence_scores,
    create_cost_matrix_from_similarity,
)


class SpectralMatcher(GraphMatcher):
    """
    Spectral graph matching algorithm.

    This algorithm uses eigendecomposition of graph matrices (adjacency or
    Laplacian) to find correspondences between graphs. It compares the spectral
    properties of the two graphs to find the best alignment.

    The algorithm is based on Umeyama's method and uses the pygmtools library
    for implementation. It captures global graph structure through eigenvalues
    and eigenvectors.

    Spectral matching is particularly good at:
    - Handling noisy graphs
    - Capturing global structure
    - Providing smooth relaxations of discrete problems

    Note: Requires pygmtools library to be installed.

    Parameters:
    ----------
    confidence_threshold : float, default=0.0
        Minimum confidence score to accept an assignment.
    method : str, default="sm"
        Spectral matching method to use:
        - "sm": Standard spectral matching (Umeyama)
        - "ipfp": Integer Projected Fixed Point (iterative refinement)
        - "rrwm": Reweighted Random Walk Matching
    use_laplacian : bool, default=False
        If True, use Laplacian matrix instead of adjacency matrix.
    n_eigs : Optional[int], default=None
        Number of eigenvectors to use. If None, uses all.
    normalize : bool, default=True
        If True, normalize the adjacency/Laplacian matrices.

    Attributes:
    ----------
    name : str
        Algorithm name ("SpectralMatcher").
    """

    def __init__(
        self,
        confidence_threshold: float = 0.0,
        method: str = "sm",
        use_laplacian: bool = False,
        n_eigs: Optional[int] = None,
        normalize: bool = True,
        **config,
    ):
        """
        Initialize SpectralMatcher.

        Args:
            confidence_threshold: Minimum confidence to accept assignment (0.0-1.0).
            method: Spectral matching method ("sm", "ipfp", or "rrwm").
            use_laplacian: Whether to use Laplacian instead of adjacency.
            n_eigs: Number of eigenvectors to use (None = all).
            normalize: Whether to normalize matrices.
            **config: Additional configuration parameters.

        Raises:
            ImportError: If pygmtools is not installed.
        """
        if not PYGMTOOLS_AVAILABLE:
            raise ImportError(
                "pygmtools is required for SpectralMatcher. "
                "Install with: pip install pygmtools"
            )

        super().__init__(
            name="SpectralMatcher",
            confidence_threshold=confidence_threshold,
            method=method,
            use_laplacian=use_laplacian,
            n_eigs=n_eigs,
            normalize=normalize,
            **config,
        )
        self.confidence_threshold = confidence_threshold
        self.method = method
        self.use_laplacian = use_laplacian
        self.n_eigs = n_eigs
        self.normalize = normalize

    def match(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> MatchingResult:
        """
        Perform spectral graph matching.

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

        # Build node similarity matrix
        similarity_matrix = self._build_similarity_matrix(
            graph_experimental, graph_structural
        )

        # Build adjacency matrices
        adj_exp = self._build_adjacency_matrix(graph_experimental)
        adj_struct = self._build_adjacency_matrix(graph_structural)

        # Convert to Laplacian if requested
        if self.use_laplacian:
            adj_exp = self._adjacency_to_laplacian(adj_exp)
            adj_struct = self._adjacency_to_laplacian(adj_struct)

        # Normalize if requested
        if self.normalize:
            adj_exp = self._normalize_matrix(adj_exp)
            adj_struct = self._normalize_matrix(adj_struct)

        # Handle rectangular case
        if n_exp != n_struct:
            similarity_matrix, adj_exp, adj_struct = self._pad_matrices(
                similarity_matrix, adj_exp, adj_struct, n_exp, n_struct
            )
            max_size = int(max(n_exp, n_struct))
        else:
            max_size = int(n_exp)

        # Convert similarity to affinity matrix in the format expected by pygmtools
        # K should be (n1*n2, n1*n2) for pairwise node affinity, OR
        # connectivity matrices A1, A2 with node similarities
        # For simplicity, we use connectivity matrices and node features
        conn1 = adj_exp
        conn2 = adj_struct

        # Check if graph size is reasonable for spectral matching
        # The affinity matrix is (n*n, n*n), which becomes impractical for large n
        if max_size > 80:
            raise ValueError(
                f"SpectralMatcher is not suitable for large graphs (max_size={max_size}). "
                f"The affinity matrix would be ({max_size*max_size}, {max_size*max_size}), "
                f"requiring approximately {(max_size*max_size)**2 * 8 / 1e9:.2f} GB of memory. "
                f"For graphs with >{max_size} nodes, use GreedyMatcher, HungarianMatcher, or QAPMatcher instead."
            )

        # Build pairwise affinity tensor K
        # K[i*n2+j, a*n2+b] = node_sim[i,a] * node_sim[j,b] * edge_affinity[ij,ab]
        K = self._build_pairwise_affinity(similarity_matrix, adj_exp, adj_struct, max_size)

        # Perform spectral matching
        if self.method == "sm":
            # Standard spectral matching
            X = pygm.sm(K, n1=max_size, n2=max_size)
        elif self.method == "ipfp":
            # Integer Projected Fixed Point
            X_init = pygm.sm(K, n1=max_size, n2=max_size)
            # IPFP signature: ipfp(K, n1, n2, X_init) - positional args
            X = pygm.ipfp(K, max_size, max_size, X_init)
        elif self.method == "rrwm":
            # Reweighted Random Walk Matching
            X_init = pygm.sm(K, n1=max_size, n2=max_size)
            # RRWM signature: rrwm(K, n1, n2, X_init) - positional args
            X = pygm.rrwm(K, max_size, max_size, X_init)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Extract assignments from permutation matrix
        assignments = self._extract_assignments_from_matrix(
            X, graph_experimental, graph_structural, n_exp, n_struct
        )

        # Convert similarity to cost for confidence computation
        cost_matrix = create_cost_matrix_from_similarity(
            similarity_matrix[:n_exp, :n_struct]
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
            "algorithm": "spectral",
            "method": self.method,
            "runtime_seconds": elapsed_time,
            "n_experimental": n_exp,
            "n_structural": n_struct,
            "use_laplacian": self.use_laplacian,
            "n_eigs": self.n_eigs,
            "confidence_threshold": self.confidence_threshold,
        }

        return MatchingResult(
            assignments=filtered_assignments,
            confidence_scores=filtered_confidence,
            unassigned_peaks=unassigned,
            metadata=metadata,
            cost_matrix=cost_matrix,
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
                if graph.has_edge(node_i, node_j):
                    edge_data = graph.get_edge_data(node_i, node_j)
                    weight = edge_data.get('weight', 1.0)
                    adj[i, j] = weight

        return adj

    def _adjacency_to_laplacian(self, adj: np.ndarray) -> np.ndarray:
        """
        Convert adjacency matrix to Laplacian.

        Args:
            adj: Adjacency matrix.

        Returns:
            Laplacian matrix L = D - A.
        """
        degree = np.sum(adj, axis=1)
        L = np.diag(degree) - adj
        return L

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

        # Pad similarity with low similarity
        padded_sim = np.zeros((max_size, max_size))
        padded_sim[:n_exp, :n_struct] = similarity

        # Pad adjacency matrices with zeros
        padded_adj_exp = np.zeros((max_size, max_size))
        padded_adj_exp[:n_exp, :n_exp] = adj_exp

        padded_adj_struct = np.zeros((max_size, max_size))
        padded_adj_struct[:n_struct, :n_struct] = adj_struct

        return padded_sim, padded_adj_exp, padded_adj_struct

    def _build_pairwise_affinity(
        self,
        similarity: np.ndarray,
        adj_exp: np.ndarray,
        adj_struct: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """
        Build pairwise affinity tensor for pygmtools.

        The affinity tensor K has shape (n*n, n*n) where K[i*n+j, a*n+b]
        represents the affinity of matching edge (i,j) to edge (a,b).

        Args:
            similarity: Node similarity matrix (n, n).
            adj_exp: Experimental adjacency (n, n).
            adj_struct: Structural adjacency (n, n).
            n: Number of nodes (same for both graphs after padding).

        Returns:
            Affinity tensor K of shape (n*n, n*n).
        """
        K = np.zeros((n * n, n * n))

        for i in range(n):
            for j in range(n):
                for a in range(n):
                    for b in range(n):
                        # Affinity of matching (i,j) to (a,b)
                        # Based on: node similarities and edge compatibility
                        node_affinity = similarity[i, a] * similarity[j, b]

                        # Edge compatibility: edges should match
                        if i == j and a == b:
                            # Diagonal: single node matching
                            edge_affinity = 1.0
                        else:
                            # Edge matching: both should be edges or both non-edges
                            exp_edge_weight = adj_exp[i, j]
                            struct_edge_weight = adj_struct[a, b]

                            if exp_edge_weight > 0 and struct_edge_weight > 0:
                                # Both are edges
                                edge_affinity = 1.0
                            elif exp_edge_weight == 0 and struct_edge_weight == 0:
                                # Both are non-edges
                                edge_affinity = 0.5
                            else:
                                # One is edge, one is not
                                edge_affinity = 0.0

                        K[i * n + j, a * n + b] = node_affinity * edge_affinity

        return K

    def _extract_assignments_from_matrix(
        self,
        X: np.ndarray,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
        n_exp: int,
        n_struct: int,
    ) -> dict[int, int]:
        """
        Extract assignments from permutation matrix.

        Args:
            X: Permutation matrix from spectral matching (n_exp, n_struct).
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

        # X is a soft assignment matrix, convert to hard assignment
        for i in range(n_exp):
            # Find best match for experimental node i
            j = np.argmax(X[i, :n_struct])

            # Only assign if above threshold
            if X[i, j] > 0.1:  # Threshold for accepting assignment
                exp_id = exp_nodes[i]
                struct_id = struct_nodes[j]
                assignments[exp_id] = struct_id

        return assignments

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpectralMatcher("
            f"method={self.method}, "
            f"use_laplacian={self.use_laplacian}, "
            f"confidence_threshold={self.confidence_threshold})"
        )
