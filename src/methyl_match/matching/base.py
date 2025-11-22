"""
Base classes for graph matching algorithms.

This module provides the abstract base class and result dataclass for all
graph matching algorithms in methyl_match.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx


@dataclass
class MatchingResult:
    """
    Result of a graph matching operation.

    This dataclass encapsulates all outputs from a graph matching algorithm,
    including the assignments, confidence scores, and metadata about the
    matching process.

    Attributes:
        assignments: Dictionary mapping experimental peak indices to structural
                    methyl indices. Keys are experimental node IDs, values are
                    structural node IDs.
        confidence_scores: Dictionary mapping experimental peak indices to
                          confidence scores (0.0-1.0). Higher scores indicate
                          more confident assignments.
        unassigned_peaks: List of experimental peak indices that could not be
                         confidently assigned.
        metadata: Additional algorithm-specific information (runtime, iterations,
                 cost, etc.).
        cost_matrix: Optional full cost matrix used for matching (n_exp × n_struct).
        similarity_matrix: Optional similarity matrix (n_exp × n_struct).
    """

    assignments: Dict[int, int]
    confidence_scores: Dict[int, float]
    unassigned_peaks: List[int] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    cost_matrix: Optional[np.ndarray] = None
    similarity_matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate the matching result."""
        # Ensure assignments and confidence scores have matching keys
        if set(self.assignments.keys()) != set(self.confidence_scores.keys()):
            raise ValueError(
                "assignments and confidence_scores must have the same keys"
            )

        # Ensure confidence scores are in valid range
        for peak_id, conf in self.confidence_scores.items():
            if not 0.0 <= conf <= 1.0:
                raise ValueError(
                    f"Confidence score for peak {peak_id} is {conf}, "
                    f"must be in range [0.0, 1.0]"
                )

    @property
    def num_assignments(self) -> int:
        """Number of confident assignments made."""
        return len(self.assignments)

    @property
    def num_unassigned(self) -> int:
        """Number of unassigned peaks."""
        return len(self.unassigned_peaks)

    @property
    def assignment_rate(self) -> float:
        """Fraction of peaks that were assigned."""
        total = self.num_assignments + self.num_unassigned
        if total == 0:
            return 0.0
        return self.num_assignments / total

    @property
    def mean_confidence(self) -> float:
        """Mean confidence score across all assignments."""
        if not self.confidence_scores:
            return 0.0
        return np.mean(list(self.confidence_scores.values()))

    def get_assignment_list(self) -> List[Tuple[int, int, float]]:
        """
        Get assignments as a list of (exp_id, struct_id, confidence) tuples.

        Returns:
            List of tuples (experimental_id, structural_id, confidence_score),
            sorted by confidence score (descending).
        """
        assignments = [
            (exp_id, self.assignments[exp_id], self.confidence_scores[exp_id])
            for exp_id in self.assignments.keys()
        ]
        return sorted(assignments, key=lambda x: x[2], reverse=True)

    def filter_by_confidence(self, threshold: float) -> "MatchingResult":
        """
        Create a new MatchingResult with only high-confidence assignments.

        Args:
            threshold: Minimum confidence score to keep (0.0-1.0).

        Returns:
            New MatchingResult with filtered assignments.
        """
        filtered_assignments = {
            exp_id: struct_id
            for exp_id, struct_id in self.assignments.items()
            if self.confidence_scores[exp_id] >= threshold
        }
        filtered_confidence = {
            exp_id: conf
            for exp_id, conf in self.confidence_scores.items()
            if conf >= threshold
        }
        filtered_unassigned = list(
            set(self.assignments.keys()) - set(filtered_assignments.keys())
        ) + self.unassigned_peaks

        return MatchingResult(
            assignments=filtered_assignments,
            confidence_scores=filtered_confidence,
            unassigned_peaks=filtered_unassigned,
            metadata={**self.metadata, "confidence_threshold": threshold},
            cost_matrix=self.cost_matrix,
            similarity_matrix=self.similarity_matrix,
        )

    def __repr__(self) -> str:
        """String representation of matching result."""
        return (
            f"MatchingResult("
            f"assignments={self.num_assignments}, "
            f"unassigned={self.num_unassigned}, "
            f"mean_confidence={self.mean_confidence:.3f})"
        )


class GraphMatcher(ABC):
    """
    Abstract base class for graph matching algorithms.

    All graph matching algorithms should inherit from this class and implement
    the match() method. This provides a unified interface for different matching
    approaches (greedy, optimal, spectral, neural, etc.).

    The general workflow is:
    1. Build similarity/cost matrices from node features
    2. Optionally incorporate edge information (topology)
    3. Solve the assignment problem
    4. Compute confidence scores
    5. Return MatchingResult

    Attributes:
        name: Human-readable name of the algorithm.
        config: Configuration dictionary for algorithm parameters.
    """

    def __init__(self, name: str = "GraphMatcher", **config):
        """
        Initialize graph matcher.

        Args:
            name: Algorithm name for identification.
            **config: Algorithm-specific configuration parameters.
        """
        self.name = name
        self.config = config

    @abstractmethod
    def match(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> MatchingResult:
        """
        Match experimental graph to structural graph.

        This is the main method that performs graph-to-graph matching. It takes
        two NetworkX graphs (experimental NMR peaks and structural methyls) and
        returns a MatchingResult containing the assignments and confidence scores.

        Args:
            graph_experimental: NetworkX graph of experimental NMR peaks.
                               Nodes should have attributes: 'h_shift', 'c_shift',
                               'intensity', and optionally 'residue_type'.
                               Edges represent NOE correlations with 'weight'.
            graph_structural: NetworkX graph of structural methyl groups.
                             Nodes should have attributes: 'position' (3D coords),
                             'residue_name', 'residue_number', and optionally
                             'residue_type'.
                             Edges represent spatial distances with 'weight'.

        Returns:
            MatchingResult object containing assignments, confidence scores,
            unassigned peaks, and metadata.

        Raises:
            ValueError: If graphs have invalid format or incompatible sizes.
        """
        pass

    def _validate_graphs(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> Tuple[int, int]:
        """
        Validate input graphs and return their sizes.

        Args:
            graph_experimental: Experimental graph to validate.
            graph_structural: Structural graph to validate.

        Returns:
            Tuple of (n_experimental, n_structural).

        Raises:
            ValueError: If graphs are invalid.
        """
        if not isinstance(graph_experimental, nx.Graph):
            raise ValueError("graph_experimental must be a NetworkX Graph")
        if not isinstance(graph_structural, nx.Graph):
            raise ValueError("graph_structural must be a NetworkX Graph")

        n_exp = int(graph_experimental.number_of_nodes())
        n_struct = int(graph_structural.number_of_nodes())

        if n_exp == 0:
            raise ValueError("graph_experimental has no nodes")
        if n_struct == 0:
            raise ValueError("graph_structural has no nodes")

        return n_exp, n_struct

    def _build_node_similarity_matrix(
        self,
        graph_experimental: nx.Graph,
        graph_structural: nx.Graph,
    ) -> np.ndarray:
        """
        Build node-to-node similarity matrix.

        Computes similarity between experimental peaks and structural methyls
        based on node features (chemical shifts, residue types, etc.).

        Args:
            graph_experimental: Experimental graph with node features.
            graph_structural: Structural graph with node features.

        Returns:
            Similarity matrix of shape (n_exp, n_struct). Higher values indicate
            more similar nodes.
        """
        n_exp = graph_experimental.number_of_nodes()
        n_struct = graph_structural.number_of_nodes()

        similarity = np.zeros((n_exp, n_struct))

        exp_nodes = list(graph_experimental.nodes())
        struct_nodes = list(graph_structural.nodes())

        for i, exp_id in enumerate(exp_nodes):
            exp_data = graph_experimental.nodes[exp_id]

            for j, struct_id in enumerate(struct_nodes):
                struct_data = graph_structural.nodes[struct_id]

                # Chemical shift similarity (if available)
                shift_sim = 0.0
                if 'h_shift' in exp_data and 'c_shift' in exp_data:
                    # Structural graph might have predicted shifts
                    # For now, we assume structural graph doesn't have shifts
                    # This will be customized by subclasses
                    shift_sim = 0.0

                # Residue type similarity
                residue_sim = 0.0
                if 'residue_type' in exp_data and 'residue_type' in struct_data:
                    residue_sim = 1.0 if exp_data['residue_type'] == struct_data['residue_type'] else 0.0

                # Combined similarity (can be customized by subclasses)
                similarity[i, j] = 0.5 * shift_sim + 0.5 * residue_sim

        return similarity

    def __repr__(self) -> str:
        """String representation of matcher."""
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.config.items())})"
