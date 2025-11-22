"""
Peak network builder for creating experimental graphs from NMR data.

This module creates graph representations of NMR peaks based on methyl-methyl
NOE correlations from 13C-13C-1H NOESY experimental data.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import networkx as nx

from methyl_match.reading.noesy_parser import NOEPeak
from methyl_match.reading.hmqc_parser import HMQCPeak

logger = logging.getLogger(__name__)


@dataclass
class PeakNetwork:
    """
    Represents a graph network of NMR peaks from experimental data.

    Attributes:
        graph: NetworkX graph with peaks as nodes
        hmqc_peaks: List of HMQCPeak objects
        noe_peaks: List of NOEPeak objects used for edges
        labels: List of peak labels/assignments
        node_features: Dictionary mapping node IDs to feature arrays
        edge_features: Dictionary mapping edge tuples to feature arrays
        adjacency_matrix: Numpy adjacency matrix
        correlation_matrix: Numpy correlation matrix (NOE intensities)
    """
    graph: nx.Graph
    hmqc_peaks: List[HMQCPeak]
    noe_peaks: List[NOEPeak]
    labels: List[str]
    node_features: Dict[int, np.ndarray]
    edge_features: Dict[Tuple[int, int], np.ndarray]
    adjacency_matrix: np.ndarray
    correlation_matrix: np.ndarray

    def __post_init__(self):
        """Validate the network."""
        if len(self.hmqc_peaks) != len(self.labels):
            raise ValueError("Number of HMQC peaks must match number of labels")
        if self.graph.number_of_nodes() != len(self.hmqc_peaks):
            raise ValueError("Graph nodes must match number of HMQC peaks")

    def get_neighbors(self, node_id: int, min_intensity: Optional[float] = None) -> List[int]:
        """
        Get neighboring nodes connected by NOE correlations.

        Args:
            node_id: Node ID
            min_intensity: Minimum NOE intensity threshold

        Returns:
            List of neighbor node IDs
        """
        neighbors = list(self.graph.neighbors(node_id))

        if min_intensity is not None:
            # Filter by NOE intensity
            filtered = []
            for neighbor in neighbors:
                edge_data = self.graph[node_id][neighbor]
                if edge_data.get('intensity', 0) >= min_intensity:
                    filtered.append(neighbor)
            return filtered

        return neighbors

    def get_subgraph(self, node_ids: List[int]) -> nx.Graph:
        """
        Extract a subgraph containing only specified nodes.

        Args:
            node_ids: List of node IDs to include

        Returns:
            NetworkX subgraph
        """
        return self.graph.subgraph(node_ids).copy()

    def get_ambiguity_report(self) -> Dict[str, any]:
        """
        Generate a comprehensive ambiguity report for the network.

        Returns:
            Dictionary containing:
                - total_edges: Total number of edges in the graph
                - ambiguity_scores: List of ambiguity scores for all edges
                - high_ambiguity_edges: Edges with score < 0.25 (very uncertain)
                - medium_ambiguity_edges: Edges with score 0.25-0.5
                - low_ambiguity_edges: Edges with score 0.5-1.0
                - certain_edges: Edges with score = 1.0 (no ambiguity)
                - avg_ambiguity_score: Mean ambiguity score
                - avg_candidates_w1: Mean number of candidates for w1
                - avg_candidates_w2: Mean number of candidates for w2
        """
        edges = list(self.graph.edges(data=True))

        if not edges:
            return {
                'total_edges': 0,
                'ambiguity_scores': [],
                'high_ambiguity_edges': [],
                'medium_ambiguity_edges': [],
                'low_ambiguity_edges': [],
                'certain_edges': [],
                'avg_ambiguity_score': 0.0,
                'avg_candidates_w1': 0.0,
                'avg_candidates_w2': 0.0,
            }

        ambiguity_scores = []
        candidates_w1 = []
        candidates_w2 = []
        high_ambig = []
        medium_ambig = []
        low_ambig = []
        certain = []

        for u, v, data in edges:
            score = data.get('ambiguity_score', 1.0)
            ambiguity_scores.append(score)
            candidates_w1.append(data.get('num_candidates_w1', 1))
            candidates_w2.append(data.get('num_candidates_w2', 1))

            edge_info = (u, v, score)
            if score == 1.0:
                certain.append(edge_info)
            elif score >= 0.5:
                low_ambig.append(edge_info)
            elif score >= 0.25:
                medium_ambig.append(edge_info)
            else:
                high_ambig.append(edge_info)

        return {
            'total_edges': len(edges),
            'ambiguity_scores': ambiguity_scores,
            'high_ambiguity_edges': high_ambig,
            'medium_ambiguity_edges': medium_ambig,
            'low_ambiguity_edges': low_ambig,
            'certain_edges': certain,
            'avg_ambiguity_score': np.mean(ambiguity_scores) if ambiguity_scores else 0.0,
            'avg_candidates_w1': np.mean(candidates_w1) if candidates_w1 else 0.0,
            'avg_candidates_w2': np.mean(candidates_w2) if candidates_w2 else 0.0,
        }

    def log_ambiguity_summary(self):
        """
        Log a human-readable summary of edge ambiguity to the logger.
        """
        report = self.get_ambiguity_report()

        logger.info("=" * 60)
        logger.info("PEAK NETWORK AMBIGUITY SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total edges: {report['total_edges']}")
        logger.info(f"Average ambiguity score: {report['avg_ambiguity_score']:.3f} (1.0 = certain, 0.0 = highly ambiguous)")
        logger.info(f"Average candidates per NOE dimension:")
        logger.info(f"  w1 (13C-1): {report['avg_candidates_w1']:.2f}")
        logger.info(f"  w2 (13C-2): {report['avg_candidates_w2']:.2f}")
        logger.info("")
        logger.info("Edge ambiguity breakdown:")
        logger.info(f"  Certain (score = 1.0):       {len(report['certain_edges']):4d} ({100*len(report['certain_edges'])/max(report['total_edges'],1):.1f}%)")
        logger.info(f"  Low ambiguity (0.5-1.0):     {len(report['low_ambiguity_edges']):4d} ({100*len(report['low_ambiguity_edges'])/max(report['total_edges'],1):.1f}%)")
        logger.info(f"  Medium ambiguity (0.25-0.5): {len(report['medium_ambiguity_edges']):4d} ({100*len(report['medium_ambiguity_edges'])/max(report['total_edges'],1):.1f}%)")
        logger.info(f"  High ambiguity (< 0.25):     {len(report['high_ambiguity_edges']):4d} ({100*len(report['high_ambiguity_edges'])/max(report['total_edges'],1):.1f}%)")
        logger.info("=" * 60)


class PeakNetworkBuilder:
    """
    Builder for creating peak network graphs from NMR experimental data.

    This builder creates a graph where:
    - Nodes represent HMQC peaks (methyl 1H-13C correlations)
    - Edges connect peaks with methyl-methyl NOE correlations (13C-13C-1H NOESY)
    - Node features include chemical shifts and intensities
    - Edge features include NOE intensities and weights

    For methyl-methyl assignment, NOE peaks have w1=13C (methyl 1), w2=13C (methyl 2),
    and w3=1H. The builder matches NOE w1/w2 to HMQC 13C chemical shifts.

    Example:
        >>> from methyl_match.reading import HMQCParser, NOESYParser
        >>> hmqc_parser = HMQCParser("hmqc.txt")
        >>> noesy_parser = NOESYParser("noesy.txt")
        >>> hmqc_peaks = hmqc_parser.parse()
        >>> noe_peaks = noesy_parser.parse()
        >>> builder = PeakNetworkBuilder(intensity_threshold=30000)
        >>> network = builder.build_network(hmqc_peaks, noe_peaks)
        >>> print(f"Nodes: {network.graph.number_of_nodes()}")
        >>> print(f"Edges: {network.graph.number_of_edges()}")
    """

    def __init__(
        self,
        intensity_threshold: float = 0.0,
        normalize_intensities: bool = True,
        chemical_shift_tolerance: Dict[str, float] = None,
        use_2d_matching: bool = False,
        create_ambiguous_edges: bool = False,
        max_ambiguous_candidates: int = 3
    ):
        """
        Initialize the peak network builder.

        Args:
            intensity_threshold: Minimum NOE intensity to create an edge
            normalize_intensities: Whether to normalize NOE intensities
            chemical_shift_tolerance: Dict with 'h' and 'c' tolerances (ppm)
            use_2d_matching: Use 2D matching (13C + 1H) instead of 1D (13C only)
            create_ambiguous_edges: Create multiple edge hypotheses for ambiguous NOE peaks
            max_ambiguous_candidates: Maximum candidates per dimension for ambiguous edges
        """
        self.intensity_threshold = intensity_threshold
        self.normalize_intensities = normalize_intensities
        self.chemical_shift_tolerance = chemical_shift_tolerance or {'h': 0.05, 'c': 0.5}
        self.use_2d_matching = use_2d_matching
        self.create_ambiguous_edges = create_ambiguous_edges
        self.max_ambiguous_candidates = max_ambiguous_candidates
        logger.info(f"Initialized PeakNetworkBuilder with intensity_threshold={intensity_threshold}, "
                   f"use_2d_matching={use_2d_matching}, create_ambiguous_edges={create_ambiguous_edges}")

    def build_network(
        self,
        hmqc_peaks: List[HMQCPeak],
        noe_peaks: List[NOEPeak]
    ) -> PeakNetwork:
        """
        Build a peak network from HMQC and NOESY data.

        Args:
            hmqc_peaks: List of HMQCPeak objects (nodes)
            noe_peaks: List of NOEPeak objects (edges)

        Returns:
            PeakNetwork object

        Example:
            >>> builder = PeakNetworkBuilder()
            >>> network = builder.build_network(hmqc_peaks, noe_peaks)
            >>> print(f"Network density: {nx.density(network.graph):.3f}")
        """
        if not hmqc_peaks:
            raise ValueError("Cannot build network from empty HMQC peak list")

        logger.info(f"Building network from {len(hmqc_peaks)} HMQC peaks and {len(noe_peaks)} NOE peaks")

        # Create graph
        graph = nx.Graph()

        # Add nodes with features
        labels = []
        node_features = {}

        for i, peak in enumerate(hmqc_peaks):
            label = peak.assignment if peak.assignment else f"Peak_{i+1}"
            graph.add_node(i, label=label)
            labels.append(label)
            node_features[i] = self._compute_node_features(peak, i, len(hmqc_peaks))

        # Normalize NOE intensities if requested
        noe_intensities = [p.intensity for p in noe_peaks]
        if self.normalize_intensities and noe_intensities:
            max_intensity = max(noe_intensities)
            min_intensity = min(noe_intensities)
            intensity_range = max_intensity - min_intensity if max_intensity > min_intensity else 1.0
        else:
            intensity_range = 1.0
            min_intensity = 0.0

        # Build correlation matrix and add edges
        correlation_matrix = np.zeros((len(hmqc_peaks), len(hmqc_peaks)))
        edge_features = {}
        edge_count = 0

        # Create mapping from assignments to node indices
        assignment_to_node = {}
        for i, peak in enumerate(hmqc_peaks):
            if peak.assignment:
                assignment_to_node[peak.assignment] = i

        # Track ambiguity statistics
        ambiguity_stats = {
            'total_noe_peaks': 0,
            'matched_edges': 0,
            'unmatched_w1': 0,
            'unmatched_w2': 0,
            'ambiguous_w1': 0,
            'ambiguous_w2': 0,
        }

        # Add edges based on NOE correlations
        for noe_peak in noe_peaks:
            ambiguity_stats['total_noe_peaks'] += 1

            if noe_peak.intensity < self.intensity_threshold:
                continue

            # Find corresponding HMQC peaks for this NOE
            if self.use_2d_matching:
                # Use 2D matching (13C + 1H) for better discrimination
                node1_id, num_cand_w1, dist_w1 = self._find_matching_node_2d(
                    noe_peak.assignment1,
                    noe_peak.w1,
                    noe_peak.w3,  # Use 1H dimension for 2D matching
                    hmqc_peaks,
                    assignment_to_node
                )
                node2_id, num_cand_w2, dist_w2 = self._find_matching_node_2d(
                    noe_peak.assignment2,
                    noe_peak.w2,
                    noe_peak.w3,  # Use 1H dimension for 2D matching
                    hmqc_peaks,
                    assignment_to_node
                )
            else:
                # Use 1D matching (13C only)
                node1_id, num_cand_w1, dist_w1 = self._find_matching_node(
                    noe_peak.assignment1,
                    noe_peak.w1,
                    hmqc_peaks,
                    assignment_to_node
                )
                node2_id, num_cand_w2, dist_w2 = self._find_matching_node(
                    noe_peak.assignment2,
                    noe_peak.w2,
                    hmqc_peaks,
                    assignment_to_node
                )

            # Track unmatched NOE dimensions
            if node1_id is None:
                ambiguity_stats['unmatched_w1'] += 1
            if node2_id is None:
                ambiguity_stats['unmatched_w2'] += 1

            # Track ambiguous matches (multiple candidates)
            if num_cand_w1 > 1:
                ambiguity_stats['ambiguous_w1'] += 1
            if num_cand_w2 > 1:
                ambiguity_stats['ambiguous_w2'] += 1

            # Decide whether to create single edge (best match) or multiple edges (all hypotheses)
            if self.create_ambiguous_edges and (num_cand_w1 > 1 or num_cand_w2 > 1):
                # Create multiple edge hypotheses for ambiguous NOE peaks
                candidates_w1 = self._get_all_matching_candidates(
                    noe_peak.assignment1, noe_peak.w1, noe_peak.w3,
                    hmqc_peaks, assignment_to_node, self.max_ambiguous_candidates
                )
                candidates_w2 = self._get_all_matching_candidates(
                    noe_peak.assignment2, noe_peak.w2, noe_peak.w3,
                    hmqc_peaks, assignment_to_node, self.max_ambiguous_candidates
                )

                # Create edges for all combinations of candidates
                for node1, dist1 in candidates_w1:
                    for node2, dist2 in candidates_w2:
                        if node1 == node2:
                            continue

                        ambiguity_stats['matched_edges'] += 1

                        # Normalize intensity
                        if self.normalize_intensities:
                            normalized_intensity = (noe_peak.intensity - min_intensity) / intensity_range
                        else:
                            normalized_intensity = noe_peak.intensity

                        # Compute ambiguity score based on number of combinations
                        ambiguity_score = 1.0 / (len(candidates_w1) * len(candidates_w2))

                        # Weight by inverse distance (closer matches get higher weight)
                        distance_weight = 1.0 / (1.0 + dist1 + dist2)
                        weighted_score = ambiguity_score * distance_weight

                        # Add or update edge
                        if not graph.has_edge(node1, node2) or \
                           graph[node1][node2].get('ambiguity_score', 0) < ambiguity_score:
                            graph.add_edge(
                                node1, node2,
                                intensity=normalized_intensity,
                                weight=weighted_score,
                                ambiguity_score=ambiguity_score,
                                num_candidates_w1=len(candidates_w1),
                                num_candidates_w2=len(candidates_w2),
                                min_dist_w1=dist1,
                                min_dist_w2=dist2,
                                is_hypothesis=True
                            )
                            edge_features[(node1, node2)] = self._compute_edge_features(
                                noe_peak, normalized_intensity
                            )
                            edge_features[(node2, node1)] = edge_features[(node1, node2)]
                            correlation_matrix[node1, node2] = noe_peak.intensity
                            correlation_matrix[node2, node1] = noe_peak.intensity
                            edge_count += 1

            elif node1_id is not None and node2_id is not None and node1_id != node2_id:
                # Create single edge with best match
                ambiguity_stats['matched_edges'] += 1

                # Normalize intensity
                if self.normalize_intensities:
                    normalized_intensity = (noe_peak.intensity - min_intensity) / intensity_range
                else:
                    normalized_intensity = noe_peak.intensity

                # Compute ambiguity score (1.0 = certain, lower = more ambiguous)
                ambiguity_score = 1.0 / (num_cand_w1 * num_cand_w2)

                # Add or update edge (use maximum intensity if multiple NOEs)
                if graph.has_edge(node1_id, node2_id):
                    # Update if this NOE has higher intensity
                    existing_intensity = graph[node1_id][node2_id]['intensity']
                    if normalized_intensity > existing_intensity:
                        graph[node1_id][node2_id]['intensity'] = normalized_intensity
                        graph[node1_id][node2_id]['weight'] = normalized_intensity
                        graph[node1_id][node2_id]['ambiguity_score'] = ambiguity_score
                        graph[node1_id][node2_id]['num_candidates_w1'] = num_cand_w1
                        graph[node1_id][node2_id]['num_candidates_w2'] = num_cand_w2
                        graph[node1_id][node2_id]['min_dist_w1'] = dist_w1
                        graph[node1_id][node2_id]['min_dist_w2'] = dist_w2
                        graph[node1_id][node2_id]['is_hypothesis'] = False
                        correlation_matrix[node1_id, node2_id] = noe_peak.intensity
                        correlation_matrix[node2_id, node1_id] = noe_peak.intensity
                else:
                    graph.add_edge(
                        node1_id, node2_id,
                        intensity=normalized_intensity,
                        weight=normalized_intensity,
                        ambiguity_score=ambiguity_score,
                        num_candidates_w1=num_cand_w1,
                        num_candidates_w2=num_cand_w2,
                        min_dist_w1=dist_w1,
                        min_dist_w2=dist_w2,
                        is_hypothesis=False
                    )
                    edge_features[(node1_id, node2_id)] = self._compute_edge_features(
                        noe_peak, normalized_intensity
                    )
                    edge_features[(node2_id, node1_id)] = edge_features[(node1_id, node2_id)]
                    correlation_matrix[node1_id, node2_id] = noe_peak.intensity
                    correlation_matrix[node2_id, node1_id] = noe_peak.intensity
                    edge_count += 1

        # Create adjacency matrix
        adjacency_matrix = nx.to_numpy_array(graph)

        logger.info(f"Built network: {len(hmqc_peaks)} nodes, {edge_count} edges")
        logger.info(f"Network density: {nx.density(graph):.3f}")

        # Log ambiguity statistics
        logger.info(f"Ambiguity report:")
        logger.info(f"  Total NOE peaks processed: {ambiguity_stats['total_noe_peaks']}")
        logger.info(f"  Successfully matched edges: {ambiguity_stats['matched_edges']}")
        logger.info(f"  Unmatched NOE w1 dimensions: {ambiguity_stats['unmatched_w1']}")
        logger.info(f"  Unmatched NOE w2 dimensions: {ambiguity_stats['unmatched_w2']}")
        logger.info(f"  Ambiguous w1 matches (>1 candidate): {ambiguity_stats['ambiguous_w1']}")
        logger.info(f"  Ambiguous w2 matches (>1 candidate): {ambiguity_stats['ambiguous_w2']}")

        if ambiguity_stats['matched_edges'] > 0:
            w1_ambig_pct = 100.0 * ambiguity_stats['ambiguous_w1'] / ambiguity_stats['matched_edges']
            w2_ambig_pct = 100.0 * ambiguity_stats['ambiguous_w2'] / ambiguity_stats['matched_edges']
            logger.info(f"  Ambiguity rate: w1={w1_ambig_pct:.1f}%, w2={w2_ambig_pct:.1f}%")

        return PeakNetwork(
            graph=graph,
            hmqc_peaks=hmqc_peaks,
            noe_peaks=noe_peaks,
            labels=labels,
            node_features=node_features,
            edge_features=edge_features,
            adjacency_matrix=adjacency_matrix,
            correlation_matrix=correlation_matrix
        )

    def _compute_node_features(
        self,
        peak: HMQCPeak,
        node_id: int,
        total_nodes: int
    ) -> np.ndarray:
        """
        Compute feature vector for a node.

        Features include:
        - 1H chemical shift
        - 13C chemical shift
        - Peak intensity (normalized)
        - Node degree (computed later, initially 0)

        Args:
            peak: HMQCPeak object
            node_id: Node index
            total_nodes: Total number of nodes

        Returns:
            Feature vector (4 dimensions)
        """
        # Chemical shifts
        h_shift = peak.h_shift
        c_shift = peak.c_shift

        # Normalized intensity (rough normalization)
        normalized_intensity = peak.intensity / 100000.0

        # Combine features
        features = np.array([
            h_shift,
            c_shift,
            normalized_intensity,
            0.0  # Degree (will be computed from graph)
        ], dtype=np.float32)

        return features

    def _compute_edge_features(
        self,
        noe_peak: NOEPeak,
        normalized_intensity: float
    ) -> np.ndarray:
        """
        Compute feature vector for an edge.

        Features include:
        - NOE intensity (normalized)
        - Edge weight (same as intensity)

        Args:
            noe_peak: NOEPeak object
            normalized_intensity: Normalized NOE intensity

        Returns:
            Feature vector (2 dimensions)
        """
        return np.array([
            normalized_intensity,
            normalized_intensity  # Weight
        ], dtype=np.float32)

    def _find_matching_node(
        self,
        assignment: Optional[str],
        chemical_shift: float,
        hmqc_peaks: List[HMQCPeak],
        assignment_to_node: Dict[str, int]
    ) -> Tuple[Optional[int], int, float]:
        """
        Find the HMQC node that matches a NOE assignment or chemical shift.

        For 13C-13C-1H methyl-methyl NOESY, the NOE peak w1 and w2 dimensions
        correspond to 13C chemical shifts, so we match against HMQC 13C shifts.

        This method now returns the BEST match (closest by chemical shift distance)
        rather than the first match, along with ambiguity metadata.

        Args:
            assignment: Peak assignment label
            chemical_shift: 13C chemical shift from NOE (w1 or w2)
            hmqc_peaks: List of HMQC peaks
            assignment_to_node: Dict mapping assignments to node IDs

        Returns:
            Tuple of (node_id, num_candidates, min_distance):
                - node_id: Best matching node ID or None if no match found
                - num_candidates: Number of HMQC peaks within tolerance
                - min_distance: Chemical shift distance to best match (ppm)
        """
        # Try exact assignment match first
        if assignment and assignment in assignment_to_node:
            node_id = assignment_to_node[assignment]
            # For exact assignment match, return with certainty metadata
            return (node_id, 1, 0.0)

        # Find all candidates within chemical shift tolerance on 13C dimension
        c_tol = self.chemical_shift_tolerance['c']
        candidates = []

        for i, peak in enumerate(hmqc_peaks):
            distance = abs(peak.c_shift - chemical_shift)
            if distance <= c_tol:
                candidates.append((i, distance))

        # No match found
        if not candidates:
            return (None, 0, float('inf'))

        # Sort by distance and return best match
        candidates.sort(key=lambda x: x[1])
        best_node_id, min_distance = candidates[0]
        num_candidates = len(candidates)

        # Log ambiguity warning if multiple candidates
        if num_candidates > 1:
            logger.debug(
                f"Ambiguous match: {num_candidates} HMQC peaks within {c_tol} ppm "
                f"of NOE shift {chemical_shift:.2f} ppm. Selected closest match "
                f"(node {best_node_id}, distance={min_distance:.3f} ppm)"
            )

        return (best_node_id, num_candidates, min_distance)

    def _find_matching_node_2d(
        self,
        assignment: Optional[str],
        c_shift: float,
        h_shift: Optional[float],
        hmqc_peaks: List[HMQCPeak],
        assignment_to_node: Dict[str, int]
    ) -> Tuple[Optional[int], int, float]:
        """
        Find the HMQC node using 2D matching (13C + 1H chemical shifts).

        This method improves upon 1D matching by considering both 13C and 1H
        dimensions, providing better discrimination when 1H shifts are available.

        Args:
            assignment: Peak assignment label
            c_shift: 13C chemical shift from NOE
            h_shift: 1H chemical shift from NOE (w3), may be None
            hmqc_peaks: List of HMQC peaks
            assignment_to_node: Dict mapping assignments to node IDs

        Returns:
            Tuple of (node_id, num_candidates, min_distance):
                - node_id: Best matching node ID or None if no match found
                - num_candidates: Number of HMQC peaks within tolerance
                - min_distance: Combined 2D distance to best match
        """
        # Try exact assignment match first
        if assignment and assignment in assignment_to_node:
            node_id = assignment_to_node[assignment]
            return (node_id, 1, 0.0)

        # If no H shift available, fall back to 1D matching
        if h_shift is None:
            return self._find_matching_node(
                assignment, c_shift, hmqc_peaks, assignment_to_node
            )

        # Find all candidates within 2D tolerance
        c_tol = self.chemical_shift_tolerance['c']
        h_tol = self.chemical_shift_tolerance['h']
        candidates = []

        for i, peak in enumerate(hmqc_peaks):
            c_distance = abs(peak.c_shift - c_shift)
            h_distance = abs(peak.h_shift - h_shift)

            # Both dimensions must be within tolerance
            if c_distance <= c_tol and h_distance <= h_tol:
                # Compute combined distance (normalized Euclidean distance)
                combined_distance = np.sqrt(
                    (c_distance / c_tol) ** 2 + (h_distance / h_tol) ** 2
                )
                candidates.append((i, combined_distance))

        # No match found
        if not candidates:
            return (None, 0, float('inf'))

        # Sort by combined distance and return best match
        candidates.sort(key=lambda x: x[1])
        best_node_id, min_distance = candidates[0]
        num_candidates = len(candidates)

        # Log ambiguity warning if multiple candidates
        if num_candidates > 1:
            logger.debug(
                f"Ambiguous 2D match: {num_candidates} HMQC peaks within tolerance "
                f"(13C={c_shift:.2f}±{c_tol}, 1H={h_shift:.3f}±{h_tol}). "
                f"Selected closest match (node {best_node_id}, distance={min_distance:.3f})"
            )

        return (best_node_id, num_candidates, min_distance)

    def _get_all_matching_candidates(
        self,
        assignment: Optional[str],
        c_shift: float,
        h_shift: Optional[float],
        hmqc_peaks: List[HMQCPeak],
        assignment_to_node: Dict[str, int],
        max_candidates: int = 3
    ) -> List[Tuple[int, float]]:
        """
        Get all candidate matches for a NOE dimension, sorted by distance.

        This is used when create_ambiguous_edges=True to get multiple
        plausible matches instead of just the best one.

        Args:
            assignment: Peak assignment label
            c_shift: 13C chemical shift from NOE
            h_shift: 1H chemical shift from NOE (for 2D matching), may be None
            hmqc_peaks: List of HMQC peaks
            assignment_to_node: Dict mapping assignments to node IDs
            max_candidates: Maximum number of candidates to return

        Returns:
            List of (node_id, distance) tuples, sorted by distance (best first)
        """
        # If exact assignment exists, return only that
        if assignment and assignment in assignment_to_node:
            return [(assignment_to_node[assignment], 0.0)]

        c_tol = self.chemical_shift_tolerance['c']
        candidates = []

        # Use 2D matching if H shift available and 2D matching enabled
        if self.use_2d_matching and h_shift is not None:
            h_tol = self.chemical_shift_tolerance['h']

            for i, peak in enumerate(hmqc_peaks):
                c_distance = abs(peak.c_shift - c_shift)
                h_distance = abs(peak.h_shift - h_shift)

                if c_distance <= c_tol and h_distance <= h_tol:
                    combined_distance = np.sqrt(
                        (c_distance / c_tol) ** 2 + (h_distance / h_tol) ** 2
                    )
                    candidates.append((i, combined_distance))
        else:
            # Use 1D matching
            for i, peak in enumerate(hmqc_peaks):
                distance = abs(peak.c_shift - c_shift)
                if distance <= c_tol:
                    candidates.append((i, distance))

        # Sort by distance and limit to max_candidates
        candidates.sort(key=lambda x: x[1])
        return candidates[:max_candidates]

    def get_statistics(self, network: PeakNetwork) -> Dict[str, float]:
        """
        Compute network statistics.

        Args:
            network: PeakNetwork object

        Returns:
            Dictionary of statistics

        Example:
            >>> stats = builder.get_statistics(network)
            >>> print(f"Average degree: {stats['avg_degree']:.2f}")
        """
        graph = network.graph

        stats = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_degree': np.mean([d for n, d in graph.degree()]),
            'avg_clustering': nx.average_clustering(graph),
            'num_components': nx.number_connected_components(graph),
        }

        # Intensity statistics
        intensities = [data['intensity'] for u, v, data in graph.edges(data=True)]
        if intensities:
            stats['avg_intensity'] = np.mean(intensities)
            stats['min_intensity'] = np.min(intensities)
            stats['max_intensity'] = np.max(intensities)

        logger.info(f"Network statistics: {stats}")
        return stats

    def visualize_network(self, network: PeakNetwork, save_path: Optional[str] = None):
        """
        Create a 2D visualization of the network.

        Args:
            network: PeakNetwork object
            save_path: Optional path to save the figure

        Note:
            Requires matplotlib. For interactive visualization, use the marimo notebook.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib required for visualization")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Use spring layout for 2D positioning
        pos = nx.spring_layout(network.graph, seed=42, k=0.5)

        # Draw nodes
        nx.draw_networkx_nodes(
            network.graph, pos,
            node_color='lightcoral',
            node_size=300,
            ax=ax
        )

        # Draw edges with width proportional to NOE intensity
        edges = network.graph.edges()
        weights = [network.graph[u][v]['intensity'] * 5 for u, v in edges]

        nx.draw_networkx_edges(
            network.graph, pos,
            width=weights,
            alpha=0.4,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            network.graph, pos,
            labels={i: network.labels[i] for i in range(len(network.labels))},
            font_size=8,
            ax=ax
        )

        ax.set_title(f"Peak Network ({len(network.noe_peaks)} NOE correlations)")
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")

        plt.tight_layout()
        return fig
