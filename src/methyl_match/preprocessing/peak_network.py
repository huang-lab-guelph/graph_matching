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
        chemical_shift_tolerance: Dict[str, float] = None
    ):
        """
        Initialize the peak network builder.

        Args:
            intensity_threshold: Minimum NOE intensity to create an edge
            normalize_intensities: Whether to normalize NOE intensities
            chemical_shift_tolerance: Dict with 'h' and 'c' tolerances (ppm)
        """
        self.intensity_threshold = intensity_threshold
        self.normalize_intensities = normalize_intensities
        self.chemical_shift_tolerance = chemical_shift_tolerance or {'h': 0.05, 'c': 0.5}
        logger.info(f"Initialized PeakNetworkBuilder with intensity_threshold={intensity_threshold}")

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

        # Add edges based on NOE correlations
        for noe_peak in noe_peaks:
            if noe_peak.intensity < self.intensity_threshold:
                continue

            # Find corresponding HMQC peaks for this NOE
            node1_id = self._find_matching_node(
                noe_peak.assignment1,
                noe_peak.w1,
                hmqc_peaks,
                assignment_to_node
            )
            node2_id = self._find_matching_node(
                noe_peak.assignment2,
                noe_peak.w2,
                hmqc_peaks,
                assignment_to_node
            )

            if node1_id is not None and node2_id is not None and node1_id != node2_id:
                # Normalize intensity
                if self.normalize_intensities:
                    normalized_intensity = (noe_peak.intensity - min_intensity) / intensity_range
                else:
                    normalized_intensity = noe_peak.intensity

                # Add or update edge (use maximum intensity if multiple NOEs)
                if graph.has_edge(node1_id, node2_id):
                    # Update if this NOE has higher intensity
                    existing_intensity = graph[node1_id][node2_id]['intensity']
                    if normalized_intensity > existing_intensity:
                        graph[node1_id][node2_id]['intensity'] = normalized_intensity
                        graph[node1_id][node2_id]['weight'] = normalized_intensity
                        correlation_matrix[node1_id, node2_id] = noe_peak.intensity
                        correlation_matrix[node2_id, node1_id] = noe_peak.intensity
                else:
                    graph.add_edge(
                        node1_id, node2_id,
                        intensity=normalized_intensity,
                        weight=normalized_intensity
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
    ) -> Optional[int]:
        """
        Find the HMQC node that matches a NOE assignment or chemical shift.

        For 13C-13C-1H methyl-methyl NOESY, the NOE peak w1 and w2 dimensions
        correspond to 13C chemical shifts, so we match against HMQC 13C shifts.

        Args:
            assignment: Peak assignment label
            chemical_shift: 13C chemical shift from NOE (w1 or w2)
            hmqc_peaks: List of HMQC peaks
            assignment_to_node: Dict mapping assignments to node IDs

        Returns:
            Node ID or None if no match found
        """
        # Try exact assignment match first
        if assignment and assignment in assignment_to_node:
            return assignment_to_node[assignment]

        # Fall back to chemical shift matching on 13C dimension
        c_tol = self.chemical_shift_tolerance['c']

        for i, peak in enumerate(hmqc_peaks):
            if abs(peak.c_shift - chemical_shift) <= c_tol:
                return i

        return None

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
