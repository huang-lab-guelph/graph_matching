"""
Methyl network builder for creating structure graphs from PDB data.

This module creates graph representations of protein methyl groups based on
spatial proximity in the protein structure.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import networkx as nx

from methyl_match.reading.pdb_parser import MethylGroup

logger = logging.getLogger(__name__)


@dataclass
class MethylNetwork:
    """
    Represents a graph network of methyl groups from protein structure.

    Attributes:
        graph: NetworkX graph with methyls as nodes
        methyls: List of MethylGroup objects
        labels: List of methyl labels
        node_features: Dictionary mapping node IDs to feature arrays
        edge_features: Dictionary mapping edge tuples to feature arrays
        adjacency_matrix: Numpy adjacency matrix
        distance_matrix: Numpy distance matrix
    """
    graph: nx.Graph
    methyls: List[MethylGroup]
    labels: List[str]
    node_features: Dict[int, np.ndarray]
    edge_features: Dict[Tuple[int, int], np.ndarray]
    adjacency_matrix: np.ndarray
    distance_matrix: np.ndarray

    def __post_init__(self):
        """Validate the network."""
        if len(self.methyls) != len(self.labels):
            raise ValueError("Number of methyls must match number of labels")
        if self.graph.number_of_nodes() != len(self.methyls):
            raise ValueError("Graph nodes must match number of methyls")

    def get_neighbors(self, node_id: int, max_distance: Optional[float] = None) -> List[int]:
        """
        Get neighboring nodes of a given node.

        Args:
            node_id: Node ID
            max_distance: Maximum distance threshold (if None, use all neighbors)

        Returns:
            List of neighbor node IDs
        """
        neighbors = list(self.graph.neighbors(node_id))

        if max_distance is not None:
            # Filter by distance
            filtered = []
            for neighbor in neighbors:
                edge_data = self.graph[node_id][neighbor]
                if edge_data.get('distance', float('inf')) <= max_distance:
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


class MethylNetworkBuilder:
    """
    Builder for creating methyl network graphs from protein structures.

    This builder creates a graph where:
    - Nodes represent methyl groups
    - Edges connect methyls within a distance threshold
    - Node features include coordinates, residue types, etc.
    - Edge features include distances and weights

    Example:
        >>> from methyl_match.reading import PDBParser
        >>> parser = PDBParser("protein.pdb")
        >>> methyls = parser.extract_methyls()
        >>> builder = MethylNetworkBuilder(distance_cutoff=12.0)
        >>> network = builder.build_network(methyls)
        >>> print(f"Nodes: {network.graph.number_of_nodes()}")
        >>> print(f"Edges: {network.graph.number_of_edges()}")
    """

    # Mapping of residue names to integer indices for one-hot encoding
    RESIDUE_TYPES = {
        'LEU': 0,
        'VAL': 1,
        'ILE': 2,
        'ALA': 3,
        'THR': 4,
        'MET': 5,
    }

    def __init__(
        self,
        distance_cutoff: float = 12.0,
        edge_weight_function: str = 'inverse',
        include_self_loops: bool = False
    ):
        """
        Initialize the methyl network builder.

        Args:
            distance_cutoff: Maximum distance (Å) for edge creation
            edge_weight_function: Function for computing edge weights
                ('inverse', 'exponential', 'uniform')
            include_self_loops: Whether to include self-loops in the graph
        """
        self.distance_cutoff = distance_cutoff
        self.edge_weight_function = edge_weight_function
        self.include_self_loops = include_self_loops
        logger.info(f"Initialized MethylNetworkBuilder with cutoff={distance_cutoff}Å")

    def build_network(self, methyls: List[MethylGroup]) -> MethylNetwork:
        """
        Build a methyl network from a list of methyl groups.

        Args:
            methyls: List of MethylGroup objects

        Returns:
            MethylNetwork object

        Example:
            >>> builder = MethylNetworkBuilder(distance_cutoff=10.0)
            >>> network = builder.build_network(methyls)
            >>> print(f"Network density: {nx.density(network.graph):.3f}")
        """
        if not methyls:
            raise ValueError("Cannot build network from empty methyl list")

        logger.info(f"Building network from {len(methyls)} methyls")

        # Create graph
        graph = nx.Graph()

        # Add nodes with features
        labels = []
        node_features = {}

        for i, methyl in enumerate(methyls):
            graph.add_node(i, label=methyl.label)
            labels.append(methyl.label)
            node_features[i] = self._compute_node_features(methyl, i, len(methyls))

        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(methyls)

        # Add edges based on distance cutoff
        edge_features = {}
        edge_count = 0

        for i in range(len(methyls)):
            for j in range(i + 1, len(methyls)):
                distance = distance_matrix[i, j]

                if distance <= self.distance_cutoff:
                    weight = self._compute_edge_weight(distance)
                    graph.add_edge(i, j, distance=distance, weight=weight)
                    edge_features[(i, j)] = self._compute_edge_features(
                        methyls[i], methyls[j], distance
                    )
                    edge_features[(j, i)] = edge_features[(i, j)]  # Symmetric
                    edge_count += 1

        # Add self-loops if requested
        if self.include_self_loops:
            for i in range(len(methyls)):
                graph.add_edge(i, i, distance=0.0, weight=1.0)
                edge_features[(i, i)] = np.array([0.0, 1.0])

        # Create adjacency matrix
        adjacency_matrix = nx.to_numpy_array(graph)

        logger.info(f"Built network: {len(methyls)} nodes, {edge_count} edges")
        logger.info(f"Network density: {nx.density(graph):.3f}")

        return MethylNetwork(
            graph=graph,
            methyls=methyls,
            labels=labels,
            node_features=node_features,
            edge_features=edge_features,
            adjacency_matrix=adjacency_matrix,
            distance_matrix=distance_matrix
        )

    def _compute_node_features(
        self,
        methyl: MethylGroup,
        node_id: int,
        total_nodes: int
    ) -> np.ndarray:
        """
        Compute feature vector for a node.

        Features include:
        - 3D coordinates (x, y, z)
        - One-hot encoded residue type (6 dimensions)
        - Normalized residue number
        - Node degree (computed later, initially 0)

        Args:
            methyl: MethylGroup object
            node_id: Node index
            total_nodes: Total number of nodes

        Returns:
            Feature vector (11 dimensions)
        """
        # Coordinates (3D)
        coords = methyl.coordinates

        # One-hot residue type (6D)
        residue_onehot = np.zeros(len(self.RESIDUE_TYPES))
        residue_idx = self.RESIDUE_TYPES.get(methyl.residue_name, -1)
        if residue_idx >= 0:
            residue_onehot[residue_idx] = 1.0

        # Normalized residue number
        normalized_res_num = methyl.residue_number / 100.0  # Approximate normalization

        # Combine features
        features = np.concatenate([
            coords,
            residue_onehot,
            [normalized_res_num]
        ])

        return features.astype(np.float32)

    def _compute_edge_features(
        self,
        methyl1: MethylGroup,
        methyl2: MethylGroup,
        distance: float
    ) -> np.ndarray:
        """
        Compute feature vector for an edge.

        Features include:
        - Distance
        - Edge weight

        Args:
            methyl1: First methyl group
            methyl2: Second methyl group
            distance: Distance between methyls

        Returns:
            Feature vector (2 dimensions)
        """
        weight = self._compute_edge_weight(distance)
        return np.array([distance, weight], dtype=np.float32)

    def _compute_distance_matrix(self, methyls: List[MethylGroup]) -> np.ndarray:
        """
        Compute pairwise distance matrix for methyls.

        Args:
            methyls: List of MethylGroup objects

        Returns:
            Distance matrix (n x n)
        """
        n = len(methyls)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(
                    methyls[i].coordinates - methyls[j].coordinates
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def _compute_edge_weight(self, distance: float) -> float:
        """
        Compute edge weight from distance.

        Args:
            distance: Distance between nodes (Å)

        Returns:
            Edge weight
        """
        if self.edge_weight_function == 'uniform':
            return 1.0
        elif self.edge_weight_function == 'inverse':
            return 1.0 / max(distance, 0.1)  # Avoid division by zero
        elif self.edge_weight_function == 'exponential':
            # Exponential decay with characteristic length = cutoff/3
            return np.exp(-distance / (self.distance_cutoff / 3.0))
        else:
            logger.warning(f"Unknown weight function '{self.edge_weight_function}', using uniform")
            return 1.0

    def get_statistics(self, network: MethylNetwork) -> Dict[str, float]:
        """
        Compute network statistics.

        Args:
            network: MethylNetwork object

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

        # Distance statistics
        distances = [data['distance'] for u, v, data in graph.edges(data=True)]
        if distances:
            stats['avg_distance'] = np.mean(distances)
            stats['min_distance'] = np.min(distances)
            stats['max_distance'] = np.max(distances)

        logger.info(f"Network statistics: {stats}")
        return stats

    def visualize_network(self, network: MethylNetwork, save_path: Optional[str] = None):
        """
        Create a 2D visualization of the network.

        Args:
            network: MethylNetwork object
            save_path: Optional path to save the figure

        Note:
            Requires matplotlib. For 3D visualization, use the marimo notebook.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib required for visualization")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Use spring layout for 2D positioning
        pos = nx.spring_layout(network.graph, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            network.graph, pos,
            node_color='lightblue',
            node_size=300,
            ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(
            network.graph, pos,
            alpha=0.3,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            network.graph, pos,
            labels={i: network.labels[i] for i in range(len(network.labels))},
            font_size=8,
            ax=ax
        )

        ax.set_title(f"Methyl Network (cutoff={self.distance_cutoff}Å)")
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")

        plt.tight_layout()
        return fig
