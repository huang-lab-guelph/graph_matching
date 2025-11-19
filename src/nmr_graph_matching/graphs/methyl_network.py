"""Build methyl network (Graph B) from protein structure."""

from typing import Optional, List, Callable
import numpy as np
import torch
from torch_geometric.data import Data

from ..data.pdb_parser import PDBParser, MethylGroup


class MethylNetworkBuilder:
    """Constructs a graph from methyl groups in a protein structure.

    This creates Graph B in the methyl assignment problem, where:
    - Nodes: Methyl carbon positions
    - Edges: Distance-based connectivity (typically 7-10 Å cutoff)
    - Node features: Coordinates, residue type
    - Edge features: Distance, edge weight
    """

    # One-hot encoding for residue types
    RESIDUE_TYPES = ["LEU", "VAL", "ILE", "ALA", "THR", "MET"]

    def __init__(
        self,
        distance_cutoff: float = 10.0,
        min_distance: float = 0.0,
        edge_weight_function: Optional[Callable] = None
    ):
        """Initialize the methyl network builder.

        Args:
            distance_cutoff: Maximum distance for edge creation (Angstroms)
            min_distance: Minimum distance for edge creation (Angstroms)
            edge_weight_function: Function to compute edge weights from distances
                                 If None, uses default declining function
        """
        self.distance_cutoff = distance_cutoff
        self.min_distance = min_distance

        if edge_weight_function is None:
            self.edge_weight_function = self._default_edge_weight
        else:
            self.edge_weight_function = edge_weight_function

    def _default_edge_weight(self, distance: float) -> float:
        """Default edge weight function.

        Declines smoothly from 1.0 at contact distance to 0.0 at cutoff.
        Uses a sigmoid-like function for smooth transition.

        Args:
            distance: Distance in Angstroms

        Returns:
            Edge weight between 0 and 1
        """
        if distance <= self.min_distance:
            return 1.0
        if distance >= self.distance_cutoff:
            return 0.0

        # Smooth declining function
        normalized_dist = (distance - self.min_distance) / (self.distance_cutoff - self.min_distance)

        # Exponential decay
        weight = np.exp(-3 * normalized_dist)

        return float(weight)

    def build_from_pdb(self, pdb_file: str, structure_id: str = "structure") -> Data:
        """Build methyl network directly from a PDB file.

        Args:
            pdb_file: Path to PDB file
            structure_id: Identifier for the structure

        Returns:
            PyTorch Geometric Data object
        """
        parser = PDBParser()
        methyl_groups = parser.parse(pdb_file, structure_id)

        return self.build(methyl_groups)

    def build(self, methyl_groups: List[MethylGroup]) -> Data:
        """Build methyl network from a list of methyl groups.

        Args:
            methyl_groups: List of MethylGroup objects

        Returns:
            PyTorch Geometric Data object with:
                - x: Node features [num_nodes, num_features]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, num_edge_features]
                - pos: 3D coordinates [num_nodes, 3]
                - methyl_names: List of methyl identifiers
        """
        if len(methyl_groups) == 0:
            return self._empty_graph()

        # Extract node features
        node_features = self._compute_node_features(methyl_groups)
        coordinates = np.array([m.coordinates for m in methyl_groups])

        # Compute distance matrix
        distance_matrix = self._compute_distance_matrix(coordinates)

        # Build edge list and edge features
        edge_index, edge_attr = self._build_edges(distance_matrix)

        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            pos=torch.FloatTensor(coordinates),
        )

        # Store additional metadata
        data.methyl_names = [m.full_name for m in methyl_groups]
        data.num_nodes = len(methyl_groups)

        return data

    def _compute_node_features(self, methyl_groups: List[MethylGroup]) -> np.ndarray:
        """Compute node features for each methyl group.

        Features include:
        - 3D coordinates (x, y, z)
        - One-hot encoded residue type
        - Residue number (normalized)

        Args:
            methyl_groups: List of MethylGroup objects

        Returns:
            Array of shape [num_nodes, num_features]
        """
        num_nodes = len(methyl_groups)
        num_residue_types = len(self.RESIDUE_TYPES)

        # Features: [x, y, z, one_hot_residue_type, normalized_res_num]
        num_features = 3 + num_residue_types + 1
        features = np.zeros((num_nodes, num_features))

        # Normalize residue numbers
        res_numbers = np.array([m.residue_number for m in methyl_groups])
        min_res = res_numbers.min()
        max_res = res_numbers.max()
        if max_res > min_res:
            normalized_res = (res_numbers - min_res) / (max_res - min_res)
        else:
            normalized_res = np.zeros_like(res_numbers)

        for i, methyl in enumerate(methyl_groups):
            # Coordinates
            features[i, 0:3] = methyl.coordinates

            # One-hot residue type
            if methyl.residue_name in self.RESIDUE_TYPES:
                res_idx = self.RESIDUE_TYPES.index(methyl.residue_name)
                features[i, 3 + res_idx] = 1.0

            # Normalized residue number
            features[i, 3 + num_residue_types] = normalized_res[i]

        return features

    def _compute_distance_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances.

        Args:
            coordinates: Array of shape [num_nodes, 3]

        Returns:
            Distance matrix of shape [num_nodes, num_nodes]
        """
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances

    def _build_edges(self, distance_matrix: np.ndarray) -> tuple:
        """Build edge list and edge attributes from distance matrix.

        Args:
            distance_matrix: Symmetric distance matrix [num_nodes, num_nodes]

        Returns:
            Tuple of:
                - edge_index: [2, num_edges] array of node indices
                - edge_attr: [num_edges, num_edge_features] array of edge features
        """
        num_nodes = distance_matrix.shape[0]
        edge_list = []
        edge_features = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Only upper triangle
                distance = distance_matrix[i, j]

                if self.min_distance <= distance <= self.distance_cutoff:
                    weight = self.edge_weight_function(distance)

                    # Add edge in both directions (undirected graph)
                    edge_list.append([i, j])
                    edge_list.append([j, i])

                    # Edge features: [distance, weight]
                    edge_feat = [distance, weight]
                    edge_features.append(edge_feat)
                    edge_features.append(edge_feat)

        if len(edge_list) == 0:
            # No edges - create empty tensors with correct shape
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 2), dtype=np.float32)
        else:
            edge_index = np.array(edge_list).T
            edge_attr = np.array(edge_features)

        return edge_index, edge_attr

    def _empty_graph(self) -> Data:
        """Create an empty graph structure.

        Returns:
            Empty PyTorch Geometric Data object
        """
        return Data(
            x=torch.FloatTensor(np.zeros((0, 11))),  # Empty node features
            edge_index=torch.LongTensor(np.zeros((2, 0), dtype=np.int64)),
            edge_attr=torch.FloatTensor(np.zeros((0, 2))),
            pos=torch.FloatTensor(np.zeros((0, 3))),
            methyl_names=[],
            num_nodes=0
        )

    def get_adjacency_matrix(self, data: Data) -> np.ndarray:
        """Extract adjacency matrix from graph data.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Symmetric adjacency matrix [num_nodes, num_nodes]
        """
        num_nodes = data.num_nodes
        adj_matrix = np.zeros((num_nodes, num_nodes))

        edge_index = data.edge_index.numpy()
        edge_weights = data.edge_attr[:, 1].numpy()  # Second column is weight

        for idx, (i, j) in enumerate(edge_index.T):
            adj_matrix[i, j] = edge_weights[idx]

        return adj_matrix

    def visualize_network(self, data: Data, output_file: Optional[str] = None):
        """Create a visualization of the methyl network.

        Args:
            data: PyTorch Geometric Data object
            output_file: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("matplotlib not available for visualization")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot nodes
        pos = data.pos.numpy()
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', s=100, alpha=0.6)

        # Plot edges
        edge_index = data.edge_index.numpy()
        for i in range(0, edge_index.shape[1], 2):  # Skip duplicate edges
            start_idx = edge_index[0, i]
            end_idx = edge_index[1, i]

            xs = [pos[start_idx, 0], pos[end_idx, 0]]
            ys = [pos[start_idx, 1], pos[end_idx, 1]]
            zs = [pos[start_idx, 2], pos[end_idx, 2]]

            ax.plot(xs, ys, zs, 'gray', alpha=0.3, linewidth=0.5)

        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(f'Methyl Network ({data.num_nodes} nodes, {edge_index.shape[1]//2} edges)')

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()
