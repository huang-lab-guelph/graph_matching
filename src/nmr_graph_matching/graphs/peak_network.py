"""Build peak network (Graph A) from NMR experimental data."""

from typing import List, Optional, Dict
import numpy as np
import torch
from torch_geometric.data import Data

from ..data.nmr_parser import Peak, NOECrosspeaks


class PeakNetworkBuilder:
    """Constructs a graph from NMR peak data.

    This creates Graph A in the methyl assignment problem, where:
    - Nodes: Individual methyl peaks from HMQC spectrum
    - Edges: NOE correlations from NOESY data
    - Node features: Chemical shifts, intensities
    - Edge features: NOE intensity, confidence scores
    """

    def __init__(
        self,
        shift_normalization: str = "standard",
        confidence_threshold: float = 0.0
    ):
        """Initialize the peak network builder.

        Args:
            shift_normalization: Method for normalizing chemical shifts
                                ('standard', 'minmax', 'none')
            confidence_threshold: Minimum confidence for including edges
        """
        self.shift_normalization = shift_normalization
        self.confidence_threshold = confidence_threshold

    def build(
        self,
        peaks: List[Peak],
        crosspeaks: List[NOECrosspeaks]
    ) -> Data:
        """Build peak network from HMQC peaks and NOESY crosspeaks.

        Args:
            peaks: List of Peak objects from HMQC
            crosspeaks: List of NOECrosspeaks from NOESY

        Returns:
            PyTorch Geometric Data object with:
                - x: Node features [num_nodes, num_features]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, num_edge_features]
                - peak_ids: List of peak identifiers
        """
        if len(peaks) == 0:
            return self._empty_graph()

        # Create peak ID to index mapping
        peak_id_to_idx = {peak.peak_id: idx for idx, peak in enumerate(peaks)}

        # Compute node features
        node_features = self._compute_node_features(peaks)

        # Build edges from crosspeaks
        edge_index, edge_attr = self._build_edges(crosspeaks, peak_id_to_idx)

        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
        )

        # Store metadata
        data.peak_ids = [peak.peak_id for peak in peaks]
        data.num_nodes = len(peaks)
        data.assignments = [peak.assignment for peak in peaks]

        return data

    def _compute_node_features(self, peaks: List[Peak]) -> np.ndarray:
        """Compute node features for each peak.

        Features include:
        - H chemical shift (normalized)
        - C chemical shift (normalized)
        - Peak intensity (log-scaled and normalized)
        - Peak confidence

        Args:
            peaks: List of Peak objects

        Returns:
            Array of shape [num_nodes, num_features]
        """
        num_peaks = len(peaks)

        # Extract raw features
        h_shifts = np.array([p.shifts[0] for p in peaks])
        c_shifts = np.array([p.shifts[1] for p in peaks])
        intensities = np.array([p.intensity for p in peaks])
        confidences = np.array([p.confidence for p in peaks])

        # Log-scale intensities (adding small constant to avoid log(0))
        log_intensities = np.log10(intensities + 1e-6)

        # Normalize features
        h_shifts_norm = self._normalize(h_shifts)
        c_shifts_norm = self._normalize(c_shifts)
        intensities_norm = self._normalize(log_intensities)

        # Combine features: [H_shift, C_shift, intensity, confidence]
        features = np.column_stack([
            h_shifts_norm,
            c_shifts_norm,
            intensities_norm,
            confidences
        ])

        return features.astype(np.float32)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values based on the specified method.

        Args:
            values: Array of values to normalize

        Returns:
            Normalized array
        """
        if self.shift_normalization == "standard":
            # Z-score normalization
            mean = values.mean()
            std = values.std()
            if std > 0:
                return (values - mean) / std
            else:
                return values - mean

        elif self.shift_normalization == "minmax":
            # Min-max normalization to [0, 1]
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                return (values - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(values)

        elif self.shift_normalization == "none":
            return values

        else:
            raise ValueError(f"Unknown normalization method: {self.shift_normalization}")

    def _build_edges(
        self,
        crosspeaks: List[NOECrosspeaks],
        peak_id_to_idx: Dict[int, int]
    ) -> tuple:
        """Build edge list and edge attributes from NOE crosspeaks.

        Args:
            crosspeaks: List of NOECrosspeaks objects
            peak_id_to_idx: Mapping from peak ID to node index

        Returns:
            Tuple of:
                - edge_index: [2, num_edges] array of node indices
                - edge_attr: [num_edges, num_edge_features] array of edge features
        """
        edge_list = []
        edge_features = []

        # Collect all NOE intensities for normalization
        all_intensities = [cp.noe_intensity for cp in crosspeaks]
        if all_intensities:
            log_intensities = np.log10(np.array(all_intensities) + 1e-6)
            norm_intensities = self._normalize(log_intensities)
        else:
            norm_intensities = []

        for idx, crosspeak in enumerate(crosspeaks):
            # Skip low-confidence crosspeaks
            if crosspeak.confidence < self.confidence_threshold:
                continue

            # Get node indices
            peak1_idx = peak_id_to_idx.get(crosspeak.peak1_id)
            peak2_idx = peak_id_to_idx.get(crosspeak.peak2_id)

            # Skip if either peak is not in the graph
            if peak1_idx is None or peak2_idx is None:
                continue

            # Add edges in both directions (undirected graph)
            edge_list.append([peak1_idx, peak2_idx])
            edge_list.append([peak2_idx, peak1_idx])

            # Edge features: [normalized_intensity, confidence]
            edge_feat = [norm_intensities[idx], crosspeak.confidence]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

        if len(edge_list) == 0:
            # No edges - create empty tensors
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
            x=torch.FloatTensor(np.zeros((0, 4))),  # 4 node features
            edge_index=torch.LongTensor(np.zeros((2, 0), dtype=np.int64)),
            edge_attr=torch.FloatTensor(np.zeros((0, 2))),
            peak_ids=[],
            num_nodes=0,
            assignments=[]
        )

    def add_confidence_scoring(
        self,
        crosspeaks: List[NOECrosspeaks],
        overlap_matrix: Optional[np.ndarray] = None,
        donor_multiplicity: Optional[Dict[int, int]] = None
    ) -> List[NOECrosspeaks]:
        """Enhance confidence scores based on peak quality factors.

        This implements confidence scoring similar to the MAGIC algorithm,
        accounting for:
        - Peak overlap (reduces confidence)
        - Donor multiplicity (multiple NOEs from same donor)
        - Shared NOE patterns

        Args:
            crosspeaks: List of NOECrosspeaks
            overlap_matrix: Matrix indicating peak overlap [num_peaks, num_peaks]
            donor_multiplicity: Dictionary mapping peak_id to number of NOEs

        Returns:
            Updated list of NOECrosspeaks with adjusted confidence scores
        """
        updated_crosspeaks = []

        for cp in crosspeaks:
            confidence = cp.confidence

            # Penalty for peak overlap
            if overlap_matrix is not None:
                peak1_idx = cp.peak1_id - 1
                peak2_idx = cp.peak2_id - 1

                if peak1_idx < overlap_matrix.shape[0] and peak2_idx < overlap_matrix.shape[1]:
                    overlap_penalty = overlap_matrix[peak1_idx, peak2_idx]
                    confidence *= (1.0 - overlap_penalty)

            # Penalty for high donor multiplicity
            if donor_multiplicity is not None:
                mult1 = donor_multiplicity.get(cp.peak1_id, 1)
                mult2 = donor_multiplicity.get(cp.peak2_id, 1)

                # Higher multiplicity reduces confidence
                mult_factor = 1.0 / np.sqrt(max(mult1, mult2))
                confidence *= mult_factor

            updated_cp = NOECrosspeaks(
                peak1_id=cp.peak1_id,
                peak2_id=cp.peak2_id,
                noe_intensity=cp.noe_intensity,
                confidence=min(1.0, max(0.0, confidence))  # Clamp to [0, 1]
            )
            updated_crosspeaks.append(updated_cp)

        return updated_crosspeaks

    def compute_peak_overlap(self, peaks: List[Peak], tolerance: float = 0.05) -> np.ndarray:
        """Compute overlap matrix for peaks based on chemical shift proximity.

        Args:
            peaks: List of Peak objects
            tolerance: Chemical shift tolerance for overlap (ppm)

        Returns:
            Overlap matrix [num_peaks, num_peaks] with values in [0, 1]
        """
        num_peaks = len(peaks)
        overlap_matrix = np.zeros((num_peaks, num_peaks))

        for i in range(num_peaks):
            for j in range(i + 1, num_peaks):
                h_diff = abs(peaks[i].shifts[0] - peaks[j].shifts[0])
                c_diff = abs(peaks[i].shifts[1] - peaks[j].shifts[1])

                # Compute overlap based on proximity
                h_overlap = max(0, 1 - h_diff / tolerance)
                c_overlap = max(0, 1 - c_diff / tolerance)

                # Overall overlap is product (both dimensions must be close)
                overlap = h_overlap * c_overlap

                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap

        return overlap_matrix

    def get_adjacency_matrix(self, data: Data) -> np.ndarray:
        """Extract adjacency matrix from graph data.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Adjacency matrix [num_nodes, num_nodes]
        """
        num_nodes = data.num_nodes
        adj_matrix = np.zeros((num_nodes, num_nodes))

        edge_index = data.edge_index.numpy()
        edge_weights = data.edge_attr[:, 0].numpy()  # First column is NOE intensity

        for idx, (i, j) in enumerate(edge_index.T):
            adj_matrix[i, j] = edge_weights[idx]

        return adj_matrix

    def get_density_matrix(self, data: Data, power: int = 2) -> np.ndarray:
        """Compute density matrix (P^n) for graph.

        The density matrix captures higher-order connectivity patterns
        and is used in the MAGIC algorithm.

        Args:
            data: PyTorch Geometric Data object
            power: Power to raise adjacency matrix to

        Returns:
            Density matrix [num_nodes, num_nodes]
        """
        adj_matrix = self.get_adjacency_matrix(data)

        # Matrix power
        density_matrix = np.linalg.matrix_power(adj_matrix, power)

        # Normalize
        max_val = density_matrix.max()
        if max_val > 0:
            density_matrix = density_matrix / max_val

        return density_matrix

    def visualize_network(
        self,
        data: Data,
        output_file: Optional[str] = None,
        layout: str = "spring"
    ):
        """Create a visualization of the peak network.

        Args:
            data: PyTorch Geometric Data object
            output_file: Optional path to save the figure
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("matplotlib and networkx required for visualization")
            return

        # Convert to NetworkX graph
        G = nx.Graph()

        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)

        # Add edges
        edge_index = data.edge_index.numpy()
        edge_weights = data.edge_attr[:, 0].numpy()

        for idx in range(0, edge_index.shape[1], 2):  # Skip duplicates
            i = edge_index[0, idx]
            j = edge_index[1, idx]
            weight = edge_weights[idx]
            G.add_edge(i, j, weight=weight)

        # Compute layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Draw graph
        plt.figure(figsize=(12, 10))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.8)

        # Draw edges with varying width based on NOE intensity
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        widths = [3 * w / max_weight for w in weights]

        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5)

        # Draw labels
        labels = {i: str(data.peak_ids[i]) for i in range(data.num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title(f'Peak Network ({data.num_nodes} peaks, {G.number_of_edges()} NOE correlations)')
        plt.axis('off')

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()
