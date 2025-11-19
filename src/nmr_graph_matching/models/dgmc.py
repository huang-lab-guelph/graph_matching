"""Deep Graph Matching Consensus (DGMC) model for NMR methyl assignment.

Based on "Deep Graph Matching Consensus" (Fey et al., ICLR 2020)
Adapted for bipartite matching between peak network and methyl network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from typing import Optional, Tuple, List


class GraphEncoder(nn.Module):
    """Graph Neural Network encoder for node embeddings.

    Uses stacked GNN layers (GCN or GAT) to compute node embeddings
    that capture both local features and graph structure.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """Initialize the graph encoder.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn' or 'gat')
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        # Input layer
        if gnn_type == "gcn":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            hidden_dim = hidden_dim * 4  # Account for multi-head attention
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        if gnn_type == "gcn":
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Apply batch norm (except last layer)
            if self.use_batch_norm and i < self.num_layers - 1:
                x = self.batch_norms[i](x)

            # Apply activation and dropout (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class ConsensusGNN(nn.Module):
    """Consensus GNN for iterative refinement of matching scores.

    This module refines initial matching scores by propagating information
    about neighborhood consistency across the graph.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize the consensus GNN.

        Args:
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers
            num_layers: Number of refinement layers
            dropout: Dropout probability
        """
        super(ConsensusGNN, self).__init__()

        self.num_layers = num_layers

        # MLP for processing concatenated features
        self.mlps = nn.ModuleList()
        for i in range(num_layers):
            input_dim = embedding_dim * 2 if i == 0 else hidden_dim
            self.mlps.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        x_s: torch.Tensor,
        x_t: torch.Tensor,
        similarity_matrix: torch.Tensor,
        edge_index_s: torch.Tensor,
        edge_index_t: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for consensus refinement.

        Args:
            x_s: Source graph node embeddings [num_nodes_s, embedding_dim]
            x_t: Target graph node embeddings [num_nodes_t, embedding_dim]
            similarity_matrix: Initial similarity scores [num_nodes_s, num_nodes_t]
            edge_index_s: Source graph edges [2, num_edges_s]
            edge_index_t: Target graph edges [2, num_edges_t]

        Returns:
            Refined similarity matrix [num_nodes_s, num_nodes_t]
        """
        # Normalize similarity matrix (soft assignments)
        S = F.softmax(similarity_matrix, dim=-1)

        # Iterative refinement
        for i in range(self.num_layers):
            # Aggregate target features based on current matching
            x_t_agg = torch.mm(S, x_t)  # [num_nodes_s, embedding_dim]

            # Concatenate with source features
            x_concat = torch.cat([x_s, x_t_agg], dim=-1)  # [num_nodes_s, embedding_dim * 2]

            # Apply MLP
            x_update = self.mlps[i](x_concat)  # [num_nodes_s, hidden_dim]

            # Update similarity based on refined features
            if i < self.num_layers - 1:
                x_s = x_s + self.output_proj(x_update)
            else:
                x_s = self.output_proj(x_update)

        # Compute final similarity
        similarity_matrix = torch.mm(x_s, x_t.t())

        return similarity_matrix


class DGMCModel(nn.Module):
    """Deep Graph Matching Consensus model for NMR methyl assignment.

    This model learns to match nodes between two graphs:
    - Peak network (Graph A): Experimental NMR peaks with NOE connections
    - Methyl network (Graph B): Structural methyl positions with distance edges

    Architecture:
    1. Graph encoders: Compute node embeddings for both graphs
    2. Initial similarity: Cross-graph similarity computation
    3. Consensus refinement: Iterative refinement using neighborhood consensus
    4. Sinkhorn normalization: Convert to doubly-stochastic matrix
    """

    def __init__(
        self,
        peak_feature_dim: int,
        methyl_feature_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_consensus_layers: int = 2,
        gnn_type: str = "gcn",
        dropout: float = 0.1,
        sinkhorn_iterations: int = 20,
        sinkhorn_tau: float = 0.05
    ):
        """Initialize the DGMC model.

        Args:
            peak_feature_dim: Dimension of peak node features
            methyl_feature_dim: Dimension of methyl node features
            embedding_dim: Dimension of node embeddings
            hidden_dim: Dimension of hidden layers
            num_encoder_layers: Number of GNN encoder layers
            num_consensus_layers: Number of consensus refinement layers
            gnn_type: Type of GNN ('gcn' or 'gat')
            dropout: Dropout probability
            sinkhorn_iterations: Number of Sinkhorn iterations
            sinkhorn_tau: Temperature parameter for Sinkhorn
        """
        super(DGMCModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_tau = sinkhorn_tau

        # Encoder for peak network (Graph A)
        self.peak_encoder = GraphEncoder(
            input_dim=peak_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_encoder_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )

        # Encoder for methyl network (Graph B)
        self.methyl_encoder = GraphEncoder(
            input_dim=methyl_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_encoder_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )

        # Consensus GNN for refinement
        self.consensus_gnn = ConsensusGNN(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_consensus_layers,
            dropout=dropout
        )

    def forward(
        self,
        data_peak: Data,
        data_methyl: Data,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the model.

        Args:
            data_peak: Peak network data
            data_methyl: Methyl network data
            return_embeddings: Whether to return node embeddings

        Returns:
            Tuple of:
                - Matching matrix [num_peaks, num_methyls]
                - Optional: (peak_embeddings, methyl_embeddings)
        """
        # Encode both graphs
        peak_embeddings = self.peak_encoder(data_peak.x, data_peak.edge_index)
        methyl_embeddings = self.methyl_encoder(data_methyl.x, data_methyl.edge_index)

        # Compute initial similarity matrix
        similarity = torch.mm(peak_embeddings, methyl_embeddings.t())

        # Refine with consensus GNN
        similarity = self.consensus_gnn(
            x_s=peak_embeddings,
            x_t=methyl_embeddings,
            similarity_matrix=similarity,
            edge_index_s=data_peak.edge_index,
            edge_index_t=data_methyl.edge_index
        )

        # Apply Sinkhorn normalization
        matching_matrix = self.sinkhorn(similarity)

        if return_embeddings:
            return matching_matrix, (peak_embeddings, methyl_embeddings)
        else:
            return matching_matrix, None

    def sinkhorn(self, log_alpha: torch.Tensor, n_iter: Optional[int] = None) -> torch.Tensor:
        """Sinkhorn algorithm for doubly-stochastic normalization.

        Args:
            log_alpha: Log-space similarity matrix [n, m]
            n_iter: Number of iterations (default: self.sinkhorn_iterations)

        Returns:
            Doubly-stochastic matrix [n, m]
        """
        if n_iter is None:
            n_iter = self.sinkhorn_iterations

        # Apply temperature
        log_alpha = log_alpha / self.sinkhorn_tau

        # Iterative normalization
        for _ in range(n_iter):
            # Normalize rows
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            # Normalize columns
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)

        return torch.exp(log_alpha)

    def predict_assignments(
        self,
        matching_matrix: torch.Tensor,
        method: str = "greedy"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict discrete assignments from matching matrix.

        Args:
            matching_matrix: Soft matching matrix [num_peaks, num_methyls]
            method: Assignment method ('greedy' or 'hungarian')

        Returns:
            Tuple of:
                - assignments: Peak to methyl assignments [num_peaks]
                - confidences: Confidence scores [num_peaks]
        """
        if method == "greedy":
            # Greedy: assign each peak to highest-scoring methyl
            confidences, assignments = torch.max(matching_matrix, dim=1)
        elif method == "hungarian":
            # Hungarian algorithm for optimal assignment
            from scipy.optimize import linear_sum_assignment
            cost_matrix = -matching_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assignments = torch.zeros(matching_matrix.size(0), dtype=torch.long)
            confidences = torch.zeros(matching_matrix.size(0))

            assignments[row_ind] = torch.tensor(col_ind, dtype=torch.long)
            confidences[row_ind] = matching_matrix[row_ind, col_ind].detach()
        else:
            raise ValueError(f"Unknown method: {method}")

        return assignments, confidences
