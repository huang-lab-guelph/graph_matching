"Module for the Deep Learning Graph Matching Model."

import torch
import pygmtools as pygm
import networkx as nx
from typing import Dict, Any, List, Tuple
import functools
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv # Or other GNN layers like GATConv

# Set the pygmtools backend to pytorch
pygm.set_backend('pytorch')

class GNNEncoder(torch.nn.Module):
    """
    A simple Graph Neural Network (GNN) encoder to learn node embeddings.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        # Input layer
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        # Output layer
        self.conv_layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = torch.relu(x)
                x = torch.dropout(x, p=0.5, train=self.training)
        return x

class GraphMatchingModel(torch.nn.Module):
    """
    A Deep Learning Graph Matching Model that uses a GNN to learn node embeddings
    and then applies pygmtools solvers.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        """
        Initializes the GraphMatchingModel with GNN encoders.

        Args:
            in_channels: Dimension of input node features (e.g., 2 for c_shift, h_shift).
            hidden_channels: Dimension of hidden layers in the GNN.
            out_channels: Dimension of the output node embeddings from the GNN.
            num_layers: Number of GCNConv layers in the encoder.
        """
        super().__init__()
        self.gnn_encoder = GNNEncoder(in_channels, hidden_channels, out_channels, num_layers)
        print("GraphMatchingModel initialized with GNN encoder and pygmtools solver.")

    def _build_affinity_matrix(
                               self, 
                               graph1: nx.Graph,
                               graph2: nx.Graph,
                               sigma: float = 1.0) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Builds the affinity matrix (K) between two graphs using GNN-learned node embeddings.

        Args:
            graph1: The first NetworkX graph.
            graph2: The second NetworkX graph.
            sigma: Standard deviation for the Gaussian affinity function.

        Returns:
            A tuple containing:
            - K: The affinity matrix.
            - (num_nodes1, num_nodes2): A tuple indicating the number of nodes in each graph.
        """
        num_nodes1 = graph1.number_of_nodes()
        num_nodes2 = graph2.number_of_nodes()

        # Handle empty graphs
        if num_nodes1 == 0 or num_nodes2 == 0:
            return torch.zeros((num_nodes1, num_nodes2)), (num_nodes1, num_nodes2)

        # Convert NetworkX graphs to PyTorch Geometric Data objects
        x1 = torch.tensor([[data['c_shift'], data['h_shift']] for _, data in graph1.nodes(data=True)], dtype=torch.float32)
        x2 = torch.tensor([[data['c_shift'], data['h_shift']] for _, data in graph2.nodes(data=True)], dtype=torch.float32)

        edge_index1 = torch.tensor(list(graph1.edges)).t().contiguous()
        if edge_index1.numel() == 0:
            edge_index1 = torch.empty((2, 0), dtype=torch.long)

        edge_index2 = torch.tensor(list(graph2.edges)).t().contiguous()
        if edge_index2.numel() == 0:
            edge_index2 = torch.empty((2, 0), dtype=torch.long)
            
        edge_attr1 = torch.tensor([graph1[u][v]['weight'] for u, v in graph1.edges()], dtype=torch.float32).unsqueeze(-1)
        edge_attr2 = torch.tensor([graph2[u][v]['distance'] for u, v in graph2.edges()], dtype=torch.float32).unsqueeze(-1)


        data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1)
        data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)

        # Get GNN-learned node embeddings
        embeddings1 = self.gnn_encoder(data1.x, data1.edge_index)
        embeddings2 = self.gnn_encoder(data2.x, data2.edge_index)
        
        # Prepare inputs for pygm.utils.build_aff_mat
        F1 = embeddings1.unsqueeze(0)
        F2 = embeddings2.unsqueeze(0)
        
        H1 = data1.edge_attr.unsqueeze(0)
        H2 = data2.edge_attr.unsqueeze(0)

        ne1 = torch.tensor([data1.edge_index.shape[1]])
        ne2 = torch.tensor([data2.edge_index.shape[1]])
        
        # Pass edge_index directly as connectivity
        G1_conn = edge_index1.t().unsqueeze(0) # Transpose and add batch dimension
        G2_conn = edge_index2.t().unsqueeze(0) # Transpose and add batch dimension
        
        # Define affinity functions
        node_aff_fn = functools.partial(pygm.utils.gaussian_aff_fn, sigma=sigma)
        edge_aff_fn = functools.partial(pygm.utils.gaussian_aff_fn, sigma=sigma)

        K = pygm.utils.build_aff_mat(F1, H1, G1_conn,
                                     F2, H2, G2_conn,
                                     torch.tensor([num_nodes1]), ne1,
                                     torch.tensor([num_nodes2]), ne2,
                                     node_aff_fn=node_aff_fn,
                                     edge_aff_fn=edge_aff_fn)
        
        if torch.isnan(K).any() or torch.isinf(K).any():
            print("Warning: NaN or Inf found in affinity matrix K.")
            K = torch.nan_to_num(K, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if K.shape[0] == 1:
            K = K.squeeze(0)

        return K, (num_nodes1, num_nodes2)

    def predict(
                self, 
                graph1: nx.Graph,
                graph2: nx.Graph) -> torch.Tensor:
        """
        Performs graph matching between two input NetworkX graphs using a GNN-enhanced
        affinity matrix and a pygmtools classic solver.

        Args:
            graph1: The first NetworkX graph.
            graph2: The second NetworkX graph.

        Returns:
            A torch.Tensor representing the discrete permutation matrix
            (assignment scores) between nodes of graph1 and graph2.
        """
        print("Performing graph matching using GNN-enhanced affinity and pygmtools classic solver.")

        # Build the affinity matrix using GNN embeddings
        K, (num_nodes1, num_nodes2) = self._build_affinity_matrix(graph1, graph2)

        if num_nodes1 == 0 or num_nodes2 == 0:
            return torch.zeros((num_nodes1, num_nodes2))

        # Solve the graph matching problem using RRWM
        K_batched = K.unsqueeze(0)
        n1_batched = torch.tensor([num_nodes1])
        n2_batched = torch.tensor([num_nodes2])

        X_soft = pygm.rrwm(K_batched, n1_batched, n2_batched, beta=100)
        X_soft = X_soft.squeeze(0)

        # Get the discrete matching matrix using the Hungarian algorithm
        X_discrete = pygm.hungarian(X_soft)

        return X_discrete

if __name__ == "__main__":
    # Example usage:
    G1 = nx.Graph()
    G1.add_node(0, c_shift=19.311, h_shift=1.394)
    G1.add_node(1, c_shift=24.873, h_shift=0.815)
    G1.add_node(2, c_shift=23.265, h_shift=0.793)
    G1.add_edge(0, 1, weight=100.0)
    G1.add_edge(1, 2, weight=120.0)

    G2 = nx.Graph()
    G2.add_node(0, c_shift=19.320, h_shift=1.390, distance=0.0)
    G2.add_node(1, c_shift=24.870, h_shift=0.820, distance=0.0)
    G2.add_node(2, c_shift=23.260, h_shift=0.790, distance=0.0)
    G2.add_node(3, c_shift=10.0, h_shift=2.0, distance=0.0)
    G2.add_edge(0, 1, distance=105.0)
    G2.add_edge(1, 2, distance=115.0)

    model = GraphMatchingModel(in_channels=2, hidden_channels=16, out_channels=16)

    try:
        prediction = model.predict(G1, G2)
        print("\nPrediction (Discrete Matching Matrix):\n", prediction)
        print(f"Prediction shape: {prediction.shape}")
    except ValueError as e:
        print(f"Error during prediction: {e}")