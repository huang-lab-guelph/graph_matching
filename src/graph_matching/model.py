"""
Module for the Deep Learning Graph Matching Model, implementing a
Graph Matching Network (GMN) architecture for similarity learning, and
using pygmtools for the final assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pygmtools as pygm
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Dict, Any, List, Tuple

# Set the pygmtools backend to pytorch
pygm.set_backend('pytorch')


class GNNEncoder(torch.nn.Module):
    """
    A simple Graph Neural Network (GNN) encoder to learn node embeddings.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))
        self.conv_layers.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = torch.relu(x)
        return x

class GraphMatchingModel(torch.nn.Module):
    """
    Graph Matching Model that uses a GNN for similarity learning and pygmtools for assignment.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.gnn_encoder = GNNEncoder(in_channels, hidden_channels, out_channels, num_layers)
        self.similarity_fc = nn.Sequential(
            nn.Linear(out_channels * 2, 64), # Input size is 2 * out_channels
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data1: Data, data2: Data) -> torch.Tensor:
        """
        Forward pass for similarity learning.
        """
        embeddings1 = self.gnn_encoder(data1.x, data1.edge_index)
        embeddings2 = self.gnn_encoder(data2.x, data2.edge_index)
        
        # Aggregate embeddings to graph level
        graph_embedding1 = global_mean_pool(embeddings1, torch.zeros(embeddings1.shape[0], dtype=torch.long, device=embeddings1.device))
        graph_embedding2 = global_mean_pool(embeddings2, torch.zeros(embeddings2.shape[0], dtype=torch.long, device=embeddings2.device))
        
        # Concatenate graph embeddings
        combined_embedding = torch.cat([graph_embedding1, graph_embedding2], dim=1)
        
        similarity_score = self.similarity_fc(combined_embedding)
        
        return torch.sigmoid(similarity_score)

    def get_assignments(self, data1: Data, data2: Data) -> Dict[int, int]:
        """
        Get one-to-one peak assignments using the trained GNN and the Hungarian algorithm.
        
        Returns:
            A dictionary mapping node indices from graph1 to graph2.
        """
        self.eval()
        with torch.no_grad():
            num_nodes1 = data1.num_nodes if hasattr(data1, 'num_nodes') else data1.x.shape[0]
            num_nodes2 = data2.num_nodes if hasattr(data2, 'num_nodes') else data2.x.shape[0]

            if num_nodes1 == 0 or num_nodes2 == 0:
                return {}

            # Get GNN-learned node embeddings
            embeddings1 = self.gnn_encoder(data1.x, data1.edge_index)
            embeddings2 = self.gnn_encoder(data2.x, data2.edge_index)
            
            # Normalize embeddings for cosine similarity
            embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
            embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)

            # Compute cosine similarity matrix
            similarity_matrix = torch.matmul(embeddings1_norm, embeddings2_norm.t())
            
            # Use the Hungarian algorithm to find the optimal one-to-one assignment
            # The hungarian solver finds the assignment that maximizes the total similarity.
            X_discrete = pygm.hungarian(similarity_matrix.unsqueeze(0)).squeeze(0)

            # Convert permutation matrix to assignment dictionary
            assignments = {
                i: j.item() for i, j in enumerate(X_discrete.argmax(dim=1))
                if X_discrete[i,j] > 0 # Only include actual assignments
            }
            
            return assignments


if __name__ == '__main__':
    # Example usage
    
    # Create two graphs
    x1 = torch.randn(5, 2)
    edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    data1 = Data(x=x1, edge_index=edge_index1)

    x2 = x1 + torch.randn_like(x1) * 0.1 # Perturbed features
    edge_index2 = torch.tensor([[0, 1, 1, 3, 3, 4], [1, 0, 3, 1, 4, 3]], dtype=torch.long)
    data2 = Data(x=x2, edge_index=edge_index2)
    
    model = GraphMatchingModel(in_channels=2, hidden_channels=16, out_channels=16)
    
    # Get similarity score
    model.train()
    similarity = model(data1, data2)
    print(f"Similarity between graph 1 and graph 2: {similarity.item():.4f}")
    
    # Get assignments
    assignments = model.get_assignments(data1, data2)
    print("\nAssignments (graph1_node -> graph2_node):")
    print(assignments)