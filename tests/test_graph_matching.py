"""
Unit tests for the graph_matching.model module.
"""

import pytest
import torch
from src.graph_matching.model import GraphMatchingModel

@pytest.fixture
def dummy_graph_data():
    """Provides dummy graph data for testing."""
    num_nodes1 = 5
    num_nodes2 = 5
    feature_dim = 16

    features1 = torch.randn(num_nodes1, feature_dim)
    adj1 = torch.randint(0, 2, (num_nodes1, num_nodes1)).float()
    adj1 = adj1 - torch.diag_embed(torch.diag(adj1))
    adj1 = (adj1 + adj1.T) / 2

    features2 = torch.randn(num_nodes2, feature_dim)
    adj2 = torch.randint(0, 2, (num_nodes2, num_nodes2)).float()
    adj2 = adj2 - torch.diag_embed(torch.diag(adj2))
    adj2 = (adj2 + adj2.T) / 2

    graph1 = {'features': features1, 'adjacency_matrix': adj1}
    graph2 = {'features': features2, 'adjacency_matrix': adj2}
    return graph1, graph2, feature_dim

def test_model_initialization():
    """Test if the GraphMatchingModel initializes correctly."""
    model = GraphMatchingModel(feature_dim=16, hidden_dim=32, output_dim=16)
    assert isinstance(model, GraphMatchingModel)
    assert model.feature_dim == 16
    assert model.hidden_dim == 32
    assert model.output_dim == 16
    assert hasattr(model, 'gnn_layer')
    assert hasattr(model, 'output_layer')

def test_model_forward_pass(dummy_graph_data):
    """Test the forward pass of the model."""
    graph1, _, feature_dim = dummy_graph_data
    model = GraphMatchingModel(feature_dim=feature_dim, hidden_dim=32, output_dim=feature_dim)
    output = model.forward(graph1['features'], graph1['adjacency_matrix'])
    assert isinstance(output, torch.Tensor)
    assert output.shape == (graph1['features'].shape[0], feature_dim)

def test_model_predict_same_nodes(dummy_graph_data):
    """Test the predict method with graphs of the same number of nodes."""
    graph1, graph2, feature_dim = dummy_graph_data
    model = GraphMatchingModel(feature_dim=feature_dim, hidden_dim=32, output_dim=feature_dim)
    prediction = model.predict(graph1, graph2)
    assert isinstance(prediction, torch.Tensor)
    # The dummy predict returns an identity matrix if nodes are equal
    assert prediction.shape == (graph1['features'].shape[0], graph2['features'].shape[0])
    # assert torch.equal(prediction, torch.eye(graph1['features'].shape[0])) # This would only pass for exact identity

def test_model_predict_different_nodes():
    """Test the predict method with graphs of different numbers of nodes."""
    feature_dim = 16
    num_nodes1 = 5
    num_nodes2 = 4

    features1 = torch.randn(num_nodes1, feature_dim)
    adj1 = torch.randint(0, 2, (num_nodes1, num_nodes1)).float()
    adj1 = (adj1 + adj1.T) / 2

    features2 = torch.randn(num_nodes2, feature_dim)
    adj2 = torch.randint(0, 2, (num_nodes2, num_nodes2)).float()
    adj2 = (adj2 + adj2.T) / 2

    graph1 = {'features': features1, 'adjacency_matrix': adj1}
    graph2 = {'features': features2, 'adjacency_matrix': adj2}

    model = GraphMatchingModel(feature_dim=feature_dim, hidden_dim=32, output_dim=feature_dim)
    prediction = model.predict(graph1, graph2)
    assert isinstance(prediction, torch.Tensor)
    # The dummy predict returns a random affinity matrix if nodes are different
    assert prediction.shape == (num_nodes1, num_nodes2)

def test_model_predict_missing_data(dummy_graph_data):
    """Test the predict method with missing graph data."""
    model = GraphMatchingModel(feature_dim=16, hidden_dim=32, output_dim=16)
    graph1_incomplete = {'features': torch.randn(5, 16)}
    graph2_incomplete = {'adjacency_matrix': torch.randint(0, 2, (5, 5)).float()}

    # Unpack the dummy_graph_data fixture
    graph1_full, graph2_full, _ = dummy_graph_data

    with pytest.raises(ValueError, match="Graph data must contain 'features' and 'adjacency_matrix'."):
        model.predict(graph1_incomplete, graph2_full)
    with pytest.raises(ValueError, match="Graph data must contain 'features' and 'adjacency_matrix'."):
        model.predict(graph1_full, graph2_incomplete)
