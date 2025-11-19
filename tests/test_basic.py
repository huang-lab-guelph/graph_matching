"""Basic tests to verify installation and functionality."""

import pytest
import numpy as np
import torch
from pathlib import Path


def test_imports():
    """Test that all main modules can be imported."""
    from nmr_graph_matching import (
        PDBParser,
        HMQCParser,
        NOESYParser,
        MethylNetworkBuilder,
        PeakNetworkBuilder,
        DGMCModel,
        GraphEncoder,
        Trainer,
        NMRDataset,
        OutputFormatter,
        AssignmentResult
    )
    assert True


def test_pdb_parser():
    """Test PDB parser initialization."""
    from nmr_graph_matching.data.pdb_parser import PDBParser

    parser = PDBParser()
    assert parser is not None
    assert parser.METHYL_ATOMS['LEU'] == ['CD1', 'CD2']
    assert parser.METHYL_ATOMS['VAL'] == ['CG1', 'CG2']


def test_hmqc_parser():
    """Test HMQC parser initialization."""
    from nmr_graph_matching.data.nmr_parser import HMQCParser

    parser = HMQCParser()
    assert parser is not None
    assert parser.peaks == []


def test_methyl_network_builder():
    """Test methyl network builder."""
    from nmr_graph_matching.graphs.methyl_network import MethylNetworkBuilder
    from nmr_graph_matching.data.pdb_parser import MethylGroup

    builder = MethylNetworkBuilder(distance_cutoff=10.0)

    # Create dummy methyl groups
    methyls = [
        MethylGroup(
            residue_name='LEU',
            residue_number=15,
            chain_id='A',
            atom_name='CD1',
            coordinates=np.array([0.0, 0.0, 0.0]),
            full_name='L15CD1'
        ),
        MethylGroup(
            residue_name='VAL',
            residue_number=23,
            chain_id='A',
            atom_name='CG1',
            coordinates=np.array([5.0, 0.0, 0.0]),
            full_name='V23CG1'
        )
    ]

    graph = builder.build(methyls)
    assert graph.num_nodes == 2
    assert graph.x.size(0) == 2


def test_peak_network_builder():
    """Test peak network builder."""
    from nmr_graph_matching.graphs.peak_network import PeakNetworkBuilder
    from nmr_graph_matching.data.nmr_parser import Peak, NOECrosspeaks

    builder = PeakNetworkBuilder()

    # Create dummy peaks
    peaks = [
        Peak(peak_id=1, shifts=np.array([0.845, 24.3]), intensity=1e5),
        Peak(peak_id=2, shifts=np.array([1.234, 19.5]), intensity=8e4)
    ]

    # Create dummy crosspeaks
    crosspeaks = [
        NOECrosspeaks(peak1_id=1, peak2_id=2, noe_intensity=5e4, confidence=0.8)
    ]

    graph = builder.build(peaks, crosspeaks)
    assert graph.num_nodes == 2
    assert graph.x.size(0) == 2


def test_dgmc_model():
    """Test DGMC model initialization."""
    from nmr_graph_matching.models.dgmc import DGMCModel

    model = DGMCModel(
        peak_feature_dim=4,
        methyl_feature_dim=11,
        embedding_dim=32,
        hidden_dim=64,
        num_encoder_layers=2,
        num_consensus_layers=1
    )

    assert model is not None
    assert model.embedding_dim == 32


def test_model_forward_pass():
    """Test model forward pass with dummy data."""
    from nmr_graph_matching.models.dgmc import DGMCModel
    from torch_geometric.data import Data

    model = DGMCModel(
        peak_feature_dim=4,
        methyl_feature_dim=11,
        embedding_dim=32,
        hidden_dim=64,
        num_encoder_layers=2,
        num_consensus_layers=1
    )

    # Create dummy graphs
    peak_graph = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        edge_attr=torch.randn(4, 2)
    )

    methyl_graph = Data(
        x=torch.randn(6, 11),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        edge_attr=torch.randn(4, 2)
    )

    model.eval()
    with torch.no_grad():
        matching_matrix, _ = model(peak_graph, methyl_graph)

    assert matching_matrix.shape == (5, 6)


def test_output_formatter():
    """Test output formatter."""
    from nmr_graph_matching.utils.output_formatter import OutputFormatter

    formatter = OutputFormatter(confidence_threshold=0.5)
    assert formatter is not None
    assert formatter.confidence_threshold == 0.5


def test_matching_algorithms():
    """Test matching algorithms."""
    from nmr_graph_matching.models.matching import (
        HungarianMatching,
        GreedyMatching,
        SinkhornMatching
    )

    # Create dummy similarity matrix
    similarity = torch.randn(3, 4)

    # Test Hungarian
    hungarian = HungarianMatching()
    row_ind, col_ind, _ = hungarian(similarity)
    assert len(row_ind) == 3

    # Test Greedy
    greedy = GreedyMatching()
    assignments, confidences = greedy(similarity)
    assert len(assignments) == 3

    # Test Sinkhorn
    sinkhorn = SinkhornMatching()
    result = sinkhorn(similarity)
    assert result.shape == similarity.shape


def test_losses():
    """Test loss functions."""
    from nmr_graph_matching.training.losses import (
        MatchingLoss,
        DistanceConsistencyLoss,
        PermutationLoss
    )

    # Dummy data
    matching_matrix = torch.randn(3, 4)
    ground_truth = torch.zeros(3, 4)
    ground_truth[0, 1] = 1.0
    ground_truth[1, 2] = 1.0
    ground_truth[2, 0] = 1.0

    # Test matching loss
    match_loss = MatchingLoss()
    loss = match_loss(matching_matrix, ground_truth)
    assert loss.item() >= 0

    # Test distance consistency loss
    peak_adj = torch.rand(3, 3)
    methyl_dist = torch.rand(4, 4)
    dist_loss = DistanceConsistencyLoss()
    loss = dist_loss(matching_matrix, peak_adj, methyl_dist)
    assert loss.item() >= 0

    # Test permutation loss
    perm_loss = PermutationLoss()
    loss = perm_loss(torch.softmax(matching_matrix, dim=1))
    assert loss.item() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
