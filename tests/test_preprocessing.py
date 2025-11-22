"""
Comprehensive tests for the preprocessing module.

Tests MethylNetworkBuilder and PeakNetworkBuilder with real data.
"""

import pytest
import numpy as np
import networkx as nx
from pathlib import Path

from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
from methyl_match.preprocessing import (
    MethylNetworkBuilder,
    MethylNetwork,
    PeakNetworkBuilder,
    PeakNetwork,
)


# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "test"


class TestMethylNetworkBuilder:
    """Tests for MethylNetworkBuilder."""

    def test_builder_initialization(self):
        """Test that builder initializes with correct parameters."""
        builder = MethylNetworkBuilder(distance_cutoff=10.0)
        assert builder.distance_cutoff == 10.0
        assert builder.edge_weight_function == 'inverse'
        assert builder.include_self_loops == False

    def test_build_network_from_pdb(self):
        """Test building network from real PDB structure."""
        # Parse PDB
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        # Build network
        builder = MethylNetworkBuilder(distance_cutoff=12.0)
        network = builder.build_network(methyls)

        # Verify network structure
        assert isinstance(network, MethylNetwork)
        assert network.graph.number_of_nodes() == len(methyls)
        assert len(network.labels) == len(methyls)
        assert len(network.methyls) == len(methyls)

        print(f"\nBuilt methyl network:")
        print(f"  Nodes: {network.graph.number_of_nodes()}")
        print(f"  Edges: {network.graph.number_of_edges()}")
        print(f"  Density: {nx.density(network.graph):.3f}")

    def test_network_nodes_have_features(self):
        """Test that network nodes have proper feature vectors."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        builder = MethylNetworkBuilder(distance_cutoff=10.0)
        network = builder.build_network(methyls)

        # Check node features
        for node_id in range(len(methyls)):
            assert node_id in network.node_features
            features = network.node_features[node_id]
            assert isinstance(features, np.ndarray)
            assert features.shape == (10,)  # 3 coords + 6 onehot + 1 res_num
            assert not np.isnan(features).any()

    def test_network_edges_respect_cutoff(self):
        """Test that edges are only created within distance cutoff."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        distance_cutoff = 8.0
        builder = MethylNetworkBuilder(distance_cutoff=distance_cutoff)
        network = builder.build_network(methyls)

        # Check all edges respect cutoff
        for u, v, data in network.graph.edges(data=True):
            assert data['distance'] <= distance_cutoff
            print(f"Edge {u}-{v}: distance={data['distance']:.2f}Å, weight={data['weight']:.3f}")
            if nx.number_of_edges(network.graph) <= 5:
                break  # Just print first few

    def test_different_distance_cutoffs(self):
        """Test that different cutoffs produce different networks."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        # Build with different cutoffs
        builder_small = MethylNetworkBuilder(distance_cutoff=6.0)
        network_small = builder_small.build_network(methyls)

        builder_large = MethylNetworkBuilder(distance_cutoff=15.0)
        network_large = builder_large.build_network(methyls)

        # Larger cutoff should have more edges
        assert network_large.graph.number_of_edges() > network_small.graph.number_of_edges()

        print(f"\nCutoff comparison:")
        print(f"  6.0Å:  {network_small.graph.number_of_edges()} edges")
        print(f"  15.0Å: {network_large.graph.number_of_edges()} edges")

    def test_edge_weight_functions(self):
        """Test different edge weight functions."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        for weight_func in ['uniform', 'inverse', 'exponential']:
            builder = MethylNetworkBuilder(
                distance_cutoff=10.0,
                edge_weight_function=weight_func
            )
            network = builder.build_network(methyls)

            # Check weights are computed
            weights = [data['weight'] for u, v, data in network.graph.edges(data=True)]
            assert len(weights) > 0
            assert all(w > 0 for w in weights)

            print(f"\n{weight_func} weights: min={min(weights):.3f}, max={max(weights):.3f}")

    def test_distance_matrix_matches_pdb(self):
        """Test that distance matrix matches PDB parser distances."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        # Get distance matrix from PDB parser
        pdb_dist_matrix, _ = parser.get_distance_matrix(methyls)

        # Build network
        builder = MethylNetworkBuilder(distance_cutoff=20.0)  # Large cutoff
        network = builder.build_network(methyls)

        # Compare distance matrices
        np.testing.assert_array_almost_equal(
            network.distance_matrix,
            pdb_dist_matrix,
            decimal=5
        )

    def test_get_neighbors(self):
        """Test neighbor retrieval functionality."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        builder = MethylNetworkBuilder(distance_cutoff=10.0)
        network = builder.build_network(methyls)

        # Get neighbors of first node
        neighbors = network.get_neighbors(0)
        assert isinstance(neighbors, list)
        assert all(isinstance(n, int) for n in neighbors)

        # Test with distance filter
        close_neighbors = network.get_neighbors(0, max_distance=7.0)
        assert len(close_neighbors) <= len(neighbors)

        print(f"\nNode 0 ({network.labels[0]}):")
        print(f"  All neighbors: {len(neighbors)}")
        print(f"  Close neighbors (<7Å): {len(close_neighbors)}")

    def test_get_statistics(self):
        """Test network statistics computation."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        builder = MethylNetworkBuilder(distance_cutoff=10.0)
        network = builder.build_network(methyls)

        stats = builder.get_statistics(network)

        # Verify all expected keys
        expected_keys = ['num_nodes', 'num_edges', 'density', 'avg_degree',
                        'avg_clustering', 'num_components', 'avg_distance',
                        'min_distance', 'max_distance']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

        print(f"\nNetwork statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def test_empty_methyl_list_raises_error(self):
        """Test that empty methyl list raises appropriate error."""
        builder = MethylNetworkBuilder()
        with pytest.raises(ValueError, match="empty methyl list"):
            builder.build_network([])


class TestPeakNetworkBuilder:
    """Tests for PeakNetworkBuilder."""

    def test_builder_initialization(self):
        """Test that builder initializes with correct parameters."""
        builder = PeakNetworkBuilder(intensity_threshold=30000)
        assert builder.intensity_threshold == 30000
        assert builder.normalize_intensities == True

    def test_build_network_from_nmr(self):
        """Test building network from real NMR data."""
        # Parse NMR data
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        # Build network
        builder = PeakNetworkBuilder(intensity_threshold=0)
        network = builder.build_network(hmqc_peaks, noe_peaks)

        # Verify network structure
        assert isinstance(network, PeakNetwork)
        assert network.graph.number_of_nodes() == len(hmqc_peaks)
        assert len(network.labels) == len(hmqc_peaks)

        print(f"\nBuilt peak network:")
        print(f"  Nodes (HMQC peaks): {network.graph.number_of_nodes()}")
        print(f"  Edges (NOE correlations): {network.graph.number_of_edges()}")
        print(f"  Density: {nx.density(network.graph):.3f}")

    def test_network_nodes_have_features(self):
        """Test that network nodes have proper feature vectors."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        builder = PeakNetworkBuilder()
        network = builder.build_network(hmqc_peaks, noe_peaks)

        # Check node features
        for node_id in range(len(hmqc_peaks)):
            assert node_id in network.node_features
            features = network.node_features[node_id]
            assert isinstance(features, np.ndarray)
            assert features.shape == (4,)  # h_shift, c_shift, intensity, degree
            assert not np.isnan(features).any()

    def test_intensity_threshold_filtering(self):
        """Test that intensity threshold filters edges correctly."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        # Build with no threshold
        builder_low = PeakNetworkBuilder(intensity_threshold=0)
        network_low = builder_low.build_network(hmqc_peaks, noe_peaks)

        # Build with high threshold
        builder_high = PeakNetworkBuilder(intensity_threshold=40000)
        network_high = builder_high.build_network(hmqc_peaks, noe_peaks)

        # High threshold should have fewer edges
        assert network_high.graph.number_of_edges() <= network_low.graph.number_of_edges()

        print(f"\nIntensity threshold comparison:")
        print(f"  No threshold:  {network_low.graph.number_of_edges()} edges")
        print(f"  >40000:        {network_high.graph.number_of_edges()} edges")

    def test_edge_features_contain_intensities(self):
        """Test that edge features contain NOE intensities."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        builder = PeakNetworkBuilder()
        network = builder.build_network(hmqc_peaks, noe_peaks)

        # Check edge features and intensities
        for u, v, data in network.graph.edges(data=True):
            assert 'intensity' in data
            assert 'weight' in data
            assert data['intensity'] >= 0  # Can be 0 after normalization
            assert (u, v) in network.edge_features or (v, u) in network.edge_features

    def test_correlation_matrix(self):
        """Test that correlation matrix is properly constructed."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        builder = PeakNetworkBuilder()
        network = builder.build_network(hmqc_peaks, noe_peaks)

        # Check correlation matrix
        assert network.correlation_matrix.shape == (len(hmqc_peaks), len(hmqc_peaks))
        # Should be symmetric
        np.testing.assert_array_almost_equal(
            network.correlation_matrix,
            network.correlation_matrix.T
        )
        # Non-negative values
        assert np.all(network.correlation_matrix >= 0)

        print(f"\nCorrelation matrix: {network.correlation_matrix.shape}")
        print(f"Non-zero entries: {np.count_nonzero(network.correlation_matrix)}")

    def test_get_neighbors(self):
        """Test neighbor retrieval functionality."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        builder = PeakNetworkBuilder()
        network = builder.build_network(hmqc_peaks, noe_peaks)

        # Find a node with neighbors
        node_with_neighbors = None
        for node in range(len(hmqc_peaks)):
            if network.graph.degree(node) > 0:
                node_with_neighbors = node
                break

        if node_with_neighbors is not None:
            neighbors = network.get_neighbors(node_with_neighbors)
            assert isinstance(neighbors, list)
            assert len(neighbors) > 0

            # Test with intensity filter
            strong_neighbors = network.get_neighbors(node_with_neighbors, min_intensity=0.5)
            assert len(strong_neighbors) <= len(neighbors)

            print(f"\nNode {node_with_neighbors} ({network.labels[node_with_neighbors]}):")
            print(f"  All neighbors: {len(neighbors)}")
            print(f"  Strong NOEs: {len(strong_neighbors)}")

    def test_get_statistics(self):
        """Test network statistics computation."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        builder = PeakNetworkBuilder()
        network = builder.build_network(hmqc_peaks, noe_peaks)

        stats = builder.get_statistics(network)

        # Verify expected keys
        expected_keys = ['num_nodes', 'num_edges', 'density', 'avg_degree',
                        'avg_clustering', 'num_components']
        for key in expected_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))

        print(f"\nPeak network statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def test_empty_hmqc_list_raises_error(self):
        """Test that empty HMQC list raises appropriate error."""
        builder = PeakNetworkBuilder()
        with pytest.raises(ValueError, match="empty HMQC peak list"):
            builder.build_network([], [])


class TestIntegration:
    """Integration tests using both network builders."""

    def test_networks_have_compatible_sizes(self):
        """Test that methyl and peak networks can have matching node counts."""
        # Parse all data
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        pdb_parser = PDBParser(str(pdb_file))
        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        methyls = pdb_parser.extract_methyls()
        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        # Build both networks
        methyl_builder = MethylNetworkBuilder(distance_cutoff=10.0)
        peak_builder = PeakNetworkBuilder()

        methyl_network = methyl_builder.build_network(methyls)
        peak_network = peak_builder.build_network(hmqc_peaks, noe_peaks)

        print(f"\n=== Network Comparison ===")
        print(f"Methyl network: {methyl_network.graph.number_of_nodes()} nodes, "
              f"{methyl_network.graph.number_of_edges()} edges")
        print(f"Peak network:   {peak_network.graph.number_of_nodes()} nodes, "
              f"{peak_network.graph.number_of_edges()} edges")

        # Both should be valid graphs
        assert methyl_network.graph.number_of_nodes() > 0
        assert peak_network.graph.number_of_nodes() > 0

    def test_network_properties(self):
        """Test that both networks have expected graph properties."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"

        pdb_parser = PDBParser(str(pdb_file))
        hmqc_parser = HMQCParser(str(hmqc_file))
        noesy_parser = NOESYParser(str(noesy_file))

        methyls = pdb_parser.extract_methyls()
        hmqc_peaks = hmqc_parser.parse()
        noe_peaks = noesy_parser.parse()

        methyl_builder = MethylNetworkBuilder(distance_cutoff=10.0)
        peak_builder = PeakNetworkBuilder()

        methyl_network = methyl_builder.build_network(methyls)
        peak_network = peak_builder.build_network(hmqc_peaks, noe_peaks)

        # Both should be connected or have few components
        methyl_components = nx.number_connected_components(methyl_network.graph)
        peak_components = nx.number_connected_components(peak_network.graph)

        print(f"\nConnected components:")
        print(f"  Methyl network: {methyl_components}")
        print(f"  Peak network: {peak_components}")

        # Verify basic graph properties (using isinstance instead of is_undirected for nx 3.x)
        assert isinstance(methyl_network.graph, nx.Graph)
        assert isinstance(peak_network.graph, nx.Graph)
        assert not methyl_network.graph.is_directed()
        assert not peak_network.graph.is_directed()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
