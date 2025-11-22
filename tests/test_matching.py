"""
Tests for the matching module.

This module tests all graph matching algorithms and utilities.
"""

import pytest
import numpy as np
import networkx as nx

from methyl_match.matching import (
    GraphMatcher,
    MatchingResult,
    GreedyMatcher,
    HungarianMatcher,
    QAPMatcher,
    SpectralMatcher,
    compute_topology_consistency,
    compute_distance_consistency,
    compute_confidence_scores,
    validate_assignment,
)


# Fixtures for test graphs


@pytest.fixture
def simple_experimental_graph():
    """Create a simple experimental graph with 3 nodes."""
    G = nx.Graph()
    G.add_node(0, residue_type='LEU', h_shift=1.0, c_shift=20.0)
    G.add_node(1, residue_type='VAL', h_shift=1.5, c_shift=21.0)
    G.add_node(2, residue_type='ILE', h_shift=2.0, c_shift=22.0)

    G.add_edge(0, 1, weight=0.8)
    G.add_edge(1, 2, weight=0.6)

    return G


@pytest.fixture
def simple_structural_graph():
    """Create a simple structural graph with 3 nodes."""
    G = nx.Graph()
    G.add_node(0, residue_type='LEU', position=np.array([0.0, 0.0, 0.0]))
    G.add_node(1, residue_type='VAL', position=np.array([5.0, 0.0, 0.0]))
    G.add_node(2, residue_type='ILE', position=np.array([10.0, 0.0, 0.0]))

    G.add_edge(0, 1, weight=5.0)
    G.add_edge(1, 2, weight=5.0)

    return G


@pytest.fixture
def larger_experimental_graph():
    """Create a larger experimental graph with 10 nodes."""
    G = nx.Graph()
    residue_types = ['LEU', 'VAL', 'ILE', 'ALA', 'THR'] * 2

    for i in range(10):
        G.add_node(i, residue_type=residue_types[i])

    # Add some edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (0, 5), (2, 7)]
    for i, j in edges:
        G.add_edge(i, j, weight=np.random.uniform(0.5, 1.0))

    return G


@pytest.fixture
def larger_structural_graph():
    """Create a larger structural graph with 10 nodes."""
    G = nx.Graph()
    residue_types = ['LEU', 'VAL', 'ILE', 'ALA', 'THR'] * 2

    for i in range(10):
        pos = np.array([i * 3.0, np.sin(i) * 2.0, np.cos(i) * 2.0])
        G.add_node(i, residue_type=residue_types[i], position=pos)

    # Add edges based on distance
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pos_i = G.nodes[nodes[i]]['position']
            pos_j = G.nodes[nodes[j]]['position']
            dist = np.linalg.norm(pos_i - pos_j)
            if dist < 10.0:
                G.add_edge(nodes[i], nodes[j], weight=dist)

    return G


# Test MatchingResult class


def test_matching_result_creation():
    """Test MatchingResult creation and validation."""
    result = MatchingResult(
        assignments={0: 0, 1: 1, 2: 2},
        confidence_scores={0: 0.9, 1: 0.8, 2: 0.7},
    )

    assert result.num_assignments == 3
    assert result.num_unassigned == 0
    assert result.assignment_rate == 1.0
    assert result.mean_confidence == pytest.approx(0.8, rel=0.01)


def test_matching_result_with_unassigned():
    """Test MatchingResult with unassigned peaks."""
    result = MatchingResult(
        assignments={0: 0, 1: 1},
        confidence_scores={0: 0.9, 1: 0.8},
        unassigned_peaks=[2, 3],
    )

    assert result.num_assignments == 2
    assert result.num_unassigned == 2
    assert result.assignment_rate == 0.5


def test_matching_result_filter_by_confidence():
    """Test filtering assignments by confidence threshold."""
    result = MatchingResult(
        assignments={0: 0, 1: 1, 2: 2, 3: 3},
        confidence_scores={0: 0.95, 1: 0.85, 2: 0.65, 3: 0.45},
    )

    filtered = result.filter_by_confidence(0.7)

    assert filtered.num_assignments == 2
    assert 0 in filtered.assignments
    assert 1 in filtered.assignments
    assert 2 not in filtered.assignments
    assert 3 not in filtered.assignments


def test_matching_result_get_assignment_list():
    """Test getting assignments as sorted list."""
    result = MatchingResult(
        assignments={0: 0, 1: 1, 2: 2},
        confidence_scores={0: 0.7, 1: 0.9, 2: 0.8},
    )

    assignment_list = result.get_assignment_list()

    assert len(assignment_list) == 3
    # Should be sorted by confidence (descending)
    assert assignment_list[0][0] == 1  # exp_id
    assert assignment_list[0][2] == 0.9  # confidence
    assert assignment_list[1][0] == 2
    assert assignment_list[2][0] == 0


def test_matching_result_invalid_confidence():
    """Test that invalid confidence scores raise error."""
    with pytest.raises(ValueError, match="must be in range"):
        MatchingResult(
            assignments={0: 0},
            confidence_scores={0: 1.5},  # Invalid: > 1.0
        )


# Test GreedyMatcher


def test_greedy_matcher_simple(simple_experimental_graph, simple_structural_graph):
    """Test GreedyMatcher on simple graphs."""
    matcher = GreedyMatcher()
    result = matcher.match(simple_experimental_graph, simple_structural_graph)

    assert isinstance(result, MatchingResult)
    assert result.num_assignments > 0
    assert result.num_assignments <= 3
    assert result.metadata['algorithm'] == 'greedy'


def test_greedy_matcher_perfect_match():
    """Test GreedyMatcher on graphs with perfect matching."""
    # Create identical graphs
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    for i in range(3):
        G_exp.add_node(i, residue_type='LEU')
        G_struct.add_node(i, residue_type='LEU')

    G_exp.add_edge(0, 1, weight=1.0)
    G_struct.add_edge(0, 1, weight=1.0)

    matcher = GreedyMatcher()
    result = matcher.match(G_exp, G_struct)

    # Should get perfect assignments
    assert result.num_assignments == 3
    assert result.mean_confidence > 0.5


def test_greedy_matcher_with_topology(simple_experimental_graph, simple_structural_graph):
    """Test GreedyMatcher with topology consideration."""
    matcher = GreedyMatcher(use_topology=True)
    result = matcher.match(simple_experimental_graph, simple_structural_graph)

    assert isinstance(result, MatchingResult)
    assert result.metadata['use_topology'] is True


# Test HungarianMatcher


def test_hungarian_matcher_simple(simple_experimental_graph, simple_structural_graph):
    """Test HungarianMatcher on simple graphs."""
    matcher = HungarianMatcher()
    result = matcher.match(simple_experimental_graph, simple_structural_graph)

    assert isinstance(result, MatchingResult)
    assert result.num_assignments == 3  # Should assign all nodes
    assert result.metadata['algorithm'] == 'hungarian'
    assert 'total_cost' in result.metadata


def test_hungarian_matcher_rectangular():
    """Test HungarianMatcher on non-square matrices."""
    # Experimental graph with 3 nodes
    G_exp = nx.Graph()
    for i in range(3):
        G_exp.add_node(i, residue_type='LEU')

    # Structural graph with 5 nodes
    G_struct = nx.Graph()
    for i in range(5):
        G_struct.add_node(i, residue_type='LEU')

    matcher = HungarianMatcher(handle_rectangular='pad')
    result = matcher.match(G_exp, G_struct)

    assert result.num_assignments == 3
    assert result.metadata['padded'] is True


def test_hungarian_matcher_optimal():
    """Test that HungarianMatcher finds optimal solution."""
    # Create graphs where optimal solution is clear
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    G_exp.add_node(0, residue_type='LEU')
    G_exp.add_node(1, residue_type='VAL')

    G_struct.add_node(0, residue_type='VAL')
    G_struct.add_node(1, residue_type='LEU')

    matcher = HungarianMatcher()
    result = matcher.match(G_exp, G_struct)

    # Should match LEU to LEU and VAL to VAL (swapped indices)
    assert result.assignments[0] == 1  # exp LEU -> struct LEU
    assert result.assignments[1] == 0  # exp VAL -> struct VAL


# Test QAPMatcher


def test_qap_matcher_simple(simple_experimental_graph, simple_structural_graph):
    """Test QAPMatcher on simple graphs."""
    matcher = QAPMatcher(method='faq')
    result = matcher.match(simple_experimental_graph, simple_structural_graph)

    assert isinstance(result, MatchingResult)
    assert result.num_assignments > 0
    assert result.metadata['algorithm'] == 'qap'
    assert result.metadata['method'] == 'faq'
    assert 'qap_fun' in result.metadata
    assert 'qap_nit' in result.metadata


def test_qap_matcher_topology_weight():
    """Test QAPMatcher with different topology weights."""
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    for i in range(3):
        G_exp.add_node(i, residue_type='LEU')
        G_struct.add_node(i, residue_type='LEU')

    G_exp.add_edge(0, 1, weight=1.0)
    G_struct.add_edge(0, 1, weight=1.0)

    # Test with high topology weight
    matcher = QAPMatcher(topology_weight=0.9)
    result = matcher.match(G_exp, G_struct)

    assert result.metadata['topology_weight'] == 0.9


# Test SpectralMatcher


def test_spectral_matcher_available():
    """Test if SpectralMatcher can be instantiated."""
    try:
        matcher = SpectralMatcher()
        assert isinstance(matcher, GraphMatcher)
    except ImportError as e:
        pytest.skip(f"pygmtools not available: {e}")


def test_spectral_matcher_simple(simple_experimental_graph, simple_structural_graph):
    """Test SpectralMatcher on simple graphs."""
    try:
        matcher = SpectralMatcher(method='sm')
        result = matcher.match(simple_experimental_graph, simple_structural_graph)

        assert isinstance(result, MatchingResult)
        assert result.metadata['algorithm'] == 'spectral'
        assert result.metadata['method'] == 'sm'
    except ImportError:
        pytest.skip("pygmtools not available")


def test_spectral_matcher_with_laplacian(simple_experimental_graph, simple_structural_graph):
    """Test SpectralMatcher with Laplacian matrix."""
    try:
        matcher = SpectralMatcher(use_laplacian=True)
        result = matcher.match(simple_experimental_graph, simple_structural_graph)

        assert result.metadata['use_laplacian'] is True
    except ImportError:
        pytest.skip("pygmtools not available")


# Test utility functions


def test_compute_topology_consistency():
    """Test topology consistency computation."""
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    G_exp.add_nodes_from([0, 1, 2])
    G_struct.add_nodes_from([0, 1, 2])

    G_exp.add_edge(0, 1)
    G_struct.add_edge(0, 1)

    # Perfect assignment
    assignments = {0: 0, 1: 1, 2: 2}

    consistency = compute_topology_consistency(assignments, G_exp, G_struct)

    assert 0.0 <= consistency <= 1.0
    assert consistency > 0.5  # Should be high for this case


def test_compute_distance_consistency():
    """Test distance consistency computation."""
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    G_exp.add_nodes_from([0, 1])
    G_struct.add_nodes_from([0, 1])

    G_exp.add_edge(0, 1, weight=0.9)  # High NOE intensity
    G_struct.add_edge(0, 1, weight=5.0)  # Short distance

    assignments = {0: 0, 1: 1}

    consistency = compute_distance_consistency(assignments, G_exp, G_struct)

    assert 0.0 <= consistency <= 1.0


def test_validate_assignment():
    """Test assignment validation."""
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    for i in range(3):
        G_exp.add_node(i)
        G_struct.add_node(i)

    G_exp.add_edge(0, 1)
    G_struct.add_edge(0, 1)

    assignments = {0: 0, 1: 1, 2: 2}

    metrics = validate_assignment(assignments, G_exp, G_struct)

    assert 'topology_consistency' in metrics
    assert 'distance_consistency' in metrics
    assert 'assignment_rate' in metrics
    assert 'uniqueness' in metrics

    assert metrics['assignment_rate'] == 1.0
    assert metrics['uniqueness'] == 1.0


def test_compute_confidence_scores():
    """Test confidence score computation."""
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    for i in range(3):
        G_exp.add_node(i)
        G_struct.add_node(i)

    G_exp.add_edge(0, 1)
    G_struct.add_edge(0, 1)

    assignments = {0: 0, 1: 1, 2: 2}
    cost_matrix = np.array([
        [0.1, 0.9, 0.8],
        [0.9, 0.1, 0.8],
        [0.8, 0.8, 0.1],
    ])

    confidence = compute_confidence_scores(
        assignments, cost_matrix, G_exp, G_struct, method='cost'
    )

    assert len(confidence) == 3
    for exp_id, conf in confidence.items():
        assert 0.0 <= conf <= 1.0


# Integration tests


def test_all_matchers_produce_results(larger_experimental_graph, larger_structural_graph):
    """Test that all matchers produce valid results."""
    matchers = [
        GreedyMatcher(),
        HungarianMatcher(),
        QAPMatcher(),
    ]

    # Add SpectralMatcher if available
    try:
        matchers.append(SpectralMatcher())
    except ImportError:
        pass

    for matcher in matchers:
        result = matcher.match(larger_experimental_graph, larger_structural_graph)

        assert isinstance(result, MatchingResult)
        assert result.num_assignments > 0
        assert result.mean_confidence >= 0.0
        assert result.mean_confidence <= 1.0
        assert 'runtime_seconds' in result.metadata


def test_matcher_comparison(simple_experimental_graph, simple_structural_graph):
    """Test that different matchers can be compared."""
    results = {}

    matchers = {
        'greedy': GreedyMatcher(),
        'hungarian': HungarianMatcher(),
        'qap': QAPMatcher(),
    }

    for name, matcher in matchers.items():
        results[name] = matcher.match(simple_experimental_graph, simple_structural_graph)

    # All should produce results
    for name, result in results.items():
        assert result.num_assignments > 0

    # Compare assignment rates
    for name, result in results.items():
        assert result.assignment_rate <= 1.0


def test_confidence_threshold_filtering():
    """Test that confidence threshold filters assignments."""
    G_exp = nx.Graph()
    G_struct = nx.Graph()

    for i in range(5):
        G_exp.add_node(i, residue_type='LEU')
        G_struct.add_node(i, residue_type='LEU')

    # Test with different thresholds
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    results = []

    for threshold in thresholds:
        matcher = HungarianMatcher(confidence_threshold=threshold)
        result = matcher.match(G_exp, G_struct)
        results.append(result)

    # Higher thresholds should generally produce fewer assignments
    for i in range(len(thresholds) - 1):
        assert results[i + 1].num_assignments <= results[i].num_assignments


# Error handling tests


def test_matcher_invalid_input():
    """Test that matchers handle invalid inputs."""
    matcher = HungarianMatcher()

    # Not a graph
    with pytest.raises(ValueError):
        matcher.match("not a graph", nx.Graph())

    # Empty graph
    with pytest.raises(ValueError):
        matcher.match(nx.Graph(), nx.Graph())


def test_matching_result_inconsistent_keys():
    """Test that MatchingResult validates consistent keys."""
    with pytest.raises(ValueError, match="must have the same keys"):
        MatchingResult(
            assignments={0: 0, 1: 1},
            confidence_scores={0: 0.9},  # Missing key 1
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
