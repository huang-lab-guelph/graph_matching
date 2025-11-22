"""
Graph matching algorithms for methyl assignment.

This module provides multiple graph matching algorithms for matching experimental
NMR peak networks to structural methyl networks. All algorithms implement a
common interface (GraphMatcher) for easy comparison and swapping.

Available Algorithms:
--------------------
- GreedyMatcher: Fast greedy matching based on node similarity
- HungarianMatcher: Optimal Linear Assignment Problem (LAP) solution
- QAPMatcher: Quadratic Assignment Problem with topology awareness
- SpectralMatcher: Spectral graph matching using eigendecomposition

Example Usage:
-------------
```python
from methyl_match.matching import HungarianMatcher
from methyl_match.preprocessing import MethylNetworkBuilder, PeakNetworkBuilder

# Build graphs
struct_builder = MethylNetworkBuilder()
peak_builder = PeakNetworkBuilder()

graph_struct = struct_builder.build_network(methyls, distance_cutoff=10.0)
graph_exp = peak_builder.build_network(hmqc_peaks, noesy_peaks)

# Perform matching
matcher = HungarianMatcher()
result = matcher.match(graph_exp, graph_struct)

# Access results
print(f"Assigned {result.num_assignments} peaks")
print(f"Mean confidence: {result.mean_confidence:.2f}")
for exp_id, struct_id, confidence in result.get_assignment_list():
    print(f"Peak {exp_id} -> Methyl {struct_id} (confidence: {confidence:.2f})")
```
"""

from .base import GraphMatcher, MatchingResult
from .utils import (
    compute_chemical_shift_similarity,
    compute_residue_type_similarity,
    compute_topology_consistency,
    compute_distance_consistency,
    compute_confidence_scores,
    validate_assignment,
)

# Import all matchers
from .greedy import GreedyMatcher
from .hungarian import HungarianMatcher
from .qap import QAPMatcher
from .spectral import SpectralMatcher

__all__ = [
    # Base classes
    "GraphMatcher",
    "MatchingResult",
    # Matchers
    "GreedyMatcher",
    "HungarianMatcher",
    "QAPMatcher",
    "SpectralMatcher",
    # Utility functions
    "compute_chemical_shift_similarity",
    "compute_residue_type_similarity",
    "compute_topology_consistency",
    "compute_distance_consistency",
    "compute_confidence_scores",
    "validate_assignment",
]
