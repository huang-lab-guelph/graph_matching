"""
Methyl Match: Automated Methyl Assignment for NMR Spectroscopy.

This package provides tools for automated methyl assignment using graph matching
algorithms to correlate NMR experimental data with protein structural information.
"""

__version__ = "0.1.0"

from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
from methyl_match.preprocessing import (
    MethylNetworkBuilder,
    MethylNetwork,
    PeakNetworkBuilder,
    PeakNetwork,
)
from methyl_match.matching import (
    GraphMatcher,
    MatchingResult,
    GreedyMatcher,
    HungarianMatcher,
    QAPMatcher,
    SpectralMatcher,
)

__all__ = [
    "PDBParser",
    "NOESYParser",
    "HMQCParser",
    "MethylNetworkBuilder",
    "MethylNetwork",
    "PeakNetworkBuilder",
    "PeakNetwork",
    "GraphMatcher",
    "MatchingResult",
    "GreedyMatcher",
    "HungarianMatcher",
    "QAPMatcher",
    "SpectralMatcher",
]
