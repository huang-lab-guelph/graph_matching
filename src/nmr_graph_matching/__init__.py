"""NMR Graph Matching: Deep learning for methyl assignment.

This package implements a deep learning approach to solving the methyl
assignment problem in NMR spectroscopy by matching two graphs:
- Peak network (experimental NMR data)
- Methyl network (protein structure)
"""

__version__ = "0.1.0"

from .data import PDBParser, HMQCParser, NOESYParser
from .graphs import MethylNetworkBuilder, PeakNetworkBuilder
from .models import DGMCModel, GraphEncoder
from .training import Trainer, NMRDataset
from .utils import OutputFormatter, AssignmentResult

__all__ = [
    "PDBParser",
    "HMQCParser",
    "NOESYParser",
    "MethylNetworkBuilder",
    "PeakNetworkBuilder",
    "DGMCModel",
    "GraphEncoder",
    "Trainer",
    "NMRDataset",
    "OutputFormatter",
    "AssignmentResult",
]


def main() -> None:
    """Main entry point for CLI."""
    print("NMR Graph Matching v{}".format(__version__))
    print("Use Python API or scripts/ for running models.")
