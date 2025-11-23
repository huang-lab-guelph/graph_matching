"""
Base classes for result formatting.

This module provides the abstract base class for all result formatters
in methyl_match.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import networkx as nx

from methyl_match.matching.base import MatchingResult
from methyl_match.preprocessing.methyl_network import MethylNetwork
from methyl_match.preprocessing.peak_network import PeakNetwork


class ResultFormatter(ABC):
    """
    Abstract base class for result formatters.

    All formatters should inherit from this class and implement the format()
    method to export results in a specific format.

    Attributes:
        include_unassigned: Whether to include unassigned peaks in output
        include_metadata: Whether to include algorithm metadata
        confidence_threshold: Minimum confidence to include (default: 0.0)
    """

    def __init__(
        self,
        include_unassigned: bool = True,
        include_metadata: bool = True,
        confidence_threshold: float = 0.0,
    ):
        """
        Initialize the result formatter.

        Args:
            include_unassigned: Whether to include unassigned peaks in output
            include_metadata: Whether to include algorithm metadata
            confidence_threshold: Minimum confidence to include (0.0-1.0)
        """
        self.include_unassigned = include_unassigned
        self.include_metadata = include_metadata
        self.confidence_threshold = confidence_threshold

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
            )

    @abstractmethod
    def format(
        self,
        result: MatchingResult,
        structural_network: MethylNetwork,
        experimental_network: PeakNetwork,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> str:
        """
        Format the matching result.

        Args:
            result: MatchingResult from a matching algorithm
            structural_network: MethylNetwork from PDB structure
            experimental_network: PeakNetwork from HMQC/NOESY data
            output_path: Optional path to save formatted output
            **kwargs: Additional formatter-specific options

        Returns:
            Formatted output as string
        """
        pass

    def _filter_assignments(
        self, result: MatchingResult
    ) -> Dict[int, tuple]:
        """
        Filter assignments by confidence threshold.

        Args:
            result: MatchingResult to filter

        Returns:
            Dictionary mapping exp_id to (struct_id, confidence)
        """
        filtered = {}
        for exp_id, struct_id in result.assignments.items():
            confidence = result.confidence_scores[exp_id]
            if confidence >= self.confidence_threshold:
                filtered[exp_id] = (struct_id, confidence)
        return filtered

    def _get_methyl_label(
        self, methyl_id: int, structural_network: MethylNetwork
    ) -> str:
        """
        Get human-readable label for a methyl group.

        Args:
            methyl_id: Methyl node ID
            structural_network: MethylNetwork containing methyl data

        Returns:
            Label string (e.g., "LEU42-CD1")
        """
        methyl = structural_network.methyls[methyl_id]
        return f"{methyl.residue_name}{methyl.residue_number}-{methyl.atom_name}"

    def _get_peak_label(
        self, peak_id: int, experimental_network: PeakNetwork
    ) -> str:
        """
        Get human-readable label for an HMQC peak.

        Args:
            peak_id: Peak node ID
            experimental_network: PeakNetwork containing peak data

        Returns:
            Label string (e.g., "Peak_5 (1H: 0.85, 13C: 21.5)")
        """
        peak = experimental_network.hmqc_peaks[peak_id]
        if peak.assignment:
            return f"{peak.assignment} (1H: {peak.h_shift:.2f}, 13C: {peak.c_shift:.1f})"
        else:
            return f"Peak_{peak.index} (1H: {peak.h_shift:.2f}, 13C: {peak.c_shift:.1f})"
