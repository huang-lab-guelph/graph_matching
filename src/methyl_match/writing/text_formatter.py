"""
Text formatter for human-readable output.

This module provides a formatter that creates human-readable text reports
of matching results with summary statistics and detailed assignment lists.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

from methyl_match.writing.base import ResultFormatter
from methyl_match.matching.base import MatchingResult
from methyl_match.preprocessing.methyl_network import MethylNetwork
from methyl_match.preprocessing.peak_network import PeakNetwork


class TextFormatter(ResultFormatter):
    """
    Formatter for human-readable text output.

    Creates a comprehensive text report including:
    - Summary statistics
    - Algorithm metadata
    - Detailed assignment list sorted by confidence
    - Unassigned peaks list

    Example output:
        ================================================================
        METHYL ASSIGNMENT RESULTS
        ================================================================
        Date: 2025-11-22 14:30:00
        Algorithm: QAPMatcher

        SUMMARY STATISTICS
        ------------------------------------------------------------------
        Total Assignments:        45
        Assignment Rate:          78.9%
        Mean Confidence:          0.723
        High Confidence (>=0.7):  32 (71.1%)
        Medium Confidence:        10 (22.2%)
        Low Confidence (<0.5):    3 (6.7%)

        DETAILED ASSIGNMENTS (sorted by confidence)
        ------------------------------------------------------------------
        Peak                            → Methyl          Confidence
        ------------------------------------------------------------------
        Peak_12 (1H: 0.85, 13C: 21.5)  → LEU42-CD1       0.952 🟢
        Peak_7 (1H: 0.91, 13C: 23.1)   → VAL8-CG2        0.891 🟢
        ...
    """

    def __init__(
        self,
        include_unassigned: bool = True,
        include_metadata: bool = True,
        confidence_threshold: float = 0.0,
        show_quality_icons: bool = True,
        show_chemical_shifts: bool = True,
    ):
        """
        Initialize text formatter.

        Args:
            include_unassigned: Include unassigned peaks section
            include_metadata: Include algorithm metadata section
            confidence_threshold: Minimum confidence to include
            show_quality_icons: Show quality icons (🟢🟡🔴)
            show_chemical_shifts: Show chemical shifts in peak labels
        """
        super().__init__(include_unassigned, include_metadata, confidence_threshold)
        self.show_quality_icons = show_quality_icons
        self.show_chemical_shifts = show_chemical_shifts

    def format(
        self,
        result: MatchingResult,
        structural_network: MethylNetwork,
        experimental_network: PeakNetwork,
        output_path: Optional[Path] = None,
        title: str = "METHYL ASSIGNMENT RESULTS",
    ) -> str:
        """
        Format matching result as human-readable text.

        Args:
            result: MatchingResult from matching algorithm
            structural_network: MethylNetwork from PDB structure
            experimental_network: PeakNetwork from HMQC/NOESY
            output_path: Optional path to save text file
            title: Title for the report

        Returns:
            Formatted text report
        """
        lines = []
        sep = "=" * 70
        subsep = "-" * 70

        # Header
        lines.append(sep)
        lines.append(title.center(70))
        lines.append(sep)
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.include_metadata and "algorithm" in result.metadata:
            lines.append(f"Algorithm: {result.metadata['algorithm']}")
        lines.append("")

        # Summary Statistics
        lines.append("SUMMARY STATISTICS")
        lines.append(subsep)

        filtered = self._filter_assignments(result)
        high_conf = sum(1 for _, (_, c) in filtered.items() if c >= 0.7)
        medium_conf = sum(1 for _, (_, c) in filtered.items() if 0.5 <= c < 0.7)
        low_conf = sum(1 for _, (_, c) in filtered.items() if c < 0.5)

        lines.append(f"Total Assignments:        {result.num_assignments}")
        lines.append(f"Assignment Rate:          {result.assignment_rate * 100:.1f}%")
        lines.append(f"Mean Confidence:          {result.mean_confidence:.3f}")
        lines.append(
            f"High Confidence (>=0.7):  {high_conf} "
            f"({100 * high_conf / max(result.num_assignments, 1):.1f}%)"
        )
        lines.append(
            f"Medium Confidence:        {medium_conf} "
            f"({100 * medium_conf / max(result.num_assignments, 1):.1f}%)"
        )
        lines.append(
            f"Low Confidence (<0.5):    {low_conf} "
            f"({100 * low_conf / max(result.num_assignments, 1):.1f}%)"
        )

        if self.include_unassigned:
            lines.append(f"Unassigned Peaks:         {result.num_unassigned}")

        lines.append("")

        # Algorithm Metadata
        if self.include_metadata and result.metadata:
            lines.append("ALGORITHM METADATA")
            lines.append(subsep)
            for key, value in result.metadata.items():
                if key != "algorithm":  # Already shown in header
                    # Format key nicely
                    display_key = key.replace("_", " ").title()
                    # Format value
                    if isinstance(value, float):
                        display_value = f"{value:.4f}"
                    else:
                        display_value = str(value)
                    lines.append(f"{display_key:25s} {display_value}")
            lines.append("")

        # Detailed Assignments
        lines.append("DETAILED ASSIGNMENTS (sorted by confidence)")
        lines.append(subsep)
        lines.append(f"{'Peak':<35s} → {'Methyl':<15s} {'Confidence':>10s}")
        lines.append(subsep)

        # Sort by confidence (descending)
        assignment_list = result.get_assignment_list()

        for exp_id, struct_id, confidence in assignment_list:
            if confidence < self.confidence_threshold:
                continue

            # Get labels
            if self.show_chemical_shifts:
                peak_label = self._get_peak_label(exp_id, experimental_network)
            else:
                peak = experimental_network.hmqc_peaks[exp_id]
                peak_label = peak.assignment if peak.assignment else f"Peak_{peak.index}"

            methyl_label = self._get_methyl_label(struct_id, structural_network)

            # Quality icon
            if self.show_quality_icons:
                if confidence >= 0.7:
                    icon = "🟢"
                elif confidence >= 0.5:
                    icon = "🟡"
                else:
                    icon = "🔴"
            else:
                icon = ""

            lines.append(
                f"{peak_label:<35s} → {methyl_label:<15s} {confidence:>6.3f} {icon}"
            )

        # Unassigned Peaks
        if self.include_unassigned and result.unassigned_peaks:
            lines.append("")
            lines.append("UNASSIGNED PEAKS")
            lines.append(subsep)
            for exp_id in result.unassigned_peaks:
                peak_label = self._get_peak_label(exp_id, experimental_network)
                lines.append(f"  {peak_label}")

        lines.append("")
        lines.append(sep)

        # Join and optionally save
        output = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(output)

        return output
