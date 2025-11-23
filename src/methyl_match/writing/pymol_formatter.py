"""
PyMOL formatter for visualization scripts.

This module provides a formatter that generates PyMOL scripts to visualize
methyl assignments with confidence-based coloring and labeling.
"""

from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

from methyl_match.writing.base import ResultFormatter
from methyl_match.matching.base import MatchingResult
from methyl_match.preprocessing.methyl_network import MethylNetwork
from methyl_match.preprocessing.peak_network import PeakNetwork


class PyMOLFormatter(ResultFormatter):
    """
    Formatter for PyMOL visualization scripts.

    Creates PyMOL scripts that:
    - Load the PDB structure
    - Color assigned methyls by confidence:
      * High confidence (>=0.7): Green
      * Medium confidence (0.5-0.7): Yellow
      * Low confidence (<0.5): Red
      * Unassigned: Gray
    - Add labels with assignment information
    - Create selections for different confidence groups
    - Set up visualization style

    Example PyMOL script:
        # Methyl Assignment Visualization
        # Generated: 2025-11-22
        # Algorithm: QAPMatcher

        load protein.pdb

        # Color all methyls gray (unassigned)
        select all_methyls, (resn LEU and name CD*) or (resn VAL and name CG*)...
        color gray, all_methyls

        # High confidence assignments (green)
        select high_conf, resi 42 and name CD1
        select high_conf, high_conf or (resi 8 and name CG2)
        color green, high_conf

        # Medium confidence assignments (yellow)
        ...

        # Labels
        label resi 42 and name CD1, "Peak_12 (0.95)"
        ...
    """

    def __init__(
        self,
        include_unassigned: bool = True,
        include_metadata: bool = True,
        confidence_threshold: float = 0.0,
        color_by_confidence: bool = True,
        show_labels: bool = True,
        sphere_scale: float = 0.5,
    ):
        """
        Initialize PyMOL formatter.

        Args:
            include_unassigned: Show unassigned methyls (gray)
            include_metadata: Include metadata as comments
            confidence_threshold: Minimum confidence to include
            color_by_confidence: Color by confidence (vs single color)
            show_labels: Add text labels to assignments
            sphere_scale: Scale factor for methyl spheres
        """
        super().__init__(include_unassigned, include_metadata, confidence_threshold)
        self.color_by_confidence = color_by_confidence
        self.show_labels = show_labels
        self.sphere_scale = sphere_scale

    def format(
        self,
        result: MatchingResult,
        structural_network: MethylNetwork,
        experimental_network: PeakNetwork,
        output_path: Optional[Path] = None,
        pdb_path: Optional[Path] = None,
    ) -> str:
        """
        Format matching result as PyMOL script.

        Args:
            result: MatchingResult from matching algorithm
            structural_network: MethylNetwork from PDB structure
            experimental_network: PeakNetwork from HMQC/NOESY
            output_path: Optional path to save PyMOL script (.pml)
            pdb_path: Path to PDB file to load (optional, uses relative path)

        Returns:
            PyMOL script as string
        """
        lines = []

        # Header
        lines.append("# " + "=" * 68)
        lines.append("# Methyl Assignment Visualization")
        lines.append("# " + "=" * 68)
        lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.include_metadata and result.metadata:
            for key, value in result.metadata.items():
                lines.append(f"# {key}: {value}")

        lines.append("#")
        lines.append(f"# Total Assignments: {result.num_assignments}")
        lines.append(f"# Assignment Rate: {result.assignment_rate * 100:.1f}%")
        lines.append(f"# Mean Confidence: {result.mean_confidence:.3f}")
        lines.append("# " + "=" * 68)
        lines.append("")

        # Load PDB
        if pdb_path:
            lines.append(f"# Load structure")
            lines.append(f"load {pdb_path}")
        else:
            lines.append("# Load your PDB structure here:")
            lines.append("# load protein.pdb")

        lines.append("")

        # Basic setup
        lines.append("# Basic visualization setup")
        lines.append("hide everything")
        lines.append("show cartoon")
        lines.append("color gray80, all")
        lines.append("")

        # Create methyl selection
        lines.append("# Create selection for all methyl groups")
        methyl_selection = self._create_methyl_selection_string(structural_network)
        lines.append(f"select all_methyls, {methyl_selection}")
        lines.append("show spheres, all_methyls")
        lines.append(f"set sphere_scale, {self.sphere_scale}, all_methyls")
        lines.append("")

        # Color unassigned methyls gray
        if self.include_unassigned:
            lines.append("# Color all methyls gray initially (unassigned)")
            lines.append("color gray, all_methyls")
            lines.append("")

        # Group assignments by confidence
        high_conf, medium_conf, low_conf = self._group_by_confidence(result)

        # High confidence (green)
        if high_conf:
            lines.append("# High confidence assignments (>=0.7) - GREEN")
            lines.extend(
                self._create_confidence_group(
                    "high_conf",
                    high_conf,
                    "green",
                    structural_network,
                    experimental_network,
                )
            )
            lines.append("")

        # Medium confidence (yellow)
        if medium_conf:
            lines.append("# Medium confidence assignments (0.5-0.7) - YELLOW")
            lines.extend(
                self._create_confidence_group(
                    "medium_conf",
                    medium_conf,
                    "yellow",
                    structural_network,
                    experimental_network,
                )
            )
            lines.append("")

        # Low confidence (red)
        if low_conf:
            lines.append("# Low confidence assignments (<0.5) - RED")
            lines.extend(
                self._create_confidence_group(
                    "low_conf",
                    low_conf,
                    "red",
                    structural_network,
                    experimental_network,
                )
            )
            lines.append("")

        # Labels
        if self.show_labels:
            lines.append("# Add labels to assignments")
            lines.extend(
                self._create_labels(result, structural_network, experimental_network)
            )
            lines.append("")

        # Final setup
        lines.append("# Final visualization setup")
        lines.append("deselect")
        lines.append("center all_methyls")
        lines.append("zoom all_methyls, 5")
        lines.append("set label_size, 14")
        lines.append("set label_color, black")
        lines.append("")
        lines.append("# Legend:")
        lines.append("# Green = High confidence (>=0.7)")
        lines.append("# Yellow = Medium confidence (0.5-0.7)")
        lines.append("# Red = Low confidence (<0.5)")
        if self.include_unassigned:
            lines.append("# Gray = Unassigned")

        # Join and save
        output = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(output)

        return output

    def _create_methyl_selection_string(
        self, structural_network: MethylNetwork
    ) -> str:
        """
        Create PyMOL selection string for all methyl groups.

        Args:
            structural_network: MethylNetwork

        Returns:
            PyMOL selection string
        """
        # Common methyl atom patterns
        selection_parts = [
            "(resn LEU and name CD*)",
            "(resn VAL and name CG*)",
            "(resn ILE and name CD1)",
            "(resn ALA and name CB)",
            "(resn THR and name CG2)",
            "(resn MET and name CE)",
        ]
        return " or ".join(selection_parts)

    def _group_by_confidence(
        self, result: MatchingResult
    ) -> Tuple[List, List, List]:
        """
        Group assignments by confidence level.

        Args:
            result: MatchingResult

        Returns:
            Tuple of (high_conf, medium_conf, low_conf) lists,
            each containing (exp_id, struct_id, confidence) tuples
        """
        high_conf = []
        medium_conf = []
        low_conf = []

        for exp_id, struct_id in result.assignments.items():
            confidence = result.confidence_scores[exp_id]

            if confidence < self.confidence_threshold:
                continue

            assignment = (exp_id, struct_id, confidence)

            if confidence >= 0.7:
                high_conf.append(assignment)
            elif confidence >= 0.5:
                medium_conf.append(assignment)
            else:
                low_conf.append(assignment)

        return high_conf, medium_conf, low_conf

    def _create_confidence_group(
        self,
        group_name: str,
        assignments: List[Tuple[int, int, float]],
        color: str,
        structural_network: MethylNetwork,
        experimental_network: PeakNetwork,
    ) -> List[str]:
        """
        Create PyMOL commands for a confidence group.

        Args:
            group_name: Name for PyMOL selection
            assignments: List of (exp_id, struct_id, confidence) tuples
            color: PyMOL color name
            structural_network: MethylNetwork
            experimental_network: PeakNetwork

        Returns:
            List of PyMOL command strings
        """
        lines = []

        # Create selection by combining residues
        for i, (exp_id, struct_id, confidence) in enumerate(assignments):
            methyl = structural_network.methyls[struct_id]
            resi = methyl.residue_number
            atom = methyl.atom_name

            if i == 0:
                lines.append(f"select {group_name}, (resi {resi} and name {atom})")
            else:
                lines.append(
                    f"select {group_name}, {group_name} or (resi {resi} and name {atom})"
                )

        # Color the selection
        lines.append(f"color {color}, {group_name}")

        return lines

    def _create_labels(
        self,
        result: MatchingResult,
        structural_network: MethylNetwork,
        experimental_network: PeakNetwork,
    ) -> List[str]:
        """
        Create PyMOL label commands.

        Args:
            result: MatchingResult
            structural_network: MethylNetwork
            experimental_network: PeakNetwork

        Returns:
            List of label command strings
        """
        lines = []

        for exp_id, struct_id in result.assignments.items():
            confidence = result.confidence_scores[exp_id]

            if confidence < self.confidence_threshold:
                continue

            methyl = structural_network.methyls[struct_id]
            peak = experimental_network.hmqc_peaks[exp_id]

            resi = methyl.residue_number
            atom = methyl.atom_name

            # Create label text
            if peak.assignment:
                label_text = f"{peak.assignment} ({confidence:.2f})"
            else:
                label_text = f"P{peak.index} ({confidence:.2f})"

            lines.append(f'label (resi {resi} and name {atom}), "{label_text}"')

        return lines
