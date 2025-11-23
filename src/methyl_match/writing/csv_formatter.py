"""
CSV formatter for tabular data export.

This module provides a formatter that exports matching results as CSV files
suitable for downstream analysis, plotting, and integration with other tools.
"""

from pathlib import Path
from typing import Optional, List, Dict
import csv

from methyl_match.writing.base import ResultFormatter
from methyl_match.matching.base import MatchingResult
from methyl_match.preprocessing.methyl_network import MethylNetwork
from methyl_match.preprocessing.peak_network import PeakNetwork


class CSVFormatter(ResultFormatter):
    """
    Formatter for CSV tabular output.

    Creates CSV files with one row per assignment containing:
    - Peak information (index, assignment, 1H shift, 13C shift, intensity)
    - Methyl information (residue name, number, atom name, coordinates)
    - Assignment confidence and quality metrics

    Example CSV output:
        peak_id,peak_assignment,peak_h_shift,peak_c_shift,peak_intensity,
        methyl_residue,methyl_number,methyl_atom,methyl_x,methyl_y,methyl_z,
        confidence,quality,distance
        12,,0.85,21.5,1000.0,LEU,42,CD1,10.5,15.2,8.3,0.952,high,0.12
        7,V8-CG2,0.91,23.1,850.0,VAL,8,CG2,5.2,10.1,12.4,0.891,high,0.05
        ...
    """

    def __init__(
        self,
        include_unassigned: bool = False,
        include_metadata: bool = False,
        confidence_threshold: float = 0.0,
        include_coordinates: bool = True,
        include_chemical_shifts: bool = True,
    ):
        """
        Initialize CSV formatter.

        Args:
            include_unassigned: Create separate CSV for unassigned peaks
            include_metadata: Include metadata header in CSV
            confidence_threshold: Minimum confidence to include
            include_coordinates: Include XYZ coordinates in output
            include_chemical_shifts: Include chemical shifts
        """
        super().__init__(include_unassigned, include_metadata, confidence_threshold)
        self.include_coordinates = include_coordinates
        self.include_chemical_shifts = include_chemical_shifts

    def format(
        self,
        result: MatchingResult,
        structural_network: MethylNetwork,
        experimental_network: PeakNetwork,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> str:
        """
        Format matching result as CSV.

        Args:
            result: MatchingResult from matching algorithm
            structural_network: MethylNetwork from PDB structure
            experimental_network: PeakNetwork from HMQC/NOESY
            output_path: Optional path to save CSV file
            **kwargs: Additional options

        Returns:
            CSV content as string
        """
        rows = self._create_assignment_rows(
            result, structural_network, experimental_network
        )

        # Convert to CSV string
        output_lines = []

        # Add metadata as comments if requested
        if self.include_metadata and result.metadata:
            for key, value in result.metadata.items():
                output_lines.append(f"# {key}: {value}")
            output_lines.append("")

        # Create CSV header
        fieldnames = self._get_fieldnames()
        output_lines.append(",".join(fieldnames))

        # Add data rows
        for row in rows:
            csv_row = [str(row.get(field, "")) for field in fieldnames]
            output_lines.append(",".join(csv_row))

        output = "\n".join(output_lines)

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="") as f:
                f.write(output)

            # Create unassigned CSV if requested
            if self.include_unassigned and result.unassigned_peaks:
                unassigned_path = output_path.parent / f"{output_path.stem}_unassigned.csv"
                self._create_unassigned_csv(
                    result, experimental_network, unassigned_path
                )

        return output

    def _get_fieldnames(self) -> List[str]:
        """
        Get CSV field names based on configuration.

        Returns:
            List of field names for CSV header
        """
        fields = [
            "peak_id",
            "peak_assignment",
        ]

        if self.include_chemical_shifts:
            fields.extend([
                "peak_h_shift",
                "peak_c_shift",
                "peak_intensity",
            ])

        fields.extend([
            "methyl_residue",
            "methyl_number",
            "methyl_atom",
        ])

        if self.include_coordinates:
            fields.extend([
                "methyl_x",
                "methyl_y",
                "methyl_z",
            ])

        fields.extend([
            "confidence",
            "quality",
        ])

        return fields

    def _create_assignment_rows(
        self,
        result: MatchingResult,
        structural_network: MethylNetwork,
        experimental_network: PeakNetwork,
    ) -> List[Dict[str, any]]:
        """
        Create list of dictionaries representing CSV rows.

        Args:
            result: MatchingResult
            structural_network: MethylNetwork
            experimental_network: PeakNetwork

        Returns:
            List of row dictionaries
        """
        rows = []

        # Sort by confidence (descending)
        assignment_list = result.get_assignment_list()

        for exp_id, struct_id, confidence in assignment_list:
            if confidence < self.confidence_threshold:
                continue

            # Get peak and methyl data
            peak = experimental_network.hmqc_peaks[exp_id]
            methyl = structural_network.methyls[struct_id]

            # Build row
            row = {
                "peak_id": peak.index,
                "peak_assignment": peak.assignment if peak.assignment else "",
            }

            if self.include_chemical_shifts:
                row.update({
                    "peak_h_shift": f"{peak.h_shift:.3f}",
                    "peak_c_shift": f"{peak.c_shift:.2f}",
                    "peak_intensity": f"{peak.intensity:.1f}",
                })

            row.update({
                "methyl_residue": methyl.residue_name,
                "methyl_number": methyl.residue_number,
                "methyl_atom": methyl.atom_name,
            })

            if self.include_coordinates:
                row.update({
                    "methyl_x": f"{methyl.coordinates[0]:.3f}",
                    "methyl_y": f"{methyl.coordinates[1]:.3f}",
                    "methyl_z": f"{methyl.coordinates[2]:.3f}",
                })

            # Quality classification
            if confidence >= 0.7:
                quality = "high"
            elif confidence >= 0.5:
                quality = "medium"
            else:
                quality = "low"

            row.update({
                "confidence": f"{confidence:.4f}",
                "quality": quality,
            })

            rows.append(row)

        return rows

    def _create_unassigned_csv(
        self,
        result: MatchingResult,
        experimental_network: PeakNetwork,
        output_path: Path,
    ):
        """
        Create CSV file for unassigned peaks.

        Args:
            result: MatchingResult
            experimental_network: PeakNetwork
            output_path: Path to save unassigned CSV
        """
        fieldnames = [
            "peak_id",
            "peak_assignment",
            "peak_h_shift",
            "peak_c_shift",
            "peak_intensity",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for exp_id in result.unassigned_peaks:
                peak = experimental_network.hmqc_peaks[exp_id]
                writer.writerow({
                    "peak_id": peak.index,
                    "peak_assignment": peak.assignment if peak.assignment else "",
                    "peak_h_shift": f"{peak.h_shift:.3f}",
                    "peak_c_shift": f"{peak.c_shift:.2f}",
                    "peak_intensity": f"{peak.intensity:.1f}",
                })
