"""Output formatting for methyl assignment results."""

from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import torch
import numpy as np
from pathlib import Path


@dataclass
class AssignmentResult:
    """Single methyl assignment result.

    Attributes:
        peak_id: ID of the peak
        h_shift: Proton chemical shift (ppm)
        c_shift: Carbon chemical shift (ppm)
        assignment: Assigned methyl identifier (e.g., "L15CD1")
        confidence: Assignment confidence score (0-1)
        noe_completeness: Fraction of expected NOEs observed
        alternative_assignments: List of alternative assignments with scores
    """
    peak_id: int
    h_shift: float
    c_shift: float
    assignment: str
    confidence: float
    noe_completeness: Optional[float] = None
    alternative_assignments: Optional[List[Dict[str, any]]] = None


class OutputFormatter:
    """Formatter for methyl assignment results.

    Converts model predictions into human-readable text format and
    optionally generates visualization scripts.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        top_k_alternatives: int = 3
    ):
        """Initialize the output formatter.

        Args:
            confidence_threshold: Minimum confidence for reporting
            top_k_alternatives: Number of alternative assignments to report
        """
        self.confidence_threshold = confidence_threshold
        self.top_k_alternatives = top_k_alternatives

    def format_assignments(
        self,
        matching_matrix: torch.Tensor,
        peak_ids: List[int],
        peak_shifts: np.ndarray,
        methyl_names: List[str],
        peak_adjacency: Optional[np.ndarray] = None,
        methyl_adjacency: Optional[np.ndarray] = None
    ) -> List[AssignmentResult]:
        """Format assignment results from matching matrix.

        Args:
            matching_matrix: Predicted matching matrix [num_peaks, num_methyls]
            peak_ids: List of peak IDs
            peak_shifts: Peak chemical shifts [num_peaks, 2] (H, C)
            methyl_names: List of methyl identifiers
            peak_adjacency: Optional peak network adjacency
            methyl_adjacency: Optional methyl network adjacency

        Returns:
            List of AssignmentResult objects
        """
        results = []

        # Convert to numpy if needed
        if isinstance(matching_matrix, torch.Tensor):
            matching_matrix = matching_matrix.detach().cpu().numpy()

        num_peaks = matching_matrix.shape[0]

        for i in range(num_peaks):
            peak_id = peak_ids[i]
            h_shift = float(peak_shifts[i, 0])
            c_shift = float(peak_shifts[i, 1])

            # Get top assignments
            scores = matching_matrix[i, :]
            top_indices = np.argsort(scores)[::-1][:self.top_k_alternatives + 1]
            top_scores = scores[top_indices]

            # Primary assignment
            assignment_idx = top_indices[0]
            assignment = methyl_names[assignment_idx]
            confidence = float(top_scores[0])

            # Alternative assignments
            alternatives = []
            for j in range(1, len(top_indices)):
                alt_idx = top_indices[j]
                alt_score = float(top_scores[j])

                # Only include if above a minimum threshold
                if alt_score > 0.1:
                    alternatives.append({
                        'assignment': methyl_names[alt_idx],
                        'confidence': alt_score
                    })

            # Compute NOE completeness if adjacency matrices provided
            noe_completeness = None
            if peak_adjacency is not None and methyl_adjacency is not None:
                noe_completeness = self._compute_noe_completeness(
                    peak_idx=i,
                    methyl_idx=assignment_idx,
                    peak_adj=peak_adjacency,
                    methyl_adj=methyl_adjacency,
                    matching=matching_matrix
                )

            result = AssignmentResult(
                peak_id=peak_id,
                h_shift=h_shift,
                c_shift=c_shift,
                assignment=assignment,
                confidence=confidence,
                noe_completeness=noe_completeness,
                alternative_assignments=alternatives if alternatives else None
            )

            results.append(result)

        return results

    def _compute_noe_completeness(
        self,
        peak_idx: int,
        methyl_idx: int,
        peak_adj: np.ndarray,
        methyl_adj: np.ndarray,
        matching: np.ndarray
    ) -> float:
        """Compute NOE completeness score.

        Fraction of structurally expected NOEs that are observed.

        Args:
            peak_idx: Index of the peak
            methyl_idx: Index of assigned methyl
            peak_adj: Peak adjacency matrix
            methyl_adj: Methyl adjacency matrix
            matching: Current matching matrix

        Returns:
            Completeness score (0-1)
        """
        # Find structurally expected neighbors (methyls close to assigned methyl)
        expected_neighbors = np.where(methyl_adj[methyl_idx, :] > 0)[0]

        if len(expected_neighbors) == 0:
            return 1.0  # No expected neighbors

        # Find observed peak neighbors
        observed_neighbors = np.where(peak_adj[peak_idx, :] > 0)[0]

        if len(observed_neighbors) == 0:
            return 0.0  # No observed NOEs

        # Count how many expected neighbors have corresponding observed peaks
        matched_neighbors = 0
        for methyl_neighbor_idx in expected_neighbors:
            # Find peaks likely assigned to this methyl neighbor
            peak_scores = matching[:, methyl_neighbor_idx]

            # Check if any observed neighbor peak is assigned to this methyl
            for peak_neighbor_idx in observed_neighbors:
                if peak_scores[peak_neighbor_idx] > 0.3:  # Threshold
                    matched_neighbors += 1
                    break

        completeness = matched_neighbors / len(expected_neighbors)
        return completeness

    def to_text(
        self,
        results: List[AssignmentResult],
        include_alternatives: bool = True,
        only_confident: bool = False
    ) -> str:
        """Convert results to text format.

        Args:
            results: List of AssignmentResult objects
            include_alternatives: Whether to include alternative assignments
            only_confident: Only include assignments above confidence threshold

        Returns:
            Formatted text string
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("NMR Methyl Assignment Results")
        lines.append("=" * 80)
        lines.append("")

        # Table header
        header = f"{'Peak_ID':<8} {'H_shift':<10} {'C_shift':<10} {'Assignment':<15} {'Confidence':<12}"
        if any(r.noe_completeness is not None for r in results):
            header += f"{'NOE_Compl':<12}"
        lines.append(header)
        lines.append("-" * len(header))

        # Results
        for result in results:
            # Skip low-confidence if requested
            if only_confident and result.confidence < self.confidence_threshold:
                continue

            line = (f"{result.peak_id:<8} "
                   f"{result.h_shift:<10.3f} "
                   f"{result.c_shift:<10.3f} "
                   f"{result.assignment:<15} "
                   f"{result.confidence:<12.3f}")

            if result.noe_completeness is not None:
                line += f"{result.noe_completeness:<12.3f}"

            lines.append(line)

            # Add alternatives if requested
            if include_alternatives and result.alternative_assignments:
                for alt in result.alternative_assignments:
                    alt_line = (f"{'':8} {'':10} {'':10} "
                              f"  → {alt['assignment']:<13} "
                              f"{alt['confidence']:<12.3f}")
                    lines.append(alt_line)

        # Summary
        lines.append("")
        lines.append("=" * 80)
        confident_count = sum(1 for r in results if r.confidence >= self.confidence_threshold)
        lines.append(f"Total assignments: {len(results)}")
        lines.append(f"High-confidence assignments (≥{self.confidence_threshold}): {confident_count}")

        if any(r.noe_completeness is not None for r in results):
            avg_completeness = np.mean([r.noe_completeness for r in results if r.noe_completeness is not None])
            lines.append(f"Average NOE completeness: {avg_completeness:.3f}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dataframe(self, results: List[AssignmentResult]) -> pd.DataFrame:
        """Convert results to pandas DataFrame.

        Args:
            results: List of AssignmentResult objects

        Returns:
            DataFrame with assignment information
        """
        data = {
            'peak_id': [r.peak_id for r in results],
            'H_shift': [r.h_shift for r in results],
            'C_shift': [r.c_shift for r in results],
            'assignment': [r.assignment for r in results],
            'confidence': [r.confidence for r in results],
        }

        # Add NOE completeness if available
        if any(r.noe_completeness is not None for r in results):
            data['noe_completeness'] = [
                r.noe_completeness if r.noe_completeness is not None else np.nan
                for r in results
            ]

        return pd.DataFrame(data)

    def to_csv(
        self,
        results: List[AssignmentResult],
        output_file: str,
        include_alternatives: bool = False
    ):
        """Save results to CSV file.

        Args:
            results: List of AssignmentResult objects
            output_file: Path to output CSV file
            include_alternatives: Whether to include alternative assignments
        """
        if not include_alternatives:
            df = self.to_dataframe(results)
            df.to_csv(output_file, index=False)
        else:
            # Flatten alternatives into separate rows
            rows = []
            for result in results:
                # Primary assignment
                row = {
                    'peak_id': result.peak_id,
                    'H_shift': result.h_shift,
                    'C_shift': result.c_shift,
                    'assignment': result.assignment,
                    'confidence': result.confidence,
                    'rank': 1
                }
                if result.noe_completeness is not None:
                    row['noe_completeness'] = result.noe_completeness
                rows.append(row)

                # Alternative assignments
                if result.alternative_assignments:
                    for rank, alt in enumerate(result.alternative_assignments, start=2):
                        alt_row = {
                            'peak_id': result.peak_id,
                            'H_shift': result.h_shift,
                            'C_shift': result.c_shift,
                            'assignment': alt['assignment'],
                            'confidence': alt['confidence'],
                            'rank': rank
                        }
                        rows.append(alt_row)

            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)

        print(f"Results saved to {output_file}")

    def generate_pymol_script(
        self,
        results: List[AssignmentResult],
        pdb_file: str,
        output_file: str,
        confidence_threshold: Optional[float] = None
    ):
        """Generate PyMOL visualization script.

        Args:
            results: List of AssignmentResult objects
            pdb_file: Path to PDB file
            output_file: Path to output PyMOL script
            confidence_threshold: Optional confidence threshold for coloring
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        lines = []
        lines.append("# PyMOL script for visualizing methyl assignments")
        lines.append(f"load {pdb_file}")
        lines.append("hide everything")
        lines.append("show cartoon")
        lines.append("")

        # Color by confidence
        lines.append("# Color assignments by confidence")
        for result in results:
            # Parse assignment to get residue and atom
            assignment = result.assignment
            # Format: L15CD1 -> residue 15, atom CD1

            # Extract residue number
            res_num = ''.join(filter(str.isdigit, assignment))
            atom_name = ''.join(filter(str.isalpha, assignment))[1:]  # Skip first letter (residue type)

            if not res_num or not atom_name:
                continue

            # Color based on confidence
            if result.confidence >= 0.8:
                color = "green"
            elif result.confidence >= confidence_threshold:
                color = "yellow"
            else:
                color = "red"

            lines.append(f"select assigned_{result.peak_id}, resi {res_num} and name {atom_name}")
            lines.append(f"show spheres, assigned_{result.peak_id}")
            lines.append(f"color {color}, assigned_{result.peak_id}")

        lines.append("")
        lines.append("# Legend")
        lines.append("# Green: High confidence (≥0.8)")
        lines.append("# Yellow: Medium confidence (≥{})".format(confidence_threshold))
        lines.append("# Red: Low confidence")

        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

        print(f"PyMOL script saved to {output_file}")
