"""
NOESY peak list parser with multi-format support.

This module provides functionality to parse 13C-13C-1H methyl-methyl NOESY
(Nuclear Overhauser Effect Spectroscopy) peak lists in various formats including
XEASY, NMRPipe, Sparky, and CSV.

For methyl-methyl assignment, the expected format is:
- w1: 13C chemical shift of first methyl (ppm)
- w2: 13C chemical shift of second methyl (ppm)
- w3: 1H chemical shift (ppm)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class NOEPeak:
    """
    Represents a NOE cross-peak between two methyl groups (13C-13C-1H NOESY).

    Attributes:
        index: Peak index/number
        w1: 13C chemical shift of first methyl (ppm)
        w2: 13C chemical shift of second methyl (ppm)
        w3: 1H chemical shift (ppm)
        intensity: Peak intensity/volume
        assignment1: Assignment for first methyl (e.g., 'L8-CD1')
        assignment2: Assignment for second methyl (e.g., 'V17-CG1')
    """
    index: int
    w1: float
    w2: float
    w3: Optional[float]
    intensity: float
    assignment1: Optional[str] = None
    assignment2: Optional[str] = None

    def __str__(self) -> str:
        """String representation of the NOE peak."""
        if self.assignment1 and self.assignment2:
            return f"NOE Peak {self.index}: {self.assignment1} <-> {self.assignment2} (I={self.intensity:.0f})"
        else:
            dims = f"({self.w1:.3f}, {self.w2:.3f}"
            if self.w3 is not None:
                dims += f", {self.w3:.3f}"
            dims += ")"
            return f"NOE Peak {self.index}: {dims} (I={self.intensity:.0f})"


class NOESYParser:
    """
    Parser for 13C-13C-1H methyl-methyl NOESY peak lists supporting multiple file formats.

    This parser is designed for methyl-methyl NOE experiments where:
    - Dimension 1 (w1): 13C of first methyl
    - Dimension 2 (w2): 13C of second methyl
    - Dimension 3 (w3): 1H (typically)

    Supported formats:
    - XEASY: Tab/space-separated with optional comments
    - NMRPipe: NMRPipe peak list format
    - Sparky: Sparky assignment format
    - CSV: Comma-separated values

    The parser automatically detects the format based on file content.

    Example:
        >>> parser = NOESYParser("methyl_noesy_peaks.txt")
        >>> peaks = parser.parse()
        >>> print(f"Found {len(peaks)} methyl-methyl NOE cross-peaks")
    """

    def __init__(self, file_path: str):
        """
        Initialize the NOESY parser.

        Args:
            file_path: Path to the NOESY peak list file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"NOESY file not found: {file_path}")

        self.format: Optional[str] = None
        logger.info(f"Initialized NOESYParser for {file_path}")

    def detect_format(self) -> str:
        """
        Auto-detect the file format based on content.

        Returns:
            Format string: 'xeasy', 'nmrpipe', 'sparky', or 'csv'

        Raises:
            ValueError: If format cannot be determined
        """
        with open(self.file_path, 'r') as f:
            # Read first few non-comment lines
            lines = []
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines.append(line)
                    if len(lines) >= 5:
                        break

        if not lines:
            raise ValueError("File appears to be empty or contains only comments")

        # Check for CSV (comma-separated)
        if ',' in lines[0] and lines[0].count(',') >= 3:
            logger.info("Detected CSV format")
            return 'csv'

        # Check for Sparky format (specific column headers and alignment)
        first_line = lines[0]
        if 'Assignment' in first_line and 'Height' in first_line:
            logger.info("Detected Sparky format")
            return 'sparky'

        # Check for NMRPipe format (has specific keywords)
        content = ''.join(lines[:10])
        if 'VARS' in content or 'FORMAT' in content:
            logger.info("Detected NMRPipe format")
            return 'nmrpipe'

        # Default to XEASY (most common for simple peak lists)
        logger.info("Detected XEASY format (default)")
        return 'xeasy'

    def parse(self, format: Optional[str] = None) -> List[NOEPeak]:
        """
        Parse the NOESY peak list.

        Args:
            format: Force specific format ('xeasy', 'csv', 'sparky', 'nmrpipe').
                   If None, auto-detect format.

        Returns:
            List of NOEPeak objects

        Example:
            >>> parser = NOESYParser("peaks.txt")
            >>> peaks = parser.parse()
            >>> for peak in peaks[:3]:
            ...     print(peak)
        """
        if format is None:
            self.format = self.detect_format()
        else:
            self.format = format.lower()

        logger.info(f"Parsing NOESY file as {self.format} format")

        if self.format == 'csv':
            return self._parse_csv()
        elif self.format == 'sparky':
            return self._parse_sparky()
        elif self.format == 'nmrpipe':
            return self._parse_nmrpipe()
        else:  # xeasy or default
            return self._parse_xeasy()

    def _parse_xeasy(self) -> List[NOEPeak]:
        """Parse XEASY format peak list (also handles Sparky-like format without header)."""
        peaks = []

        with open(self.file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                try:
                    # Split by whitespace
                    parts = line.split()

                    if len(parts) < 4:
                        logger.warning(f"Line {line_num}: Not enough columns, skipping")
                        continue

                    # Try to determine format: XEASY (index first) vs Sparky-like (assignment first)
                    # Try to parse first column as integer
                    try:
                        index = int(parts[0])
                        # XEASY format: index w1 w2 w3 intensity [assignments]
                        start_idx = 0
                        assignment_text = None
                    except ValueError:
                        # Sparky-like format: assignment w1 w2 w3 intensity
                        index = line_num
                        start_idx = -1  # Shift by -1 since assignment is first
                        assignment_text = parts[0]

                    # Parse chemical shifts and intensity
                    if start_idx == -1:
                        # Sparky-like: assignment w1 w2 w3 intensity
                        w1 = float(parts[1])
                        w2 = float(parts[2])
                        w3 = float(parts[3]) if len(parts) > 3 else None
                        intensity = float(parts[4]) if len(parts) > 4 else 1.0
                    else:
                        # XEASY: index w1 w2 w3 intensity [assignments]
                        w1 = float(parts[1])
                        w2 = float(parts[2])

                        # Check if 3D or 2D
                        try:
                            w3_candidate = float(parts[3])
                            intensity_idx = 4
                            w3 = w3_candidate
                        except (ValueError, IndexError):
                            w3 = None
                            intensity_idx = 3

                        # Get intensity
                        if len(parts) > intensity_idx:
                            intensity = float(parts[intensity_idx])
                        else:
                            intensity = 1.0  # Default

                        # Parse assignments if present (after intensity)
                        if len(parts) > intensity_idx + 1:
                            assignment_text = ' '.join(parts[intensity_idx + 1:])

                    # Parse assignment text
                    assignment1 = None
                    assignment2 = None
                    if assignment_text:
                        # Assignments might be in format "L8-CD1-V17-CG1" or "L8-CD1 <-> V17-CG1" or just "?"
                        assignments = re.split(r'\s*<?-?>\s*', assignment_text)
                        if len(assignments) >= 2:
                            assignment1 = assignments[0].strip()
                            assignment2 = assignments[1].strip()
                        elif len(assignments) == 1:
                            assignment1 = assignments[0].strip()

                    peak = NOEPeak(
                        index=index,
                        w1=w1,
                        w2=w2,
                        w3=w3,
                        intensity=intensity,
                        assignment1=assignment1,
                        assignment2=assignment2
                    )
                    peaks.append(peak)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Line {line_num}: Parse error - {e}")
                    continue

        logger.info(f"Parsed {len(peaks)} peaks from XEASY format")
        return peaks

    def _parse_csv(self) -> List[NOEPeak]:
        """Parse CSV format peak list."""
        try:
            df = pd.read_csv(self.file_path)

            # Common column name variations
            index_col = self._find_column(df, ['Index', 'index', 'ID', 'id', 'Peak'])
            w1_col = self._find_column(df, ['w1', 'W1', 'H1', 'F1', 'dim1'])
            w2_col = self._find_column(df, ['w2', 'W2', 'H2', 'F2', 'dim2'])
            w3_col = self._find_column(df, ['w3', 'W3', 'H3', 'F3', 'C13', 'dim3'], required=False)
            intensity_col = self._find_column(df, ['Intensity', 'intensity', 'Volume', 'volume', 'Height', 'height'])
            assign1_col = self._find_column(df, ['Assignment1', 'Assign1', 'assignment1'], required=False)
            assign2_col = self._find_column(df, ['Assignment2', 'Assign2', 'assignment2'], required=False)

            # If no separate assignment columns, check for single assignment column
            if assign1_col is None:
                assign_col = self._find_column(df, ['Assignment', 'assignment', 'Assign'], required=False)
            else:
                assign_col = None

            peaks = []
            for idx, row in df.iterrows():
                w3 = row[w3_col] if w3_col else None

                # Parse assignments
                assignment1 = None
                assignment2 = None
                if assign1_col and assign2_col:
                    assignment1 = str(row[assign1_col]) if pd.notna(row[assign1_col]) else None
                    assignment2 = str(row[assign2_col]) if pd.notna(row[assign2_col]) else None
                elif assign_col:
                    # Split single assignment column
                    assign_text = str(row[assign_col]) if pd.notna(row[assign_col]) else ""
                    assignments = re.split(r'\s*<?-?>\s*', assign_text)
                    if len(assignments) >= 2:
                        assignment1 = assignments[0].strip()
                        assignment2 = assignments[1].strip()

                peak = NOEPeak(
                    index=int(row[index_col]),
                    w1=float(row[w1_col]),
                    w2=float(row[w2_col]),
                    w3=float(w3) if w3 is not None and pd.notna(w3) else None,
                    intensity=float(row[intensity_col]),
                    assignment1=assignment1,
                    assignment2=assignment2
                )
                peaks.append(peak)

            logger.info(f"Parsed {len(peaks)} peaks from CSV format")
            return peaks

        except Exception as e:
            logger.error(f"Failed to parse CSV format: {e}")
            raise

    def _parse_sparky(self) -> List[NOEPeak]:
        """Parse Sparky format peak list."""
        peaks = []

        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        # Skip header line(s)
        data_lines = [l for l in lines if l.strip() and not l.strip().startswith('Assignment')]

        for line_num, line in enumerate(data_lines, 1):
            try:
                parts = line.split()
                if len(parts) < 4:
                    continue

                # Sparky format: Assignment w1 w2 [w3] Height
                assignment_text = parts[0]
                w1 = float(parts[1])
                w2 = float(parts[2])

                # Check if 3D
                try:
                    w3 = float(parts[3])
                    intensity = float(parts[4])
                except (ValueError, IndexError):
                    w3 = None
                    intensity = float(parts[3])

                # Parse assignment (format: "L8CD1-V17CG1" or "L8CD1")
                assignments = re.split(r'[-_]', assignment_text)
                assignment1 = assignments[0] if len(assignments) > 0 else None
                assignment2 = assignments[1] if len(assignments) > 1 else None

                peak = NOEPeak(
                    index=line_num,
                    w1=w1,
                    w2=w2,
                    w3=w3,
                    intensity=intensity,
                    assignment1=assignment1,
                    assignment2=assignment2
                )
                peaks.append(peak)

            except (ValueError, IndexError) as e:
                logger.warning(f"Line {line_num}: Parse error - {e}")
                continue

        logger.info(f"Parsed {len(peaks)} peaks from Sparky format")
        return peaks

    def _parse_nmrpipe(self) -> List[NOEPeak]:
        """Parse NMRPipe format peak list."""
        # NMRPipe format is complex; this is a simplified parser
        # For full support, consider using nmrglue library
        logger.warning("NMRPipe format support is limited. Consider using nmrglue for full support.")

        # Fall back to XEASY-style parsing for now
        return self._parse_xeasy()

    def _find_column(self, df: pd.DataFrame, names: List[str], required: bool = True) -> Optional[str]:
        """
        Find a column in dataframe by checking multiple possible names.

        Args:
            df: DataFrame to search
            names: List of possible column names
            required: If True, raise error if not found

        Returns:
            Column name if found, None otherwise

        Raises:
            ValueError: If required=True and column not found
        """
        for name in names:
            if name in df.columns:
                return name

        if required:
            raise ValueError(f"Could not find required column. Tried: {names}")
        return None

    def get_peak_by_index(self, index: int, peaks: Optional[List[NOEPeak]] = None) -> Optional[NOEPeak]:
        """
        Find a specific peak by its index.

        Args:
            index: Peak index to find
            peaks: List of peaks to search. If None, parse from file.

        Returns:
            NOEPeak if found, None otherwise
        """
        if peaks is None:
            peaks = self.parse()

        for peak in peaks:
            if peak.index == index:
                return peak

        logger.warning(f"Peak with index {index} not found")
        return None

    def filter_by_intensity(self, peaks: List[NOEPeak], min_intensity: float) -> List[NOEPeak]:
        """
        Filter peaks by minimum intensity threshold.

        Args:
            peaks: List of peaks to filter
            min_intensity: Minimum intensity threshold

        Returns:
            Filtered list of peaks

        Example:
            >>> parser = NOESYParser("peaks.txt")
            >>> all_peaks = parser.parse()
            >>> strong_peaks = parser.filter_by_intensity(all_peaks, 50000)
            >>> print(f"Strong peaks: {len(strong_peaks)}/{len(all_peaks)}")
        """
        filtered = [p for p in peaks if p.intensity >= min_intensity]
        logger.info(f"Filtered {len(filtered)}/{len(peaks)} peaks with intensity >= {min_intensity}")
        return filtered

    def get_correlation_matrix(self, peaks: List[NOEPeak], labels: List[str]) -> np.ndarray:
        """
        Create a correlation matrix from NOE peaks.

        Args:
            peaks: List of NOE peaks
            labels: List of methyl labels to create matrix for

        Returns:
            Correlation matrix where entry [i,j] is the NOE intensity
            between methyls i and j (0 if no NOE observed)

        Example:
            >>> parser = NOESYParser("peaks.txt")
            >>> peaks = parser.parse()
            >>> labels = ['L8-CD1', 'L8-CD2', 'V17-CG1', 'V17-CG2']
            >>> matrix = parser.get_correlation_matrix(peaks, labels)
        """
        n = len(labels)
        matrix = np.zeros((n, n))

        # Create label to index mapping
        label_to_idx = {label: i for i, label in enumerate(labels)}

        for peak in peaks:
            if peak.assignment1 and peak.assignment2:
                idx1 = label_to_idx.get(peak.assignment1)
                idx2 = label_to_idx.get(peak.assignment2)

                if idx1 is not None and idx2 is not None:
                    # Use maximum intensity if multiple NOEs between same pair
                    matrix[idx1, idx2] = max(matrix[idx1, idx2], peak.intensity)
                    matrix[idx2, idx1] = matrix[idx1, idx2]

        logger.info(f"Created {n}x{n} correlation matrix from {len(peaks)} peaks")
        return matrix
