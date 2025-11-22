"""
HMQC peak list parser with multi-format support.

This module provides functionality to parse HMQC (Heteronuclear Multiple Quantum
Correlation) peak lists in various formats including XEASY, NMRPipe, Sparky, and CSV.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HMQCPeak:
    """
    Represents an HMQC peak (methyl 1H-13C correlation).

    Attributes:
        index: Peak index/number
        h_shift: Proton chemical shift (ppm)
        c_shift: Carbon-13 chemical shift (ppm)
        intensity: Peak intensity/volume
        assignment: Methyl group assignment (e.g., 'L8-CD1')
    """
    index: int
    h_shift: float
    c_shift: float
    intensity: float
    assignment: Optional[str] = None

    def __str__(self) -> str:
        """String representation of the HMQC peak."""
        if self.assignment:
            return f"HMQC Peak {self.index}: {self.assignment} (1H={self.h_shift:.3f}, 13C={self.c_shift:.2f}, I={self.intensity:.0f})"
        else:
            return f"HMQC Peak {self.index}: (1H={self.h_shift:.3f}, 13C={self.c_shift:.2f}, I={self.intensity:.0f})"


class HMQCParser:
    """
    Parser for HMQC peak lists supporting multiple file formats.

    Supported formats:
    - XEASY: Tab/space-separated with optional comments
    - NMRPipe: NMRPipe peak list format
    - Sparky: Sparky assignment format
    - CSV: Comma-separated values

    The parser automatically detects the format based on file content.

    Example:
        >>> parser = HMQCParser("hmqc_peaks.txt")
        >>> peaks = parser.parse()
        >>> print(f"Found {len(peaks)} HMQC peaks")
        >>> for peak in peaks[:3]:
        ...     print(f"{peak.assignment}: 1H={peak.h_shift:.3f}, 13C={peak.c_shift:.2f}")
    """

    def __init__(self, file_path: str):
        """
        Initialize the HMQC parser.

        Args:
            file_path: Path to the HMQC peak list file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"HMQC file not found: {file_path}")

        self.format: Optional[str] = None
        logger.info(f"Initialized HMQCParser for {file_path}")

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

        # Check for Sparky format (specific column headers)
        first_line = lines[0]
        if 'Assignment' in first_line and any(kw in first_line for kw in ['w1', 'w2', 'Height']):
            logger.info("Detected Sparky format")
            return 'sparky'

        # Check for NMRPipe format
        content = ''.join(lines[:10])
        if 'VARS' in content or 'FORMAT' in content:
            logger.info("Detected NMRPipe format")
            return 'nmrpipe'

        # Default to XEASY
        logger.info("Detected XEASY format (default)")
        return 'xeasy'

    def parse(self, format: Optional[str] = None) -> List[HMQCPeak]:
        """
        Parse the HMQC peak list.

        Args:
            format: Force specific format ('xeasy', 'csv', 'sparky', 'nmrpipe').
                   If None, auto-detect format.

        Returns:
            List of HMQCPeak objects

        Example:
            >>> parser = HMQCParser("hmqc.txt")
            >>> peaks = parser.parse()
            >>> assigned_peaks = [p for p in peaks if p.assignment]
            >>> print(f"Assigned: {len(assigned_peaks)}/{len(peaks)}")
        """
        if format is None:
            self.format = self.detect_format()
        else:
            self.format = format.lower()

        logger.info(f"Parsing HMQC file as {self.format} format")

        if self.format == 'csv':
            return self._parse_csv()
        elif self.format == 'sparky':
            return self._parse_sparky()
        elif self.format == 'nmrpipe':
            return self._parse_nmrpipe()
        else:  # xeasy or default
            return self._parse_xeasy()

    def _parse_xeasy(self) -> List[HMQCPeak]:
        """Parse XEASY format peak list."""
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

                    # Parse fields (Index, C13, H1, Intensity, [Assignment])
                    index = int(parts[0])
                    c_shift = float(parts[1])
                    h_shift = float(parts[2])
                    intensity = float(parts[3])

                    # Parse assignment if present
                    assignment = None
                    if len(parts) > 4:
                        assignment = parts[4].strip()

                    peak = HMQCPeak(
                        index=index,
                        h_shift=h_shift,
                        c_shift=c_shift,
                        intensity=intensity,
                        assignment=assignment
                    )
                    peaks.append(peak)

                except (ValueError, IndexError) as e:
                    logger.warning(f"Line {line_num}: Parse error - {e}")
                    continue

        logger.info(f"Parsed {len(peaks)} peaks from XEASY format")
        return peaks

    def _parse_csv(self) -> List[HMQCPeak]:
        """Parse CSV format peak list."""
        try:
            df = pd.read_csv(self.file_path)

            # Common column name variations
            index_col = self._find_column(df, ['Index', 'index', 'ID', 'id', 'Peak'])
            h_col = self._find_column(df, ['H1_ppm', 'h_shift', 'H_shift', 'w2', 'w1', '1H', 'H1'])
            c_col = self._find_column(df, ['C13_ppm', 'c_shift', 'C_shift', '13C', 'C13', 'w1', 'w2'])
            intensity_col = self._find_column(df, ['Intensity', 'intensity', 'Volume', 'volume', 'Height', 'height'])
            assignment_col = self._find_column(df, ['Assignment', 'assignment', 'Assign', 'Label'], required=False)

            peaks = []
            for idx, row in df.iterrows():
                assignment = None
                if assignment_col and pd.notna(row[assignment_col]):
                    assignment = str(row[assignment_col]).strip()

                peak = HMQCPeak(
                    index=int(row[index_col]),
                    h_shift=float(row[h_col]),
                    c_shift=float(row[c_col]),
                    intensity=float(row[intensity_col]),
                    assignment=assignment
                )
                peaks.append(peak)

            logger.info(f"Parsed {len(peaks)} peaks from CSV format")
            return peaks

        except Exception as e:
            logger.error(f"Failed to parse CSV format: {e}")
            raise

    def _parse_sparky(self) -> List[HMQCPeak]:
        """Parse Sparky format peak list."""
        peaks = []

        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        # Skip header line(s)
        data_lines = [l for l in lines if l.strip() and not l.strip().startswith('Assignment')]

        for line_num, line in enumerate(data_lines, 1):
            try:
                parts = line.split()
                if len(parts) < 3:
                    continue

                # Sparky format: Assignment w1 w2 [Height]
                # Height/intensity is optional
                assignment = parts[0].strip() if parts[0] else None
                w1 = float(parts[1])  # Often C13
                w2 = float(parts[2])  # Often H1

                # Intensity is optional (default to 1.0 if missing)
                if len(parts) >= 4:
                    intensity = float(parts[3])
                else:
                    intensity = 1.0
                    logger.debug(f"Line {line_num}: No intensity column, using default 1.0")

                peak = HMQCPeak(
                    index=line_num,
                    h_shift=w2,
                    c_shift=w1,
                    intensity=intensity,
                    assignment=assignment
                )
                peaks.append(peak)

            except (ValueError, IndexError) as e:
                logger.warning(f"Line {line_num}: Parse error - {e}")
                continue

        logger.info(f"Parsed {len(peaks)} peaks from Sparky format")
        return peaks

    def _parse_nmrpipe(self) -> List[HMQCPeak]:
        """Parse NMRPipe format peak list."""
        # NMRPipe format is complex; this is a simplified parser
        logger.warning("NMRPipe format support is limited. Consider using nmrglue for full support.")

        # Fall back to XEASY-style parsing
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

    def get_peak_by_assignment(self, assignment: str, peaks: Optional[List[HMQCPeak]] = None) -> Optional[HMQCPeak]:
        """
        Find a specific peak by its assignment.

        Args:
            assignment: Assignment label to find (e.g., 'L8-CD1')
            peaks: List of peaks to search. If None, parse from file.

        Returns:
            HMQCPeak if found, None otherwise

        Example:
            >>> parser = HMQCParser("hmqc.txt")
            >>> peak = parser.get_peak_by_assignment("L8-CD1")
            >>> if peak:
            ...     print(f"Found: 1H={peak.h_shift:.3f}, 13C={peak.c_shift:.2f}")
        """
        if peaks is None:
            peaks = self.parse()

        for peak in peaks:
            if peak.assignment == assignment:
                return peak

        logger.warning(f"Peak with assignment '{assignment}' not found")
        return None

    def get_chemical_shifts(self, peaks: Optional[List[HMQCPeak]] = None) -> Dict[str, tuple]:
        """
        Get a dictionary mapping assignments to chemical shifts.

        Args:
            peaks: List of peaks. If None, parse from file.

        Returns:
            Dictionary mapping assignment -> (h_shift, c_shift)

        Example:
            >>> parser = HMQCParser("hmqc.txt")
            >>> shifts = parser.get_chemical_shifts()
            >>> print(shifts['L8-CD1'])  # (0.879, 21.34)
        """
        if peaks is None:
            peaks = self.parse()

        shifts = {}
        for peak in peaks:
            if peak.assignment:
                shifts[peak.assignment] = (peak.h_shift, peak.c_shift)

        logger.info(f"Extracted chemical shifts for {len(shifts)} assignments")
        return shifts

    def filter_by_intensity(self, peaks: List[HMQCPeak], min_intensity: float) -> List[HMQCPeak]:
        """
        Filter peaks by minimum intensity threshold.

        Args:
            peaks: List of peaks to filter
            min_intensity: Minimum intensity threshold

        Returns:
            Filtered list of peaks

        Example:
            >>> parser = HMQCParser("hmqc.txt")
            >>> all_peaks = parser.parse()
            >>> strong_peaks = parser.filter_by_intensity(all_peaks, 100000)
            >>> print(f"Strong peaks: {len(strong_peaks)}/{len(all_peaks)}")
        """
        filtered = [p for p in peaks if p.intensity >= min_intensity]
        logger.info(f"Filtered {len(filtered)}/{len(peaks)} peaks with intensity >= {min_intensity}")
        return filtered

    def get_unassigned_peaks(self, peaks: Optional[List[HMQCPeak]] = None) -> List[HMQCPeak]:
        """
        Get all peaks that don't have assignments.

        Args:
            peaks: List of peaks. If None, parse from file.

        Returns:
            List of unassigned peaks

        Example:
            >>> parser = HMQCParser("hmqc.txt")
            >>> unassigned = parser.get_unassigned_peaks()
            >>> print(f"Unassigned peaks: {len(unassigned)}")
        """
        if peaks is None:
            peaks = self.parse()

        unassigned = [p for p in peaks if not p.assignment]
        logger.info(f"Found {len(unassigned)} unassigned peaks")
        return unassigned

    def match_chemical_shifts(self, h_shift: float, c_shift: float,
                            h_tolerance: float = 0.05, c_tolerance: float = 0.5,
                            peaks: Optional[List[HMQCPeak]] = None) -> List[HMQCPeak]:
        """
        Find peaks matching given chemical shifts within tolerance.

        Args:
            h_shift: Target proton chemical shift (ppm)
            c_shift: Target carbon chemical shift (ppm)
            h_tolerance: Tolerance for proton shift (ppm)
            c_tolerance: Tolerance for carbon shift (ppm)
            peaks: List of peaks to search. If None, parse from file.

        Returns:
            List of matching peaks

        Example:
            >>> parser = HMQCParser("hmqc.txt")
            >>> matches = parser.match_chemical_shifts(0.88, 21.3, h_tolerance=0.01, c_tolerance=0.5)
            >>> for peak in matches:
            ...     print(peak)
        """
        if peaks is None:
            peaks = self.parse()

        matches = []
        for peak in peaks:
            h_diff = abs(peak.h_shift - h_shift)
            c_diff = abs(peak.c_shift - c_shift)

            if h_diff <= h_tolerance and c_diff <= c_tolerance:
                matches.append(peak)

        logger.info(f"Found {len(matches)} peaks matching shifts (1H={h_shift}±{h_tolerance}, 13C={c_shift}±{c_tolerance})")
        return matches
