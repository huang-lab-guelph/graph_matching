"""Parsers for NMR spectroscopy data files (NOESY, HMQC)."""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class Peak:
    """Represents a single NMR peak.

    Attributes:
        peak_id: Unique identifier for the peak
        shifts: Chemical shifts in each dimension (ppm)
        intensity: Peak intensity/volume
        assignment: Optional assignment string (e.g., "L15HD1-L15CD1")
        linewidths: Optional linewidths in each dimension
        confidence: Optional confidence score (0-1)
    """
    peak_id: int
    shifts: np.ndarray
    intensity: float
    assignment: Optional[str] = None
    linewidths: Optional[np.ndarray] = None
    confidence: float = 1.0


@dataclass
class NOECrosspeaks:
    """Represents NOE cross-peaks between two methyl groups.

    Attributes:
        peak1_id: ID of the first peak
        peak2_id: ID of the second peak
        noe_intensity: NOE intensity
        confidence: Confidence score accounting for overlap, etc.
    """
    peak1_id: int
    peak2_id: int
    noe_intensity: float
    confidence: float = 1.0


class HMQCParser:
    """Parser for 2D 1H,13C-HMQC peak lists.

    Supports multiple formats:
    - XEASY (2D)
    - NMRPipe tab format
    - Sparky format
    - Simple CSV format
    """

    def __init__(self):
        """Initialize the HMQC parser."""
        self.peaks: List[Peak] = []
        self.format: Optional[str] = None

    def parse(self, filepath: str, format: Optional[str] = None) -> List[Peak]:
        """Parse an HMQC peak list file.

        Args:
            filepath: Path to the peak list file
            format: Format specification ('xeasy', 'nmrpipe', 'sparky', 'csv')
                   If None, will attempt auto-detection

        Returns:
            List of Peak objects
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect format if not specified
        if format is None:
            format = self._detect_format(filepath)

        self.format = format

        # Parse based on format
        if format == "xeasy":
            self.peaks = self._parse_xeasy(filepath)
        elif format == "nmrpipe":
            self.peaks = self._parse_nmrpipe(filepath)
        elif format == "sparky":
            self.peaks = self._parse_sparky(filepath)
        elif format == "csv":
            self.peaks = self._parse_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return self.peaks

    def _detect_format(self, filepath: Path) -> str:
        """Auto-detect the file format based on content.

        Args:
            filepath: Path to the file

        Returns:
            Format string
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Check first few lines for format indicators
        first_lines = '\n'.join(lines[:10])

        if 'VARS' in first_lines and 'FORMAT' in first_lines:
            return "nmrpipe"
        elif re.search(r'# Number of dimensions', first_lines):
            return "xeasy"
        elif ',' in lines[0] and ('H_shift' in first_lines or 'C_shift' in first_lines):
            return "csv"
        else:
            # Default to Sparky format
            return "sparky"

    def _parse_xeasy(self, filepath: Path) -> List[Peak]:
        """Parse XEASY format 2D peak list.

        Format:
        # Number of dimensions 2
        # FORMAT xeasy2D
        Peak_ID  Shift1  Shift2  Color  Type  Volume  Vol_Err  ...
        """
        peaks = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                peak_id = int(parts[0])
                h_shift = float(parts[1])
                c_shift = float(parts[2])
                intensity = float(parts[5])

                # Check if assignment is provided
                assignment = None
                if len(parts) > 9:
                    assignment = parts[9] if parts[9] != '0' else None

                peak = Peak(
                    peak_id=peak_id,
                    shifts=np.array([h_shift, c_shift]),
                    intensity=intensity,
                    assignment=assignment
                )
                peaks.append(peak)

        return peaks

    def _parse_nmrpipe(self, filepath: Path) -> List[Peak]:
        """Parse NMRPipe tab format.

        Format:
        VARS   INDEX X_AXIS Y_AXIS X_PPM Y_PPM VOL HEIGHT ASS
        FORMAT %5d %9.3f %9.3f %8.3f %8.3f %+e %+e %s
        """
        peaks = []
        reading_data = False

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('VARS'):
                    continue
                if line.startswith('FORMAT'):
                    reading_data = True
                    continue
                if not reading_data or not line:
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                peak_id = int(parts[0])
                h_shift = float(parts[3])  # X_PPM
                c_shift = float(parts[4])  # Y_PPM
                intensity = float(parts[5])  # VOL

                assignment = None
                if len(parts) > 7:
                    assignment = parts[7] if parts[7] not in ['None', '?', '-'] else None

                peak = Peak(
                    peak_id=peak_id,
                    shifts=np.array([h_shift, c_shift]),
                    intensity=intensity,
                    assignment=assignment
                )
                peaks.append(peak)

        return peaks

    def _parse_sparky(self, filepath: Path) -> List[Peak]:
        """Parse Sparky format.

        Format:
        Assignment  w1  w2  intensity
        L15HD1-L15CD1  0.845  24.3  50000
        """
        peaks = []
        peak_id = 1

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Assignment'):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                assignment = parts[0] if parts[0] not in ['?', '-'] else None
                h_shift = float(parts[1])
                c_shift = float(parts[2])
                intensity = float(parts[3])

                peak = Peak(
                    peak_id=peak_id,
                    shifts=np.array([h_shift, c_shift]),
                    intensity=intensity,
                    assignment=assignment
                )
                peaks.append(peak)
                peak_id += 1

        return peaks

    def _parse_csv(self, filepath: Path) -> List[Peak]:
        """Parse simple CSV format.

        Expected columns: peak_id, H_shift, C_shift, intensity, [assignment]
        """
        df = pd.read_csv(filepath)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        peaks = []
        for idx, row in df.iterrows():
            peak_id = row.get('peak_id', idx + 1)
            h_shift = row['h_shift']
            c_shift = row['c_shift']
            intensity = row['intensity']

            assignment = None
            if 'assignment' in row:
                assignment = row['assignment'] if pd.notna(row['assignment']) else None

            peak = Peak(
                peak_id=int(peak_id),
                shifts=np.array([float(h_shift), float(c_shift)]),
                intensity=float(intensity),
                assignment=assignment
            )
            peaks.append(peak)

        return peaks

    def get_peaks(self) -> List[Peak]:
        """Return the list of parsed peaks."""
        return self.peaks

    def get_peak_dataframe(self) -> pd.DataFrame:
        """Convert peaks to a pandas DataFrame."""
        data = {
            'peak_id': [p.peak_id for p in self.peaks],
            'H_shift': [p.shifts[0] for p in self.peaks],
            'C_shift': [p.shifts[1] for p in self.peaks],
            'intensity': [p.intensity for p in self.peaks],
            'assignment': [p.assignment for p in self.peaks],
        }
        return pd.DataFrame(data)


class NOESYParser:
    """Parser for 3D CCH-NOESY peak lists.

    The NOESY data contains cross-peaks showing spatial proximity between
    methyl groups. This parser extracts both individual peaks and identifies
    NOE cross-peak relationships.
    """

    def __init__(self, tolerance: float = 0.05):
        """Initialize the NOESY parser.

        Args:
            tolerance: Chemical shift tolerance for matching peaks (ppm)
        """
        self.peaks: List[Peak] = []
        self.crosspeaks: List[NOECrosspeaks] = []
        self.tolerance = tolerance
        self.format: Optional[str] = None

    def parse(self, filepath: str, format: Optional[str] = None) -> List[Peak]:
        """Parse a NOESY peak list file.

        Args:
            filepath: Path to the peak list file
            format: Format specification ('xeasy', 'nmrpipe', 'csv')

        Returns:
            List of Peak objects with 3D chemical shifts
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if format is None:
            format = self._detect_format(filepath)

        self.format = format

        if format == "xeasy":
            self.peaks = self._parse_xeasy_3d(filepath)
        elif format == "nmrpipe":
            self.peaks = self._parse_nmrpipe_3d(filepath)
        elif format == "csv":
            self.peaks = self._parse_csv_3d(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return self.peaks

    def _detect_format(self, filepath: Path) -> str:
        """Auto-detect file format."""
        with open(filepath, 'r') as f:
            lines = [line for line in f if line.strip()]
            first_lines = '\n'.join(lines[:10])

        if 'VARS' in first_lines:
            return "nmrpipe"
        elif '# Number of dimensions 3' in first_lines:
            return "xeasy"
        else:
            return "csv"

    def _parse_xeasy_3d(self, filepath: Path) -> List[Peak]:
        """Parse XEASY format 3D peak list."""
        peaks = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 7:
                    continue

                peak_id = int(parts[0])
                shift1 = float(parts[1])  # C1
                shift2 = float(parts[2])  # H
                shift3 = float(parts[3])  # C2
                intensity = float(parts[6])

                assignment = None
                if len(parts) > 11:
                    assignment = parts[11] if parts[11] != '0' else None

                peak = Peak(
                    peak_id=peak_id,
                    shifts=np.array([shift1, shift2, shift3]),
                    intensity=intensity,
                    assignment=assignment
                )
                peaks.append(peak)

        return peaks

    def _parse_nmrpipe_3d(self, filepath: Path) -> List[Peak]:
        """Parse NMRPipe 3D format."""
        peaks = []
        reading_data = False

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('FORMAT'):
                    reading_data = True
                    continue
                if not reading_data or not line or line.startswith('VARS'):
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                peak_id = int(parts[0])
                shift1 = float(parts[4])  # X_PPM
                shift2 = float(parts[5])  # Y_PPM
                shift3 = float(parts[6])  # Z_PPM
                intensity = float(parts[7])  # VOL

                assignment = parts[9] if len(parts) > 9 and parts[9] != '?' else None

                peak = Peak(
                    peak_id=peak_id,
                    shifts=np.array([shift1, shift2, shift3]),
                    intensity=intensity,
                    assignment=assignment
                )
                peaks.append(peak)

        return peaks

    def _parse_csv_3d(self, filepath: Path) -> List[Peak]:
        """Parse CSV format for 3D NOESY."""
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower().str.strip()

        peaks = []
        for idx, row in df.iterrows():
            peak_id = row.get('peak_id', idx + 1)
            shifts = np.array([
                float(row['shift1']),
                float(row['shift2']),
                float(row['shift3'])
            ])
            intensity = float(row['intensity'])

            assignment = None
            if 'assignment' in row:
                assignment = row['assignment'] if pd.notna(row['assignment']) else None

            peak = Peak(
                peak_id=int(peak_id),
                shifts=shifts,
                intensity=intensity,
                assignment=assignment
            )
            peaks.append(peak)

        return peaks

    def match_with_hmqc(self, hmqc_peaks: List[Peak]) -> List[NOECrosspeaks]:
        """Identify NOE cross-peaks by matching with HMQC diagonal peaks.

        This method finds which 3D NOESY peaks correspond to NOE correlations
        between two methyl groups identified in the 2D HMQC.

        Args:
            hmqc_peaks: List of Peak objects from HMQC spectrum

        Returns:
            List of NOECrosspeaks objects
        """
        self.crosspeaks = []

        # Build index of HMQC peaks for fast lookup
        hmqc_dict = {}
        for peak in hmqc_peaks:
            h_shift = peak.shifts[0]
            c_shift = peak.shifts[1]
            hmqc_dict[peak.peak_id] = (h_shift, c_shift)

        # For each NOESY peak, try to match both ends to HMQC peaks
        for noesy_peak in self.peaks:
            c1_shift = noesy_peak.shifts[0]
            h_shift = noesy_peak.shifts[1]
            c2_shift = noesy_peak.shifts[2]

            # Find HMQC peaks matching the two ends
            match1 = self._find_matching_peak(hmqc_peaks, h_shift, c1_shift)
            match2 = self._find_matching_peak(hmqc_peaks, h_shift, c2_shift)

            if match1 is not None and match2 is not None and match1 != match2:
                # Calculate confidence based on peak quality
                confidence = self._calculate_confidence(noesy_peak)

                crosspeak = NOECrosspeaks(
                    peak1_id=match1,
                    peak2_id=match2,
                    noe_intensity=noesy_peak.intensity,
                    confidence=confidence
                )
                self.crosspeaks.append(crosspeak)

        return self.crosspeaks

    def _find_matching_peak(
        self,
        hmqc_peaks: List[Peak],
        h_shift: float,
        c_shift: float
    ) -> Optional[int]:
        """Find HMQC peak matching given chemical shifts.

        Args:
            hmqc_peaks: List of HMQC peaks
            h_shift: Proton chemical shift
            c_shift: Carbon chemical shift

        Returns:
            Peak ID if match found, None otherwise
        """
        for peak in hmqc_peaks:
            h_diff = abs(peak.shifts[0] - h_shift)
            c_diff = abs(peak.shifts[1] - c_shift)

            if h_diff < self.tolerance and c_diff < self.tolerance:
                return peak.peak_id

        return None

    def _calculate_confidence(self, peak: Peak) -> float:
        """Calculate confidence score for a peak.

        Factors considered:
        - Peak intensity (higher is better)
        - Signal-to-noise (if available)
        - Linewidth (sharper is better)

        Args:
            peak: Peak object

        Returns:
            Confidence score between 0 and 1
        """
        # Simple intensity-based confidence
        # In practice, this should be more sophisticated
        if peak.intensity <= 0:
            return 0.0

        # Normalize by typical intensity range (this should be calibrated)
        confidence = min(1.0, peak.intensity / 1e5)

        return confidence

    def get_peaks(self) -> List[Peak]:
        """Return the list of parsed NOESY peaks."""
        return self.peaks

    def get_crosspeaks(self) -> List[NOECrosspeaks]:
        """Return the list of identified NOE cross-peaks."""
        return self.crosspeaks

    def build_connectivity_matrix(self, num_peaks: int) -> np.ndarray:
        """Build adjacency matrix from NOE cross-peaks.

        Args:
            num_peaks: Number of diagonal peaks (from HMQC)

        Returns:
            Symmetric matrix where entry (i,j) is the NOE intensity
        """
        matrix = np.zeros((num_peaks, num_peaks))

        for cp in self.crosspeaks:
            i = cp.peak1_id - 1  # Convert to 0-indexed
            j = cp.peak2_id - 1

            # Weight by intensity and confidence
            weight = cp.noe_intensity * cp.confidence

            matrix[i, j] = weight
            matrix[j, i] = weight  # Symmetric

        return matrix
