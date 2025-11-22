"""
Comprehensive tests for the reading module.

Tests PDBParser, NOESYParser, and HMQCParser with real data files.
"""

import pytest
import numpy as np
from pathlib import Path

from methyl_match.reading import (
    PDBParser,
    MethylGroup,
    NOESYParser,
    NOEPeak,
    HMQCParser,
    HMQCPeak,
)


# Test data directory
TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "test"


class TestPDBParser:
    """Tests for PDB structure parser."""

    def test_parser_initialization(self):
        """Test that parser initializes correctly with valid file."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        assert parser.pdb_path.exists()
        assert parser.structure is None  # Not parsed yet

    def test_parser_initialization_missing_file(self):
        """Test that parser raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            PDBParser("nonexistent_file.pdb")

    def test_parse_structure(self):
        """Test parsing PDB structure."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        structure = parser.parse()
        assert structure is not None
        assert parser.structure is not None

    def test_extract_methyls(self):
        """Test extraction of methyl groups from structure."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        # Ubiquitin should have multiple methyl groups
        assert len(methyls) > 0
        print(f"\nFound {len(methyls)} methyl groups in ubiquitin")

        # Check that all methyls are MethylGroup instances
        assert all(isinstance(m, MethylGroup) for m in methyls)

        # Check that methyls have valid properties
        for methyl in methyls[:3]:
            assert methyl.residue_name in ['LEU', 'VAL', 'ILE', 'ALA', 'THR', 'MET']
            assert methyl.residue_number > 0
            assert methyl.chain_id is not None
            assert methyl.coordinates.shape == (3,)
            assert methyl.label is not None
            print(f"  {methyl.label}: {methyl.coordinates}")

    def test_extract_methyls_specific_chain(self):
        """Test extraction of methyls from specific chain."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls(chain_ids=['A'])

        assert len(methyls) > 0
        assert all(m.chain_id == 'A' for m in methyls)

    def test_methyl_group_label_format(self):
        """Test that methyl labels follow expected format."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        for methyl in methyls[:5]:
            # Label should be like 'L8-CD1' or 'V17-CG1'
            assert '-' in methyl.label
            parts = methyl.label.split('-')
            assert len(parts) == 2
            # First part: one letter + number
            assert parts[0][0] in ['L', 'V', 'I', 'A', 'T', 'M']
            assert parts[0][1:].isdigit()
            # Second part: atom name
            assert parts[1] in ['CD1', 'CD2', 'CG1', 'CG2', 'CB', 'CE']

    def test_get_methyl_by_label(self):
        """Test finding methyl by label."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        if len(methyls) > 0:
            # Get first methyl's label
            target_label = methyls[0].label
            found_methyl = parser.get_methyl_by_label(target_label, methyls)

            assert found_methyl is not None
            assert found_methyl.label == target_label
            np.testing.assert_array_equal(found_methyl.coordinates, methyls[0].coordinates)

    def test_get_methyl_by_label_not_found(self):
        """Test that get_methyl_by_label returns None for invalid label."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        found = parser.get_methyl_by_label("X999-ZZ9", methyls)
        assert found is None

    def test_calculate_distance(self):
        """Test distance calculation between methyls."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        if len(methyls) >= 2:
            distance = parser.calculate_distance(methyls[0], methyls[1])
            assert distance > 0
            assert isinstance(distance, float)
            print(f"\nDistance between {methyls[0].label} and {methyls[1].label}: {distance:.2f} Å")

    def test_get_distance_matrix(self):
        """Test generation of distance matrix."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        parser = PDBParser(str(pdb_file))
        methyls = parser.extract_methyls()

        dist_matrix, labels = parser.get_distance_matrix(methyls)

        assert dist_matrix.shape[0] == len(methyls)
        assert dist_matrix.shape[1] == len(methyls)
        assert len(labels) == len(methyls)

        # Check that matrix is symmetric
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)

        # Check that diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(dist_matrix), np.zeros(len(methyls)))

        # Check that all distances are non-negative
        assert np.all(dist_matrix >= 0)

        print(f"\nDistance matrix shape: {dist_matrix.shape}")
        print(f"Distance range: {dist_matrix[dist_matrix > 0].min():.2f} - {dist_matrix.max():.2f} Å")


class TestNOESYParser:
    """Tests for NOESY peak list parser."""

    def test_parser_initialization(self):
        """Test parser initialization with valid file."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        parser = NOESYParser(str(noesy_file))
        assert parser.file_path.exists()

    def test_parser_initialization_missing_file(self):
        """Test parser raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            NOESYParser("nonexistent_noesy.txt")

    def test_detect_format_xeasy(self):
        """Test format detection for XEASY format."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        parser = NOESYParser(str(noesy_file))
        format_detected = parser.detect_format()
        assert format_detected in ['xeasy', 'nmrpipe', 'sparky', 'csv']

    def test_detect_format_sparky(self):
        """Test format detection for Sparky format."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy_sparky.txt"
        parser = NOESYParser(str(noesy_file))
        format_detected = parser.detect_format()
        assert format_detected in ['sparky', 'xeasy']

    def test_parse_xeasy_format(self):
        """Test parsing XEASY format NOESY file."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        parser = NOESYParser(str(noesy_file))
        peaks = parser.parse()

        assert len(peaks) > 0
        print(f"\nParsed {len(peaks)} NOESY peaks from XEASY format")

        # Check peak properties
        for peak in peaks[:3]:
            assert isinstance(peak, NOEPeak)
            assert peak.index > 0
            assert isinstance(peak.w1, float)
            assert isinstance(peak.w2, float)
            assert isinstance(peak.intensity, float)
            assert peak.intensity > 0
            print(f"  {peak}")

    def test_parse_sparky_format(self):
        """Test parsing Sparky format NOESY file."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy_sparky.txt"
        parser = NOESYParser(str(noesy_file))
        peaks = parser.parse()

        assert len(peaks) > 0
        print(f"\nParsed {len(peaks)} NOESY peaks from Sparky format")

        for peak in peaks[:3]:
            assert isinstance(peak, NOEPeak)
            print(f"  {peak}")

    def test_peak_assignments(self):
        """Test that peak assignments are parsed correctly."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        parser = NOESYParser(str(noesy_file))
        peaks = parser.parse()

        # Check if any peaks have assignments
        assigned_peaks = [p for p in peaks if p.assignment1 or p.assignment2]
        print(f"\nFound {len(assigned_peaks)} assigned peaks out of {len(peaks)} total")

        if assigned_peaks:
            peak = assigned_peaks[0]
            print(f"  Example: {peak.assignment1} <-> {peak.assignment2}")

    def test_get_peak_by_index(self):
        """Test finding peak by index."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        parser = NOESYParser(str(noesy_file))
        peaks = parser.parse()

        if peaks:
            target_index = peaks[0].index
            found_peak = parser.get_peak_by_index(target_index, peaks)

            assert found_peak is not None
            assert found_peak.index == target_index

    def test_filter_by_intensity(self):
        """Test filtering peaks by intensity."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        parser = NOESYParser(str(noesy_file))
        peaks = parser.parse()

        if peaks:
            min_intensity = 40000
            filtered = parser.filter_by_intensity(peaks, min_intensity)

            assert len(filtered) <= len(peaks)
            assert all(p.intensity >= min_intensity for p in filtered)
            print(f"\nFiltered peaks: {len(filtered)}/{len(peaks)} with intensity >= {min_intensity}")

    def test_get_correlation_matrix(self):
        """Test generation of correlation matrix."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        parser = NOESYParser(str(noesy_file))
        peaks = parser.parse()

        # Get labels from assigned peaks
        labels = []
        for peak in peaks:
            if peak.assignment1 and peak.assignment1 not in labels:
                labels.append(peak.assignment1)
            if peak.assignment2 and peak.assignment2 not in labels:
                labels.append(peak.assignment2)

        if labels:
            matrix = parser.get_correlation_matrix(peaks, labels)

            assert matrix.shape[0] == len(labels)
            assert matrix.shape[1] == len(labels)

            # Matrix should be symmetric
            np.testing.assert_array_almost_equal(matrix, matrix.T)

            print(f"\nCorrelation matrix shape: {matrix.shape}")
            print(f"Non-zero entries: {np.count_nonzero(matrix)}")


class TestHMQCParser:
    """Tests for HMQC peak list parser."""

    def test_parser_initialization(self):
        """Test parser initialization with valid file."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        assert parser.file_path.exists()

    def test_parser_initialization_missing_file(self):
        """Test parser raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            HMQCParser("nonexistent_hmqc.txt")

    def test_detect_format_xeasy(self):
        """Test format detection for XEASY format."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        format_detected = parser.detect_format()
        assert format_detected in ['xeasy', 'nmrpipe', 'sparky', 'csv']

    def test_detect_format_csv(self):
        """Test format detection for CSV format."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.csv"
        parser = HMQCParser(str(hmqc_file))
        format_detected = parser.detect_format()
        assert format_detected == 'csv'

    def test_parse_xeasy_format(self):
        """Test parsing XEASY format HMQC file."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        peaks = parser.parse()

        assert len(peaks) > 0
        print(f"\nParsed {len(peaks)} HMQC peaks from XEASY format")

        # Check peak properties
        for peak in peaks[:3]:
            assert isinstance(peak, HMQCPeak)
            assert peak.index > 0
            assert isinstance(peak.h_shift, float)
            assert isinstance(peak.c_shift, float)
            assert isinstance(peak.intensity, float)
            assert peak.intensity > 0
            print(f"  {peak}")

    def test_parse_csv_format(self):
        """Test parsing CSV format HMQC file."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.csv"
        parser = HMQCParser(str(hmqc_file))
        peaks = parser.parse()

        assert len(peaks) > 0
        print(f"\nParsed {len(peaks)} HMQC peaks from CSV format")

        for peak in peaks[:3]:
            assert isinstance(peak, HMQCPeak)
            print(f"  {peak}")

    def test_peak_assignments(self):
        """Test that peak assignments are parsed correctly."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        peaks = parser.parse()

        # Check if any peaks have assignments
        assigned_peaks = [p for p in peaks if p.assignment]
        print(f"\nFound {len(assigned_peaks)} assigned peaks out of {len(peaks)} total")

        if assigned_peaks:
            peak = assigned_peaks[0]
            print(f"  Example: {peak.assignment} at 1H={peak.h_shift:.3f}, 13C={peak.c_shift:.2f}")

    def test_get_peak_by_assignment(self):
        """Test finding peak by assignment."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        peaks = parser.parse()

        assigned_peaks = [p for p in peaks if p.assignment]
        if assigned_peaks:
            target_assignment = assigned_peaks[0].assignment
            found_peak = parser.get_peak_by_assignment(target_assignment, peaks)

            assert found_peak is not None
            assert found_peak.assignment == target_assignment

    def test_get_chemical_shifts(self):
        """Test extraction of chemical shifts dictionary."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        shifts = parser.get_chemical_shifts()

        assert isinstance(shifts, dict)
        print(f"\nExtracted chemical shifts for {len(shifts)} assignments")

        if shifts:
            # Check structure of shifts dictionary
            for assignment, (h_shift, c_shift) in list(shifts.items())[:3]:
                assert isinstance(h_shift, float)
                assert isinstance(c_shift, float)
                print(f"  {assignment}: 1H={h_shift:.3f}, 13C={c_shift:.2f}")

    def test_filter_by_intensity(self):
        """Test filtering peaks by intensity."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        peaks = parser.parse()

        if peaks:
            min_intensity = 100000
            filtered = parser.filter_by_intensity(peaks, min_intensity)

            assert len(filtered) <= len(peaks)
            assert all(p.intensity >= min_intensity for p in filtered)
            print(f"\nFiltered peaks: {len(filtered)}/{len(peaks)} with intensity >= {min_intensity}")

    def test_get_unassigned_peaks(self):
        """Test getting unassigned peaks."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        unassigned = parser.get_unassigned_peaks()

        assert isinstance(unassigned, list)
        assert all(not p.assignment for p in unassigned)
        print(f"\nFound {len(unassigned)} unassigned peaks")

    def test_match_chemical_shifts(self):
        """Test matching peaks by chemical shifts."""
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"
        parser = HMQCParser(str(hmqc_file))
        peaks = parser.parse()

        if peaks:
            # Use first peak's shifts as target
            target_peak = peaks[0]
            matches = parser.match_chemical_shifts(
                target_peak.h_shift,
                target_peak.c_shift,
                h_tolerance=0.01,
                c_tolerance=0.5,
                peaks=peaks
            )

            # Should at least match itself
            assert len(matches) >= 1
            print(f"\nFound {len(matches)} peaks matching shifts "
                  f"(1H={target_peak.h_shift:.3f}±0.01, 13C={target_peak.c_shift:.2f}±0.5)")


class TestIntegration:
    """Integration tests using multiple parsers together."""

    def test_parse_all_files(self):
        """Test parsing all available test files."""
        pdb_file = TEST_DATA_DIR / "1ubq.pdb"
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"

        # Parse PDB
        pdb_parser = PDBParser(str(pdb_file))
        methyls = pdb_parser.extract_methyls()

        # Parse NOESY
        noesy_parser = NOESYParser(str(noesy_file))
        noe_peaks = noesy_parser.parse()

        # Parse HMQC
        hmqc_parser = HMQCParser(str(hmqc_file))
        hmqc_peaks = hmqc_parser.parse()

        print(f"\n=== Integration Test Results ===")
        print(f"PDB methyls: {len(methyls)}")
        print(f"NOESY peaks: {len(noe_peaks)}")
        print(f"HMQC peaks: {len(hmqc_peaks)}")

        assert len(methyls) > 0
        assert len(noe_peaks) > 0
        assert len(hmqc_peaks) > 0

    def test_cross_reference_assignments(self):
        """Test cross-referencing assignments between NOESY and HMQC."""
        noesy_file = TEST_DATA_DIR / "example_methyl_noesy.txt"
        hmqc_file = TEST_DATA_DIR / "example_hmqc.txt"

        noesy_parser = NOESYParser(str(noesy_file))
        noe_peaks = noesy_parser.parse()

        hmqc_parser = HMQCParser(str(hmqc_file))
        hmqc_peaks = hmqc_parser.parse()

        # Get assigned labels from both
        noesy_labels = set()
        for peak in noe_peaks:
            if peak.assignment1:
                noesy_labels.add(peak.assignment1)
            if peak.assignment2:
                noesy_labels.add(peak.assignment2)

        hmqc_labels = {p.assignment for p in hmqc_peaks if p.assignment}

        # Check overlap
        common_labels = noesy_labels & hmqc_labels
        print(f"\n=== Cross-Reference Test ===")
        print(f"NOESY labels: {len(noesy_labels)}")
        print(f"HMQC labels: {len(hmqc_labels)}")
        print(f"Common labels: {len(common_labels)}")

        if common_labels:
            print(f"Examples: {list(common_labels)[:5]}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
