"""
Unit tests for the writing module.

Tests all formatters (TextFormatter, CSVFormatter, PyMOLFormatter) with
synthetic data to verify correct output generation.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import networkx as nx

from methyl_match.writing import TextFormatter, CSVFormatter, PyMOLFormatter
from methyl_match.matching.base import MatchingResult
from methyl_match.preprocessing.methyl_network import MethylNetwork
from methyl_match.preprocessing.peak_network import PeakNetwork
from methyl_match.reading.pdb_parser import MethylGroup
from methyl_match.reading.hmqc_parser import HMQCPeak
from methyl_match.reading.noesy_parser import NOEPeak


# Fixtures for test data
@pytest.fixture
def sample_methyls():
    """Create sample methyl groups for testing."""
    return [
        MethylGroup(
            residue_name="LEU",
            residue_number=42,
            chain_id="A",
            atom_name="CD1",
            coordinates=np.array([10.5, 15.2, 8.3]),
            label="L42-CD1",
        ),
        MethylGroup(
            residue_name="VAL",
            residue_number=8,
            chain_id="A",
            atom_name="CG2",
            coordinates=np.array([5.2, 10.1, 12.4]),
            label="V8-CG2",
        ),
        MethylGroup(
            residue_name="ILE",
            residue_number=23,
            chain_id="A",
            atom_name="CD1",
            coordinates=np.array([8.1, 9.5, 14.2]),
            label="I23-CD1",
        ),
    ]


@pytest.fixture
def sample_peaks():
    """Create sample HMQC peaks for testing."""
    return [
        HMQCPeak(index=1, h_shift=0.85, c_shift=21.5, intensity=1000.0),
        HMQCPeak(index=2, h_shift=0.91, c_shift=23.1, intensity=850.0, assignment="V8-CG2"),
        HMQCPeak(index=3, h_shift=0.78, c_shift=19.8, intensity=920.0),
        HMQCPeak(index=4, h_shift=0.95, c_shift=22.3, intensity=650.0),
    ]


@pytest.fixture
def sample_noe_peaks():
    """Create sample NOE peaks for testing."""
    return [
        NOEPeak(index=1, w1=21.5, w2=23.1, w3=0.88, intensity=500.0),
        NOEPeak(index=2, w1=19.8, w2=21.5, w3=0.82, intensity=300.0),
    ]


@pytest.fixture
def structural_network(sample_methyls):
    """Create sample structural network."""
    from methyl_match.preprocessing import MethylNetworkBuilder

    builder = MethylNetworkBuilder(distance_cutoff=10.0)
    return builder.build_network(sample_methyls)


@pytest.fixture
def experimental_network(sample_peaks, sample_noe_peaks):
    """Create sample experimental network."""
    from methyl_match.preprocessing import PeakNetworkBuilder

    builder = PeakNetworkBuilder(intensity_threshold=0.0)
    return builder.build_network(sample_peaks, sample_noe_peaks)


@pytest.fixture
def sample_result():
    """Create sample matching result."""
    return MatchingResult(
        assignments={
            0: 0,  # Peak 0 -> Methyl 0
            1: 1,  # Peak 1 -> Methyl 1
            2: 2,  # Peak 2 -> Methyl 2
        },
        confidence_scores={
            0: 0.95,  # High confidence
            1: 0.65,  # Medium confidence
            2: 0.42,  # Low confidence
        },
        unassigned_peaks=[3],
        metadata={
            "algorithm": "QAPMatcher",
            "runtime": 0.123,
            "iterations": 50,
        },
    )


# TextFormatter tests
class TestTextFormatter:
    """Tests for TextFormatter."""

    def test_basic_formatting(self, sample_result, structural_network, experimental_network):
        """Test basic text formatting."""
        formatter = TextFormatter()
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Check header
        assert "METHYL ASSIGNMENT RESULTS" in output
        assert "SUMMARY STATISTICS" in output

        # Check statistics
        assert "Total Assignments:        3" in output
        assert "Mean Confidence:" in output
        assert "High Confidence (>=0.7):  1" in output

        # Check assignments section
        assert "DETAILED ASSIGNMENTS" in output
        assert "LEU42-CD1" in output
        assert "VAL8-CG2" in output

    def test_confidence_threshold(self, sample_result, structural_network, experimental_network):
        """Test confidence threshold filtering."""
        formatter = TextFormatter(confidence_threshold=0.7)
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Only high confidence should be included
        assert "LEU42-CD1" in output  # 0.95
        assert "VAL8-CG2" not in output  # 0.65
        assert "ILE23-CD1" not in output  # 0.42

    def test_quality_icons(self, sample_result, structural_network, experimental_network):
        """Test quality icon display."""
        formatter = TextFormatter(show_quality_icons=True)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "🟢" in output  # High confidence
        assert "🟡" in output  # Medium confidence
        assert "🔴" in output  # Low confidence

    def test_no_quality_icons(self, sample_result, structural_network, experimental_network):
        """Test output without quality icons."""
        formatter = TextFormatter(show_quality_icons=False)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "🟢" not in output
        assert "🟡" not in output
        assert "🔴" not in output

    def test_metadata_inclusion(self, sample_result, structural_network, experimental_network):
        """Test metadata section."""
        formatter = TextFormatter(include_metadata=True)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "ALGORITHM METADATA" in output
        assert "QAPMatcher" in output
        assert "Runtime" in output

    def test_unassigned_section(self, sample_result, structural_network, experimental_network):
        """Test unassigned peaks section."""
        formatter = TextFormatter(include_unassigned=True)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "UNASSIGNED PEAKS" in output
        assert "Peak_4" in output or "0.95" in output  # Peak 3 (index 4)

    def test_file_output(self, sample_result, structural_network, experimental_network):
        """Test writing to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.txt"
            formatter = TextFormatter()
            formatter.format(
                sample_result,
                structural_network,
                experimental_network,
                output_path=output_path,
            )

            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
                assert "METHYL ASSIGNMENT RESULTS" in content


# CSVFormatter tests
class TestCSVFormatter:
    """Tests for CSVFormatter."""

    def test_basic_formatting(self, sample_result, structural_network, experimental_network):
        """Test basic CSV formatting."""
        formatter = CSVFormatter()
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Check header
        assert "peak_id,peak_assignment" in output
        assert "methyl_residue,methyl_number" in output
        assert "confidence,quality" in output

        # Check data rows
        lines = output.strip().split("\n")
        assert len(lines) >= 4  # Header + 3 assignments

    def test_confidence_threshold(self, sample_result, structural_network, experimental_network):
        """Test confidence threshold filtering."""
        formatter = CSVFormatter(confidence_threshold=0.7)
        output = formatter.format(sample_result, structural_network, experimental_network)

        lines = output.strip().split("\n")
        # Should only have header + 1 assignment (confidence >= 0.7)
        assert len(lines) == 2

    def test_coordinates_included(self, sample_result, structural_network, experimental_network):
        """Test coordinate inclusion."""
        formatter = CSVFormatter(include_coordinates=True)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "methyl_x,methyl_y,methyl_z" in output
        assert "10.500" in output  # Methyl 0 x-coordinate

    def test_coordinates_excluded(self, sample_result, structural_network, experimental_network):
        """Test coordinate exclusion."""
        formatter = CSVFormatter(include_coordinates=False)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "methyl_x" not in output
        assert "methyl_y" not in output
        assert "methyl_z" not in output

    def test_quality_classification(self, sample_result, structural_network, experimental_network):
        """Test quality classification in CSV."""
        formatter = CSVFormatter()
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "high" in output  # For confidence 0.95
        assert "medium" in output  # For confidence 0.65
        assert "low" in output  # For confidence 0.42

    def test_file_output(self, sample_result, structural_network, experimental_network):
        """Test writing to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            formatter = CSVFormatter()
            formatter.format(
                sample_result,
                structural_network,
                experimental_network,
                output_path=output_path,
            )

            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
                assert "peak_id" in content
                assert "confidence" in content

    def test_unassigned_csv(self, sample_result, structural_network, experimental_network):
        """Test unassigned peaks CSV creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            formatter = CSVFormatter(include_unassigned=True)
            formatter.format(
                sample_result,
                structural_network,
                experimental_network,
                output_path=output_path,
            )

            unassigned_path = Path(tmpdir) / "results_unassigned.csv"
            assert unassigned_path.exists()

            with open(unassigned_path) as f:
                content = f.read()
                assert "peak_id" in content


# PyMOLFormatter tests
class TestPyMOLFormatter:
    """Tests for PyMOLFormatter."""

    def test_basic_formatting(self, sample_result, structural_network, experimental_network):
        """Test basic PyMOL script generation."""
        formatter = PyMOLFormatter()
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Check header
        assert "# Methyl Assignment Visualization" in output
        assert "# Generated:" in output

        # Check basic commands
        assert "hide everything" in output
        assert "show cartoon" in output
        assert "select all_methyls" in output
        assert "show spheres, all_methyls" in output

    def test_pdb_loading(self, sample_result, structural_network, experimental_network):
        """Test PDB loading command."""
        formatter = PyMOLFormatter()
        output = formatter.format(
            sample_result,
            structural_network,
            experimental_network,
            pdb_path=Path("protein.pdb"),
        )

        assert "load protein.pdb" in output

    def test_confidence_coloring(self, sample_result, structural_network, experimental_network):
        """Test confidence-based coloring."""
        formatter = PyMOLFormatter(color_by_confidence=True)
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Check color commands
        assert "color green, high_conf" in output  # High confidence
        assert "color yellow, medium_conf" in output  # Medium confidence
        assert "color red, low_conf" in output  # Low confidence

    def test_methyl_selection(self, sample_result, structural_network, experimental_network):
        """Test methyl selection string."""
        formatter = PyMOLFormatter()
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Check methyl residue patterns
        assert "resn LEU and name CD*" in output
        assert "resn VAL and name CG*" in output
        assert "resn ILE and name CD1" in output

    def test_labels(self, sample_result, structural_network, experimental_network):
        """Test label generation."""
        formatter = PyMOLFormatter(show_labels=True)
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Check label commands
        assert "label" in output
        lines = [line for line in output.split("\n") if line.startswith("label")]
        assert len(lines) >= 3  # At least 3 labels for 3 assignments

    def test_no_labels(self, sample_result, structural_network, experimental_network):
        """Test output without labels."""
        formatter = PyMOLFormatter(show_labels=False)
        output = formatter.format(sample_result, structural_network, experimental_network)

        # Should not have label commands
        label_lines = [line for line in output.split("\n") if line.startswith("label")]
        assert len(label_lines) == 0

    def test_unassigned_coloring(self, sample_result, structural_network, experimental_network):
        """Test unassigned methyls coloring."""
        formatter = PyMOLFormatter(include_unassigned=True)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "color gray, all_methyls" in output

    def test_file_output(self, sample_result, structural_network, experimental_network):
        """Test writing to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "visualization.pml"
            formatter = PyMOLFormatter()
            formatter.format(
                sample_result,
                structural_network,
                experimental_network,
                output_path=output_path,
            )

            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
                assert "# Methyl Assignment Visualization" in content
                assert "select all_methyls" in content

    def test_sphere_scale(self, sample_result, structural_network, experimental_network):
        """Test sphere scale parameter."""
        formatter = PyMOLFormatter(sphere_scale=0.8)
        output = formatter.format(sample_result, structural_network, experimental_network)

        assert "set sphere_scale, 0.8" in output


# Integration tests
class TestFormatterIntegration:
    """Integration tests for all formatters."""

    def test_all_formatters_with_same_data(
        self, sample_result, structural_network, experimental_network
    ):
        """Test that all formatters work with the same data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Text formatter
            text_formatter = TextFormatter()
            text_output = text_formatter.format(
                sample_result,
                structural_network,
                experimental_network,
                output_path=tmpdir / "results.txt",
            )
            assert len(text_output) > 0

            # CSV formatter
            csv_formatter = CSVFormatter()
            csv_output = csv_formatter.format(
                sample_result,
                structural_network,
                experimental_network,
                output_path=tmpdir / "results.csv",
            )
            assert len(csv_output) > 0

            # PyMOL formatter
            pymol_formatter = PyMOLFormatter()
            pymol_output = pymol_formatter.format(
                sample_result,
                structural_network,
                experimental_network,
                output_path=tmpdir / "visualization.pml",
            )
            assert len(pymol_output) > 0

            # Check all files were created
            assert (tmpdir / "results.txt").exists()
            assert (tmpdir / "results.csv").exists()
            assert (tmpdir / "visualization.pml").exists()

    def test_empty_result(self, structural_network, experimental_network):
        """Test formatters with empty results."""
        empty_result = MatchingResult(
            assignments={},
            confidence_scores={},
            unassigned_peaks=[0, 1, 2, 3],
            metadata={"algorithm": "TestMatcher"},
        )

        # All formatters should handle empty results gracefully
        text_formatter = TextFormatter()
        text_output = text_formatter.format(
            empty_result, structural_network, experimental_network
        )
        assert "Total Assignments:        0" in text_output

        csv_formatter = CSVFormatter()
        csv_output = csv_formatter.format(
            empty_result, structural_network, experimental_network
        )
        # Should have header but no data rows
        lines = csv_output.strip().split("\n")
        assert len(lines) == 1  # Only header

        pymol_formatter = PyMOLFormatter()
        pymol_output = pymol_formatter.format(
            empty_result, structural_network, experimental_network
        )
        assert "# Total Assignments: 0" in pymol_output
