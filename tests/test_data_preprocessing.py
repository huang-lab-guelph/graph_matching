"""
Unit tests for the data_preprocessing.loader module.
"""

import os
import pytest
from src.data_preprocessing.loader import load_peak_list, load_dataset, preprocess_data

# Define dummy data for testing
DUMMY_PEAK_LIST_CONTENT = """id,1H,13C,intensity
peakA,1.0,10.0,100
peakB,1.1,10.5,95
"""

@pytest.fixture
def dummy_peak_file(tmp_path):
    """Creates a dummy peak list file for testing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(DUMMY_PEAK_LIST_CONTENT)
    return str(file_path)

@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Creates a dummy directory with multiple peak list files."""
    dir_path = tmp_path / "dataset"
    dir_path.mkdir()
    (dir_path / "sample1.txt").write_text(DUMMY_PEAK_LIST_CONTENT)
    (dir_path / "sample2.csv").write_text(DUMMY_PEAK_LIST_CONTENT)
    return str(dir_path)

def test_load_peak_list_exists(dummy_peak_file):
    """Test loading a valid peak list file."""
    peaks = load_peak_list(dummy_peak_file)
    # The current load_peak_list returns dummy data, not parsed content.
    # So we check for the structure of the dummy data.
    assert isinstance(peaks, list)
    assert len(peaks) == 2 # Current dummy implementation returns 2 peaks
    assert "id" in peaks[0]
    assert "1H" in peaks[0]

def test_load_peak_list_not_found():
    """Test loading a non-existent peak list file."""
    peaks = load_peak_list("non_existent_file.txt")
    assert isinstance(peaks, list)
    assert len(peaks) == 0 # Returns empty list for not found files

def test_load_dataset(dummy_dataset_dir):
    """Test loading multiple peak list files from a directory."""
    dataset = load_dataset(dummy_dataset_dir)
    assert isinstance(dataset, dict)
    assert len(dataset) == 2 # Two dummy files created
    assert "sample1" in dataset
    assert "sample2" in dataset
    assert isinstance(dataset["sample1"], list)
    assert len(dataset["sample1"]) == 2 # Each file returns 2 dummy peaks

def test_load_dataset_empty_dir(tmp_path):
    """Test loading from an empty directory."""
    empty_dir = tmp_path / "empty_dataset"
    empty_dir.mkdir()
    dataset = load_dataset(str(empty_dir))
    assert isinstance(dataset, dict)
    assert len(dataset) == 0

def test_load_dataset_non_existent_dir():
    """Test loading from a non-existent directory."""
    dataset = load_dataset("non_existent_dir")
    assert isinstance(dataset, dict)
    assert len(dataset) == 0

def test_preprocess_data(dummy_dataset_dir):
    """Test the data preprocessing function."""
    raw_data = load_dataset(dummy_dataset_dir)
    processed_data = preprocess_data(raw_data)
    assert isinstance(processed_data, dict)
    assert "sample1" in processed_data
    assert "sample2" in processed_data
    assert "peaks" in processed_data["sample1"]
    assert "graph_representation" in processed_data["sample1"]
    assert processed_data["sample1"]["graph_representation"] == "TODO" # Placeholder check
