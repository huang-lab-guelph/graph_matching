"""Dataset classes for NMR graph matching."""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from ..data.pdb_parser import PDBParser
from ..data.nmr_parser import HMQCParser, NOESYParser
from ..graphs.methyl_network import MethylNetworkBuilder
from ..graphs.peak_network import PeakNetworkBuilder


class NMRDataset(Dataset):
    """Dataset for NMR methyl assignment graph matching.

    Each sample contains:
    - Peak network (Graph A) from HMQC and NOESY data
    - Methyl network (Graph B) from PDB structure
    - Ground truth assignments (if available)
    """

    def __init__(
        self,
        data_list: List[Dict[str, str]],
        distance_cutoff: float = 10.0,
        shift_normalization: str = "standard",
        confidence_threshold: float = 0.0,
        cache_graphs: bool = True
    ):
        """Initialize the NMR dataset.

        Args:
            data_list: List of dictionaries with keys:
                - 'pdb_file': Path to PDB file
                - 'hmqc_file': Path to HMQC peak list
                - 'noesy_file': Path to NOESY peak list
                - 'name': Optional sample name
            distance_cutoff: Distance cutoff for methyl network edges (Å)
            shift_normalization: Chemical shift normalization method
            confidence_threshold: Minimum confidence for peak network edges
            cache_graphs: Whether to cache constructed graphs
        """
        super(NMRDataset, self).__init__()

        self.data_list = data_list
        self.distance_cutoff = distance_cutoff
        self.shift_normalization = shift_normalization
        self.confidence_threshold = confidence_threshold
        self.cache_graphs = cache_graphs

        # Initialize builders
        self.methyl_builder = MethylNetworkBuilder(distance_cutoff=distance_cutoff)
        self.peak_builder = PeakNetworkBuilder(
            shift_normalization=shift_normalization,
            confidence_threshold=confidence_threshold
        )

        # Cache for constructed graphs
        self._cache = {} if cache_graphs else None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'peak_graph': Peak network Data object
                - 'methyl_graph': Methyl network Data object
                - 'ground_truth': Ground truth assignment matrix (if available)
                - 'name': Sample name
        """
        # Check cache first
        if self.cache_graphs and idx in self._cache:
            return self._cache[idx]

        # Load data paths
        sample_info = self.data_list[idx]
        pdb_file = sample_info['pdb_file']
        hmqc_file = sample_info['hmqc_file']
        noesy_file = sample_info['noesy_file']
        sample_name = sample_info.get('name', f'sample_{idx}')

        # Build methyl network
        methyl_graph = self.methyl_builder.build_from_pdb(pdb_file)

        # Parse NMR data
        hmqc_parser = HMQCParser()
        hmqc_peaks = hmqc_parser.parse(hmqc_file)

        noesy_parser = NOESYParser()
        noesy_peaks = noesy_parser.parse(noesy_file)
        crosspeaks = noesy_parser.match_with_hmqc(hmqc_peaks)

        # Build peak network
        peak_graph = self.peak_builder.build(hmqc_peaks, crosspeaks)

        # Extract ground truth assignments if available
        ground_truth = self._extract_ground_truth(
            hmqc_peaks,
            methyl_graph.methyl_names
        )

        sample = {
            'peak_graph': peak_graph,
            'methyl_graph': methyl_graph,
            'ground_truth': ground_truth,
            'name': sample_name
        }

        # Cache if enabled
        if self.cache_graphs:
            self._cache[idx] = sample

        return sample

    def _extract_ground_truth(
        self,
        peaks: List,
        methyl_names: List[str]
    ) -> Optional[torch.Tensor]:
        """Extract ground truth assignment matrix from peak assignments.

        Args:
            peaks: List of Peak objects with optional assignments
            methyl_names: List of methyl identifiers from structure

        Returns:
            Ground truth matrix [num_peaks, num_methyls] or None if no assignments
        """
        # Check if any peaks have assignments
        has_assignments = any(peak.assignment is not None for peak in peaks)
        if not has_assignments:
            return None

        num_peaks = len(peaks)
        num_methyls = len(methyl_names)

        # Create mapping from methyl name to index
        methyl_to_idx = {name: idx for idx, name in enumerate(methyl_names)}

        # Build ground truth matrix
        ground_truth = torch.zeros((num_peaks, num_methyls), dtype=torch.float32)

        for peak_idx, peak in enumerate(peaks):
            if peak.assignment is None:
                continue

            # Parse assignment string (e.g., "L15HD1-L15CD1")
            # Extract methyl identifier
            assignment = peak.assignment
            if '-' in assignment:
                parts = assignment.split('-')
                methyl_id = parts[0] if parts[0] in methyl_to_idx else parts[1]
            else:
                methyl_id = assignment

            # Find matching methyl
            if methyl_id in methyl_to_idx:
                methyl_idx = methyl_to_idx[methyl_id]
                ground_truth[peak_idx, methyl_idx] = 1.0

        return ground_truth

    @staticmethod
    def from_directory(
        data_dir: str,
        pdb_pattern: str = "*.pdb",
        hmqc_pattern: str = "*hmqc*.txt",
        noesy_pattern: str = "*noesy*.txt"
    ) -> 'NMRDataset':
        """Create dataset from a directory of data files.

        Args:
            data_dir: Directory containing data files
            pdb_pattern: Glob pattern for PDB files
            hmqc_pattern: Glob pattern for HMQC files
            noesy_pattern: Glob pattern for NOESY files

        Returns:
            NMRDataset instance
        """
        data_dir = Path(data_dir)

        # Find all PDB files
        pdb_files = sorted(data_dir.glob(pdb_pattern))

        data_list = []
        for pdb_file in pdb_files:
            # Try to find corresponding HMQC and NOESY files
            base_name = pdb_file.stem

            # Look for HMQC file
            hmqc_candidates = list(data_dir.glob(f"{base_name}*{hmqc_pattern}"))
            if not hmqc_candidates:
                hmqc_candidates = list(data_dir.glob(hmqc_pattern))

            # Look for NOESY file
            noesy_candidates = list(data_dir.glob(f"{base_name}*{noesy_pattern}"))
            if not noesy_candidates:
                noesy_candidates = list(data_dir.glob(noesy_pattern))

            if hmqc_candidates and noesy_candidates:
                data_list.append({
                    'pdb_file': str(pdb_file),
                    'hmqc_file': str(hmqc_candidates[0]),
                    'noesy_file': str(noesy_candidates[0]),
                    'name': base_name
                })

        return NMRDataset(data_list)


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """Custom collate function for batching NMR data.

    Since graphs have different sizes, we keep them as lists rather than
    batching them into a single large graph.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    return {
        'peak_graphs': [sample['peak_graph'] for sample in batch],
        'methyl_graphs': [sample['methyl_graph'] for sample in batch],
        'ground_truths': [sample['ground_truth'] for sample in batch],
        'names': [sample['name'] for sample in batch]
    }
