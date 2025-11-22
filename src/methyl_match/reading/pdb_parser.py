"""
PDB structure parser for extracting methyl groups.

This module provides functionality to parse PDB files and extract methyl group
information including positions, residue types, and atom coordinates.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from Bio.PDB import PDBParser as BioPDBParser
    from Bio.PDB.Structure import Structure
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MethylGroup:
    """
    Represents a methyl group in a protein structure.

    Attributes:
        residue_name: Three-letter residue code (e.g., 'LEU', 'VAL')
        residue_number: Residue sequence number
        chain_id: Chain identifier
        atom_name: Methyl carbon atom name (e.g., 'CD1', 'CG1', 'CB')
        coordinates: XYZ coordinates of the methyl carbon (Angstroms)
        label: Full label (e.g., 'L8-CD1')
    """
    residue_name: str
    residue_number: int
    chain_id: str
    atom_name: str
    coordinates: np.ndarray
    label: str

    def __post_init__(self):
        """Validate and convert coordinates to numpy array."""
        if not isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.array(self.coordinates)
        if self.coordinates.shape != (3,):
            raise ValueError(f"Coordinates must be 3D, got shape {self.coordinates.shape}")


class PDBParser:
    """
    Parser for extracting methyl groups from PDB structure files.

    This parser identifies and extracts methyl groups from the following residues:
    - LEU (leucine): CD1, CD2
    - VAL (valine): CG1, CG2
    - ILE (isoleucine): CD1
    - ALA (alanine): CB
    - THR (threonine): CG2
    - MET (methionine): CE

    Example:
        >>> parser = PDBParser("structure.pdb")
        >>> methyls = parser.extract_methyls()
        >>> print(f"Found {len(methyls)} methyl groups")
    """

    # Mapping of residue types to their methyl carbon atom names
    METHYL_ATOMS = {
        'LEU': ['CD1', 'CD2'],
        'VAL': ['CG1', 'CG2'],
        'ILE': ['CD1'],
        'ALA': ['CB'],
        'THR': ['CG2'],
        'MET': ['CE'],
    }

    # One-letter code mapping
    RESIDUE_CODE = {
        'LEU': 'L',
        'VAL': 'V',
        'ILE': 'I',
        'ALA': 'A',
        'THR': 'T',
        'MET': 'M',
    }

    def __init__(self, pdb_path: str):
        """
        Initialize the PDB parser.

        Args:
            pdb_path: Path to the PDB file

        Raises:
            FileNotFoundError: If PDB file doesn't exist
            ImportError: If BioPython is not installed
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError(
                "BioPython is required for PDB parsing. "
                "Install it with: pip install biopython"
            )

        self.pdb_path = Path(pdb_path)
        if not self.pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        self.structure: Optional[Structure] = None
        logger.info(f"Initialized PDBParser for {pdb_path}")

    def parse(self) -> Structure:
        """
        Parse the PDB file using BioPython.

        Returns:
            BioPython Structure object

        Raises:
            Exception: If parsing fails
        """
        try:
            parser = BioPDBParser(QUIET=True)
            self.structure = parser.get_structure('protein', str(self.pdb_path))
            logger.info(f"Successfully parsed PDB structure from {self.pdb_path}")
            return self.structure
        except Exception as e:
            logger.error(f"Failed to parse PDB file: {e}")
            raise

    def extract_methyls(self, chain_ids: Optional[List[str]] = None) -> List[MethylGroup]:
        """
        Extract all methyl groups from the protein structure.

        Args:
            chain_ids: List of chain IDs to process. If None, process all chains.

        Returns:
            List of MethylGroup objects

        Example:
            >>> parser = PDBParser("1ubq.pdb")
            >>> methyls = parser.extract_methyls(chain_ids=['A'])
            >>> for m in methyls[:3]:
            ...     print(f"{m.label}: {m.coordinates}")
        """
        if self.structure is None:
            self.parse()

        methyls = []

        for model in self.structure:
            for chain in model:
                # Skip chains not in the specified list
                if chain_ids is not None and chain.id not in chain_ids:
                    continue

                for residue in chain:
                    # Skip hetero residues and water
                    if residue.id[0] != ' ':
                        continue

                    res_name = residue.resname

                    # Check if this residue type has methyl groups
                    if res_name not in self.METHYL_ATOMS:
                        continue

                    # Extract each methyl carbon atom
                    for atom_name in self.METHYL_ATOMS[res_name]:
                        if atom_name in residue:
                            atom = residue[atom_name]
                            coords = atom.get_coord()

                            # Create label (e.g., 'L8-CD1')
                            one_letter = self.RESIDUE_CODE[res_name]
                            res_num = residue.id[1]
                            label = f"{one_letter}{res_num}-{atom_name}"

                            methyl = MethylGroup(
                                residue_name=res_name,
                                residue_number=res_num,
                                chain_id=chain.id,
                                atom_name=atom_name,
                                coordinates=coords,
                                label=label
                            )
                            methyls.append(methyl)
                            logger.debug(f"Found methyl: {label} at {coords}")

        logger.info(f"Extracted {len(methyls)} methyl groups from structure")
        return methyls

    def get_methyl_by_label(self, label: str, methyls: Optional[List[MethylGroup]] = None) -> Optional[MethylGroup]:
        """
        Find a specific methyl group by its label.

        Args:
            label: Label in format 'L8-CD1' (residue code + number + atom name)
            methyls: List of methyls to search. If None, extract from structure.

        Returns:
            MethylGroup if found, None otherwise

        Example:
            >>> parser = PDBParser("1ubq.pdb")
            >>> methyl = parser.get_methyl_by_label("L8-CD1")
            >>> print(methyl.coordinates if methyl else "Not found")
        """
        if methyls is None:
            methyls = self.extract_methyls()

        for methyl in methyls:
            if methyl.label == label:
                return methyl

        logger.warning(f"Methyl with label '{label}' not found")
        return None

    def calculate_distance(self, methyl1: MethylGroup, methyl2: MethylGroup) -> float:
        """
        Calculate the Euclidean distance between two methyl groups.

        Args:
            methyl1: First methyl group
            methyl2: Second methyl group

        Returns:
            Distance in Angstroms

        Example:
            >>> parser = PDBParser("1ubq.pdb")
            >>> methyls = parser.extract_methyls()
            >>> if len(methyls) >= 2:
            ...     dist = parser.calculate_distance(methyls[0], methyls[1])
            ...     print(f"Distance: {dist:.2f} Å")
        """
        return float(np.linalg.norm(methyl1.coordinates - methyl2.coordinates))

    def get_distance_matrix(self, methyls: Optional[List[MethylGroup]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate pairwise distance matrix for all methyl groups.

        Args:
            methyls: List of methyl groups. If None, extract from structure.

        Returns:
            Tuple of (distance matrix, list of labels)

        Example:
            >>> parser = PDBParser("1ubq.pdb")
            >>> dist_matrix, labels = parser.get_distance_matrix()
            >>> print(f"Distance matrix shape: {dist_matrix.shape}")
        """
        if methyls is None:
            methyls = self.extract_methyls()

        n = len(methyls)
        dist_matrix = np.zeros((n, n))
        labels = [m.label for m in methyls]

        for i in range(n):
            for j in range(i + 1, n):
                dist = self.calculate_distance(methyls[i], methyls[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        logger.info(f"Calculated {n}x{n} distance matrix")
        return dist_matrix, labels
