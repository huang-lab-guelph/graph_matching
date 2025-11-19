"""PDB parser for extracting methyl group coordinates and information."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import PDBParser as BioPDBParser, Structure
from dataclasses import dataclass


@dataclass
class MethylGroup:
    """Represents a single methyl group in a protein structure.

    Attributes:
        residue_name: Three-letter residue code (LEU, VAL, ILE, ALA, THR, MET)
        residue_number: Residue sequence number
        chain_id: Chain identifier
        atom_name: Methyl carbon atom name (e.g., CD1, CG2)
        coordinates: 3D coordinates (x, y, z) in Angstroms
        full_name: Complete identifier (e.g., "L15CD1")
    """
    residue_name: str
    residue_number: int
    chain_id: str
    atom_name: str
    coordinates: np.ndarray
    full_name: str


class PDBParser:
    """Parser for extracting methyl groups from PDB files.

    This parser identifies and extracts methyl carbon positions from protein
    structures, focusing on residues commonly used in methyl-TROSY NMR:
    - Leucine (LEU): CD1, CD2
    - Valine (VAL): CG1, CG2
    - Isoleucine (ILE): CG2, CD1
    - Alanine (ALA): CB
    - Threonine (THR): CG2
    - Methionine (MET): CE
    """

    # Mapping of residue types to their methyl carbon atom names
    METHYL_ATOMS = {
        "LEU": ["CD1", "CD2"],
        "VAL": ["CG1", "CG2"],
        "ILE": ["CG2", "CD1"],
        "ALA": ["CB"],
        "THR": ["CG2"],
        "MET": ["CE"],
    }

    # One-letter codes for residue types
    RESIDUE_ONE_LETTER = {
        "LEU": "L", "VAL": "V", "ILE": "I",
        "ALA": "A", "THR": "T", "MET": "M"
    }

    def __init__(self, permissive: bool = True, quiet: bool = True):
        """Initialize the PDB parser.

        Args:
            permissive: If True, skip errors in PDB files
            quiet: If True, suppress warnings
        """
        self.parser = BioPDBParser(PERMISSIVE=permissive, QUIET=quiet)
        self.structure: Optional[Structure] = None
        self.methyl_groups: List[MethylGroup] = []

    def parse(self, pdb_file: str, structure_id: str = "structure") -> List[MethylGroup]:
        """Parse a PDB file and extract all methyl groups.

        Args:
            pdb_file: Path to PDB file
            structure_id: Identifier for the structure

        Returns:
            List of MethylGroup objects
        """
        self.structure = self.parser.get_structure(structure_id, pdb_file)
        self.methyl_groups = []

        # Iterate through structure hierarchy: Model -> Chain -> Residue -> Atom
        for model in self.structure:
            for chain in model:
                for residue in chain:
                    self._extract_methyls_from_residue(residue, chain.id)

        return self.methyl_groups

    def _extract_methyls_from_residue(self, residue, chain_id: str) -> None:
        """Extract methyl groups from a single residue.

        Args:
            residue: BioPython residue object
            chain_id: Chain identifier
        """
        res_name = residue.get_resname()

        # Check if this residue type contains methyls
        if res_name not in self.METHYL_ATOMS:
            return

        res_num = residue.get_id()[1]  # Residue number

        # Extract each methyl carbon atom
        for atom_name in self.METHYL_ATOMS[res_name]:
            if atom_name in residue:
                atom = residue[atom_name]
                coordinates = atom.get_coord()

                # Create short identifier (e.g., "L15CD1")
                one_letter = self.RESIDUE_ONE_LETTER[res_name]
                full_name = f"{one_letter}{res_num}{atom_name}"

                methyl = MethylGroup(
                    residue_name=res_name,
                    residue_number=res_num,
                    chain_id=chain_id,
                    atom_name=atom_name,
                    coordinates=coordinates,
                    full_name=full_name
                )

                self.methyl_groups.append(methyl)

    def get_methyl_groups(self) -> List[MethylGroup]:
        """Return the list of extracted methyl groups.

        Returns:
            List of MethylGroup objects
        """
        return self.methyl_groups

    def get_coordinates_matrix(self) -> np.ndarray:
        """Get coordinates of all methyl groups as a numpy array.

        Returns:
            Array of shape (N, 3) where N is the number of methyls
        """
        if not self.methyl_groups:
            return np.array([]).reshape(0, 3)

        return np.array([m.coordinates for m in self.methyl_groups])

    def get_residue_types(self) -> List[str]:
        """Get residue types for all methyl groups.

        Returns:
            List of three-letter residue codes
        """
        return [m.residue_name for m in self.methyl_groups]

    def get_full_names(self) -> List[str]:
        """Get full identifiers for all methyl groups.

        Returns:
            List of identifiers (e.g., ["L15CD1", "V23CG1", ...])
        """
        return [m.full_name for m in self.methyl_groups]

    def filter_by_residue_types(self, residue_types: List[str]) -> List[MethylGroup]:
        """Filter methyl groups by residue type.

        Args:
            residue_types: List of three-letter codes (e.g., ["LEU", "VAL"])

        Returns:
            Filtered list of MethylGroup objects
        """
        return [m for m in self.methyl_groups if m.residue_name in residue_types]

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise distance matrix between all methyl groups.

        Returns:
            Symmetric matrix of shape (N, N) with distances in Angstroms
        """
        coords = self.get_coordinates_matrix()
        n = len(coords)

        if n == 0:
            return np.array([]).reshape(0, 0)

        # Compute pairwise distances using broadcasting
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        return distances

    def get_methyl_pairs_within_distance(
        self,
        max_distance: float = 10.0,
        min_distance: float = 0.0
    ) -> List[Tuple[int, int, float]]:
        """Get pairs of methyls within a specified distance range.

        Args:
            max_distance: Maximum distance in Angstroms
            min_distance: Minimum distance in Angstroms

        Returns:
            List of tuples (index1, index2, distance)
        """
        distances = self.compute_distance_matrix()
        pairs = []

        n = len(self.methyl_groups)
        for i in range(n):
            for j in range(i + 1, n):  # Only upper triangle
                dist = distances[i, j]
                if min_distance <= dist <= max_distance:
                    pairs.append((i, j, dist))

        return pairs

    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics of extracted methyls.

        Returns:
            Dictionary with counts per residue type
        """
        summary = {}
        for methyl in self.methyl_groups:
            res_name = methyl.residue_name
            summary[res_name] = summary.get(res_name, 0) + 1

        summary["TOTAL"] = len(self.methyl_groups)
        return summary
