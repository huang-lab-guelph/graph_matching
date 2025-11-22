"""
File reading module for parsing PDB structures and NMR peak lists.

This module provides parsers for:
- PDB structure files (methyl group extraction)
- NOESY peak lists (NOE connectivity data)
- HMQC peak lists (chemical shift data)

Supported formats: XEASY, NMRPipe, Sparky, CSV
"""

from methyl_match.reading.pdb_parser import PDBParser, MethylGroup
from methyl_match.reading.noesy_parser import NOESYParser, NOEPeak
from methyl_match.reading.hmqc_parser import HMQCParser, HMQCPeak

__all__ = [
    "PDBParser",
    "MethylGroup",
    "NOESYParser",
    "NOEPeak",
    "HMQCParser",
    "HMQCPeak",
]
