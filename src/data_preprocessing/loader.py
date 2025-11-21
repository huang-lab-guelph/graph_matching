"""
Module for loading and preprocessing NMR data, specifically peak lists,
for methyl assignment.
"""

import os
from typing import List, Dict, Any, Tuple
import networkx as nx
import torch
import re

# Specific methyl groups to consider based on user's request
# For experimental peak IDs
EXP_SPECIFIC_METHYL_TYPES = {
    'I': ['I'], # Simplified, just checking the prefix
    'M': ['M'],
    'L': ['LV'],
    'V': ['LV']
}

def parse_exp_peak_id(peak_id: str) -> Tuple[str, str, str]:
    """
    Parses an experimental peak ID. Handles two formats:
    1. 'A47CB-HB' -> ('A', '47', 'CB')
    2. 'IX1', 'LVX1?' -> ('I', '1', 'X') or ('LV', '1', 'X')
    """
    # Try to match the 'A47CB-HB' format
    match1 = re.match(r"([A-Z])(\d+)([A-Z0-9]+)", peak_id)
    if match1:
        aa_code, res_num, atom_part = match1.groups()
        atom_name_base = atom_part.split('-')[0]
        return aa_code, res_num, atom_name_base
    
    # Try to match the 'IX1' or 'LVX1?' format
    match2 = re.match(r"([A-Z]+)(\d+)", peak_id.replace('?', '').replace('X', '')) # Clean up the label
    if match2:
        aa_code, res_num = match2.groups()
        # For this format, we don't have a specific atom name, so we can return the AA code
        return aa_code, res_num, aa_code 

    return None, None, None


def is_specific_methyl_type_exp(peak_id: str) -> bool:
    """
    Checks if an experimental peak ID corresponds to one of the requested methyl types.
    """
    # For 'IX1' or 'LVX1?' formats
    if peak_id.startswith('I'):
        return True
    if peak_id.startswith('M'):
        return True
    if peak_id.startswith('LV'):
        return True
        
    # For 'A47CB-HB' format
    aa_code, _, atom_name_base = parse_exp_peak_id(peak_id)
    if aa_code is None:
        return False
    
    if aa_code in EXP_SPECIFIC_METHYL_TYPES:
        # This part of the logic might need to be refined if the label formats are mixed.
        # For now, the startswith checks handle the new format.
        return True

    return False


def load_hmqc_peak_list(file_path: str, filter_specific_types: bool = False) -> List[Dict[str, Any]]:
    """
    Loads a 2D HMQC peak list file.

    Args:
        file_path: The path to the HMQC peak list file.
        filter_specific_types: If True, only load peaks matching SPECIFIC_METHYL_TYPES.

    Returns:
        A list of dictionaries, where each dictionary represents a peak
        and its associated properties.
    """
    peaks = []
    if not os.path.exists(file_path):
        print(f"Warning: Peak list file not found: {file_path}")
        return peaks

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip() or line.strip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                print(f"Warning: Skipping malformed line {i+1} in {file_path}: {line.strip()}")
                continue
            
            peak_id = parts[0]
            if filter_specific_types and not is_specific_methyl_type_exp(peak_id):
                continue

            peak = {
                'id': peak_id,
                'c_shift': float(parts[1]),
                'h_shift': float(parts[2]),
                'type': parts[3] if len(parts) > 3 else None,
                'comment': ' '.join(parts[4:]) if len(parts) > 4 else None
            }
            peaks.append(peak)
    return peaks

def load_cch_noesy_peak_list(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a 3D CCH-NOESY peak list file.

    Args:
        file_path: The path to the CCH-NOESY peak list file.

    Returns:
        A list of dictionaries, where each dictionary represents an NOE cross-peak.
    """
    peaks = []
    if not os.path.exists(file_path):
        print(f"Warning: Peak list file not found: {file_path}")
        return peaks

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip() or line.strip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"Warning: Skipping malformed line {i+1} in {file_path}: {line.strip()}")
                continue

            peak = {
                'assignment': parts[0],
                'c_shift_donor': float(parts[1]),
                'c_shift_acceptor': float(parts[2]),
                'h_shift_acceptor': float(parts[3]),
                'intensity': float(parts[4])
            }
            peaks.append(peak)
    return peaks

def load_dataset(directory_path: str, filter_specific_types: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads HMQC and CCH-NOESY peak lists from a given directory.
    It expects to find one '*_hmqc.list' and one '*_cch.list' file.

    Args:
        directory_path: The path to the directory containing peak list files.
        filter_specific_types: If True, only load peaks matching SPECIFIC_METHYL_TYPES.

    Returns:
        A tuple containing the loaded HMQC peak list and CCH-NOESY peak list.
        Returns ([], []) if files are not found.
    """
    hmqc_peaks = []
    cch_peaks = []
    
    if not os.path.isdir(directory_path):
        print(f"Warning: Dataset directory not found: {directory_path}")
        return hmqc_peaks, cch_peaks

    hmqc_file = next((f for f in os.listdir(directory_path) if f.endswith('hmqc.list')), None)
    cch_file = next((f for f in os.listdir(directory_path) if f.endswith('_cch.list')), None)

    if not hmqc_file:
        hmqc_file = next((f for f in os.listdir(directory_path) if f.endswith('hmqc_rmI22_mx9_mx7.list')), None)
    if not cch_file:
        cch_file = next((f for f in os.listdir(directory_path) if f.endswith('_purged.list')), None)

    if hmqc_file:
        hmqc_peaks = load_hmqc_peak_list(os.path.join(directory_path, hmqc_file), filter_specific_types)
    else:
        print(f"Warning: No HMQC file ('*_hmqc.list') found in {directory_path}")

    if cch_file:
        cch_peaks = load_cch_noesy_peak_list(os.path.join(directory_path, cch_file))
    else:
        print(f"Warning: No CCH-NOESY file ('*_cch.list') found in {directory_path}")

    return hmqc_peaks, cch_peaks

def preprocess_data(
    hmqc_peaks: List[Dict[str, Any]],
    cch_peaks: List[Dict[str, Any]],
    tolerance: float = 0.03
) -> nx.Graph:
    """
    Constructs a graph from HMQC and CCH-NOESY data.

    Nodes are methyl groups from the HMQC list.
    Edges are NOE connections from the CCH-NOESY list.

    Args:
        hmqc_peaks: A list of peaks from an HMQC file.
        cch_peaks: A list of peaks from a CCH-NOESY file.
        tolerance: The chemical shift tolerance for matching peaks.

    Returns:
        A networkx graph representing the NOE network.
    """
    G = nx.Graph()

    # Add nodes from HMQC data
    for i, peak in enumerate(hmqc_peaks):
        G.add_node(i, **peak)

    # Helper to find a node in the graph based on chemical shifts
    def find_node(c_shift, h_shift):
        for node_idx, data in G.nodes(data=True):
            if (abs(data['c_shift'] - c_shift) < tolerance and
                abs(data['h_shift'] - h_shift) < tolerance):
                return node_idx
        return None

    # Add edges from CCH-NOESY data
    for noe in cch_peaks:
        # The CCH NOESY gives C-C-H shifts. The donor C is matched against the HMQC C shift.
        # The acceptor C and H are matched against the HMQC C and H shifts.
        donor_c = noe['c_shift_donor']
        acceptor_c = noe['c_shift_acceptor']
        acceptor_h = noe['h_shift_acceptor']
        
        # Find the acceptor node by matching C and H shifts
        acceptor_node = find_node(acceptor_c, acceptor_h)

        if acceptor_node is None:
            continue

        # Find potential donor nodes by matching only the C shift
        potential_donors = [
            node_idx for node_idx, data in G.nodes(data=True)
            if abs(data['c_shift'] - donor_c) < tolerance
        ]

        for donor_node in potential_donors:
            if donor_node != acceptor_node:
                # Add edge with intensity as a weight
                if G.has_edge(donor_node, acceptor_node):
                    # If edge exists, add intensity to the weight
                    G[donor_node][acceptor_node]['weight'] += noe['intensity']
                else:
                    G.add_edge(donor_node, acceptor_node, weight=noe['intensity'])

    return G

def normalize_label(label: str) -> str:
    """
    Normalizes a label from experimental data (e.g., 'A47CB-HB') to a
    standardized format (e.g., 'ALA47-CB') for matching with PDB labels.
    """
    # Mapping from single-letter to three-letter amino acid codes
    aa_code_map = {
        'A': 'ALA', 'V': 'VAL', 'L': 'LEU', 'I': 'ILE', 'M': 'MET', 'T': 'THR'
        # Add other amino acids if needed
    }
    
    # Heuristic parsing of the experimental label
    # This might need to be adjusted if the label format varies.
    import re
    match = re.match(r"([A-Z])(\d+)([A-Z0-9]+)", label)
    if not match:
        return None
        
    aa_code, res_num, atom_part = match.groups()
    
    if aa_code not in aa_code_map:
        return None
        
    three_letter_aa = aa_code_map[aa_code]
    
    # Simplify atom name (e.g., 'CB-HB' -> 'CB')
    atom_name = atom_part.split('-')[0]
    
    return f"{three_letter_aa}{res_num}-{atom_name}"

def establish_ground_truth_mapping(
    exp_graph: nx.Graph,
    gt_graph: nx.Graph
) -> Dict[int, int]:
    """
    Establishes a pseudo ground truth mapping between an experimental graph
    and a ground truth graph. 
    
    For now, this function creates a synthetic identity mapping for the purpose
    of developing the training loop. This will be replaced with a more
    realistic mapping later.

    Args:
        exp_graph: The experimental graph from NMR data.
        gt_graph: The ground truth graph from PDB.

    Returns:
        A dictionary mapping node indices from exp_graph to gt_graph.
    """
    mapping = {}
    num_nodes_to_map = min(exp_graph.number_of_nodes(), gt_graph.number_of_nodes())
    for i in range(num_nodes_to_map):
        mapping[i] = i
            
    return mapping
