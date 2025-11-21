"""
Module for loading and preprocessing NMR data, specifically peak lists,
for methyl assignment.
"""

import os
from typing import List, Dict, Any, Tuple
import networkx as nx

def load_hmqc_peak_list(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a 2D HMQC peak list file.

    Args:
        file_path: The path to the HMQC peak list file.

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
            
            peak = {
                'id': parts[0],
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

def load_dataset(directory_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads HMQC and CCH-NOESY peak lists from a given directory.
    It expects to find one '*_hmqc.list' and one '*_cch.list' file.

    Args:
        directory_path: The path to the directory containing peak list files.

    Returns:
        A tuple containing the loaded HMQC peak list and CCH-NOESY peak list.
        Returns ([], []) if files are not found.
    """
    hmqc_peaks = []
    cch_peaks = []
    
    if not os.path.isdir(directory_path):
        print(f"Warning: Dataset directory not found: {directory_path}")
        return hmqc_peaks, cch_peaks

    hmqc_file = next((f for f in os.listdir(directory_path) if f.endswith('_hmqc.list')), None)
    cch_file = next((f for f in os.listdir(directory_path) if f.endswith('_cch.list')), None)

    if hmqc_file:
        hmqc_peaks = load_hmqc_peak_list(os.path.join(directory_path, hmqc_file))
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


if __name__ == '__main__':
    # Example usage:
    hmqc_data, cch_data = load_dataset('data/train')
    
    print(f"Loaded {len(hmqc_data)} HMQC peaks.")
    print(f"Loaded {len(cch_data)} CCH-NOESY peaks.")

    if hmqc_data and cch_data:
        graph = preprocess_data(hmqc_data, cch_data)
        print("\nGraph constructed:")
        print(f"  Number of nodes: {graph.number_of_nodes()}")
        print(f"  Number of edges: {graph.number_of_edges()}")

        # Print some node and edge info
        if graph.number_of_nodes() > 0:
            print("\nExample node:")
            node_idx = list(graph.nodes)[0]
            print(f"  Node {node_idx}: {graph.nodes[node_idx]}")
        
        if graph.number_of_edges() > 0:
            print("\nExample edge:")
            edge = list(graph.edges(data=True))[0]
            print(f"  Edge: {edge}")