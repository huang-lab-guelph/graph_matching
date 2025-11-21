"""
Module for parsing PDB files to extract methyl group information and
construct ground truth NOE network graphs.
"""

from Bio.PDB import PDBParser, Selection, NeighborSearch
import networkx as nx
import os # Import os for file operations
from typing import List, Dict, Any, Tuple

# List of standard amino acids that typically contain methyl groups
# This list can be expanded or refined based on specific needs.
METHYL_CONTAINING_RESIDUES = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'THR']

def preprocess_pdb_file(pdb_file_path: str, output_pdb_path: str) -> None:
    """
    Preprocesses a PDB file to fix common formatting issues for Bio.PDB parser.
    Specifically, it replaces tab characters with multiple spaces to help with column alignment.
    This is a less destructive approach than trying to reformat the whole line.

    Args:
        pdb_file_path: Path to the input PDB file.
        output_pdb_path: Path to save the preprocessed PDB file.
    """
    with open(pdb_file_path, 'r') as infile, open(output_pdb_path, 'w') as outfile:
        for line in infile:
            # Replace all tab characters with 4 spaces.
            # 4 spaces is a common tab width and might help align columns.
            processed_line = line.replace('\t', '    ')
            outfile.write(processed_line)


def get_methyl_atoms(model) -> List[Tuple[Any, str, str]]:
    """
    Identifies methyl group atoms (C and attached H) in a PDB model,
    considering only standard ATOM records and common methyl-containing residues.

    Args:
        model: A Bio.PDB Model object.

    Returns:
        A list of tuples, where each tuple contains (atom, residue_id_str, atom_name_str).
        The residue_id_str is formatted as "RESIDUE_NAME_RESIDUE_NUMBER".
    """
    methyl_atoms_info = []
    
    for residue in Selection.unfold_entities(model, 'R'):
        resname = residue.get_resname()
        res_id_tuple = residue.get_id() # (hetero_flag, sequence_identifier, insertion_code)

        # Only process standard ATOM residues (hetero_flag is ' ')
        if res_id_tuple[0] != ' ':
            continue

        # Check against METHYL_CONTAINING_RESIDUES list
        if resname not in METHYL_CONTAINING_RESIDUES:
            continue

        res_id_str = f"{resname}{res_id_tuple[1]}"
        
        for atom in residue.get_atoms():
            atom_name = atom.get_name()
            # Common methyl group atom names within these residues
            if atom_name.startswith(('CB', 'CG', 'CD', 'CE')): # Beta-carbons, Gamma, Delta, Epsilon carbons
                methyl_atoms_info.append((atom, res_id_str, atom_name))
    return methyl_atoms_info

def calculate_noe_network(
    pdb_file: str,
    distance_threshold: float = 6.0, # Typical NOE distance threshold in Angstroms
    min_atom_distance: float = 2.0 # Minimum distance to consider two atoms distinct (e.g., prevent intra-methyl contacts)
) -> nx.Graph:
    """
    Parses a PDB file, identifies methyl groups, and constructs a graph
    representing the theoretical NOE network based on inter-methyl distances.

    Args:
        pdb_file: Path to the PDB file.
        distance_threshold: Maximum distance (in Angstroms) between methyl groups
                            to consider an NOE contact.
        min_atom_distance: Minimum distance between atoms to consider them as part of
                           different methyl groups (prevents self-loops or intra-methyl NOEs).

    Returns:
        A networkx graph where nodes represent methyl groups and edges represent
        theoretical NOE contacts. Node attributes include 'residue_id', 'atom_name',
        'c_shift', 'h_shift' (placeholders for now). Edge attributes include distance.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    methyl_atoms_info = get_methyl_atoms(model)
    if not methyl_atoms_info:
        print(f"No methyl atoms found in standard methyl-containing residues in {pdb_file}.")
        return nx.Graph()

    # Create a mapping from (residue_id_str, atom_name_str) to node index
    methyl_node_map = {}
    node_idx_counter = 0
    ground_truth_graph = nx.Graph()

    # Add nodes to the graph
    for atom_obj, res_id_str, atom_name_str in methyl_atoms_info:
        node_key = f"{res_id_str}-{atom_name_str}"
        if node_key not in methyl_node_map:
            methyl_node_map[node_key] = node_idx_counter
            ground_truth_graph.add_node(
                node_idx_counter,
                label=node_key, # For easier identification
                residue_id=res_id_str,
                atom_name=atom_name_str,
                atom_obj=atom_obj, # Store atom object for distance calculation
                # Placeholders for c_shift and h_shift. These would ideally come from
                # BMRB or other prediction tools. For now, they are 0.0.
                c_shift=0.0,
                h_shift=0.0
            )
            node_idx_counter += 1

    # Perform neighbor search to find contacts
    all_atoms_in_structure = Selection.unfold_entities(model, 'A')
    ns = NeighborSearch(all_atoms_in_structure)

    # List of node indices and their corresponding atom objects for efficient iteration
    # Filter to only include atoms that are part of the methyl groups we added as nodes
    nodes_with_methyl_atoms = [(idx, ground_truth_graph.nodes[idx]['atom_obj']) 
                               for idx in ground_truth_graph.nodes 
                               if 'atom_obj' in ground_truth_graph.nodes[idx]]


    # Find theoretical NOE contacts between methyl groups
    for i in range(len(nodes_with_methyl_atoms)):
        node_idx_i, atom_i = nodes_with_methyl_atoms[i]
        
        # Search for neighbors of atom_i within the distance_threshold
        neighbors = ns.search(atom_i.get_coord(), distance_threshold)

        for neighbor_atom_obj in neighbors:
            # Check if the neighbor atom is also a methyl atom that we added to our graph
            for node_idx_j, atom_j in nodes_with_methyl_atoms:
                if atom_j == neighbor_atom_obj: # Match atom objects
                    # Ensure it's not the same node and atoms are not too close (intra-methyl or intra-group)
                    if node_idx_i != node_idx_j and (atom_i - atom_j) > min_atom_distance:
                        if not ground_truth_graph.has_edge(node_idx_i, node_idx_j):
                            distance = (atom_i - atom_j)
                            ground_truth_graph.add_edge(node_idx_i, node_idx_j, distance=distance)

    return ground_truth_graph

if __name__ == '__main__':
    # Example usage:
    original_pdb_file_path = "data/abl1-255no65.pdb"
    preprocessed_pdb_file_path = "data/abl1-255no65_preprocessed.pdb"

    # Ensure the PDB file exists for testing
    if not os.path.exists(original_pdb_file_path):
        print(f"Error: PDB file {original_pdb_file_path} not found. Please ensure it is in the 'data' directory.")
    else:
        print(f"Preprocessing PDB file: {original_pdb_file_path} -> {preprocessed_pdb_file_path}")
        preprocess_pdb_file(original_pdb_file_path, preprocessed_pdb_file_path)

        print(f"Processing preprocessed PDB file: {preprocessed_pdb_file_path}")
        ground_truth_graph = calculate_noe_network(preprocessed_pdb_file_path)

        print("\nGround Truth Graph constructed:")
        print(f"  Number of nodes: {ground_truth_graph.number_of_nodes()}")
        print(f"  Number of edges: {ground_truth_graph.number_of_edges()}")

        if ground_truth_graph.number_of_nodes() > 0:
            print("\nExample node (first 5):")
            for i, node_data in enumerate(ground_truth_graph.nodes(data=True)):
                if i >= 5: break
                print(f"  Node {node_data[0]}: Label={node_data[1]['label']}, Residue={node_data[1]['residue_id']}, Atom={node_data[1]['atom_name']}")
        
        if ground_truth_graph.number_of_edges() > 0:
            print("\nExample edge (first 5):")
            for i, edge_data in enumerate(ground_truth_graph.edges(data=True)):
                if i >= 5: break
                print(f"  Edge {edge_data[0]}-{edge_data[1]}: Distance={edge_data[2]['distance']:.2f}")
        
        # Clean up preprocessed file
        if os.path.exists(preprocessed_pdb_file_path):
            os.remove(preprocessed_pdb_file_path)
