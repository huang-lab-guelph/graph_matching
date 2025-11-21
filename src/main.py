"""
Main script for training, evaluation, and inference of the Methyl Assignment
Deep Learning Graph Matching project.
"""

import os
import torch
import networkx as nx
from src.data_preprocessing.loader import load_dataset, preprocess_data
from src.data_preprocessing.pdb_parser import calculate_noe_network, preprocess_pdb_file
from src.graph_matching.model import GraphMatchingModel
from typing import Dict, Any

def run_matching_pipeline(experimental_data_path: str, original_pdb_file_path: str):
    """
    Runs a graph matching pipeline.
    Matches an experimental graph (from NMR data) against a ground truth graph (from PDB).
    """
    print(f"--- Running Graph Matching Pipeline ---")
    print(f"Experimental data from: {experimental_data_path}")
    print(f"PDB file for ground truth: {original_pdb_file_path}")

    # 1. Load and preprocess experimental data
    print("Loading and preprocessing experimental NMR data...")
    hmqc_peaks, cch_peaks = load_dataset(experimental_data_path)
    if not hmqc_peaks or not cch_peaks:
        print(f"No HMQC or CCH-NOESY data found in {experimental_data_path}. Skipping matching pipeline.")
        return

    experimental_graph = preprocess_data(hmqc_peaks, cch_peaks)
    print(f"Experimental graph constructed with {experimental_graph.number_of_nodes()} nodes and {experimental_graph.number_of_edges()} edges.")

    if experimental_graph.number_of_nodes() == 0:
        print("Experimental graph is empty. Skipping matching pipeline.")
        return

    # 2. Preprocess PDB file and generate ground truth graph
    print("Preprocessing PDB file and generating ground truth graph...")
    if not os.path.exists(original_pdb_file_path):
        print(f"Error: PDB file not found at {original_pdb_file_path}. Cannot generate ground truth graph.")
        return

    preprocessed_pdb_path = original_pdb_file_path.replace(".pdb", "_preprocessed.pdb")
    preprocess_pdb_file(original_pdb_file_path, preprocessed_pdb_path)

    ground_truth_graph = calculate_noe_network(preprocessed_pdb_path)
    print(f"Ground truth graph constructed with {ground_truth_graph.number_of_nodes()} nodes and {ground_truth_graph.number_of_edges()} edges.")

    # Clean up preprocessed PDB file
    if os.path.exists(preprocessed_pdb_path):
        os.remove(preprocessed_pdb_path)

    if ground_truth_graph.number_of_nodes() == 0:
        print("Ground truth graph is empty. Skipping matching pipeline.")
        return

    # Initialize the model with GNN dimensions
    feature_dim = 2 # c_shift, h_shift
    model = GraphMatchingModel(in_channels=feature_dim, hidden_channels=16, out_channels=16)

    # 3. Perform graph matching
    print("Performing graph matching...")
    try:
        matching_matrix = model.predict(experimental_graph, ground_truth_graph)
        print("\nMatching Matrix (discrete permutation):\n", matching_matrix)
        print(f"Matching Matrix shape: {matching_matrix.shape}")

        # For proper evaluation, we would need to know the true mapping between
        # experimental_graph nodes and ground_truth_graph nodes.
        # This involves comparing peak IDs from HMQC to residue_id/atom_name from PDB.
        # For now, we'll just print a message indicating where evaluation would go.
        print("\nEvaluation would involve comparing this matching matrix with the true assignment.")

    except ValueError as e:
        print(f"Error during graph matching: {e}")

    print("Graph matching pipeline complete.")


if __name__ == "__main__":
    # Ensure dummy data directories and PDB file exist
    experimental_data_dir = "data/train"
    pdb_path = "data/1IEP.pdb"
    
    run_matching_pipeline(experimental_data_dir, pdb_path)
