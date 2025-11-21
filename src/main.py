"""
Main script for training, evaluation, and inference of the
Graph Matching Model.
"""

import os
import torch
import torch.optim as optim
import networkx as nx
from src.data_preprocessing.loader import load_dataset, preprocess_data, establish_ground_truth_mapping, load_hmqc_peak_list, load_cch_noesy_peak_list
from src.data_preprocessing.pdb_parser import calculate_noe_network, preprocess_pdb_file
from src.graph_matching.model import GraphMatchingModel
from src.graph_matching.loss import GraphMatchingLoss
from torch_geometric.data import Data

def nx_to_pyg_data(graph: nx.Graph) -> Data:
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.
    """
    # Create a mapping from node in NetworkX to a continuous index
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    
    x = torch.tensor([[data['c_shift'], data['h_shift']] for _, data in graph.nodes(data=True)], dtype=torch.float32)

    # Remap edges to the new continuous indices
    edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edges).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # Determine edge_attr based on graph type
    if graph.number_of_edges() > 0:
        first_edge = list(graph.edges)[0]
        if 'weight' in graph[first_edge[0]][first_edge[1]]:
            edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges()], dtype=torch.float32).unsqueeze(-1)
        elif 'distance' in graph[first_edge[0]][first_edge[1]]:
            edge_attr = torch.tensor([graph[u][v]['distance'] for u, v in graph.edges()], dtype=torch.float32).unsqueeze(-1)
        else:
            edge_attr = None
    else:
        edge_attr = None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def run_training_pipeline(
    model: GraphMatchingModel,
    optimizer: optim.Optimizer,
    loss_fn: GraphMatchingLoss,
    positive_pair: tuple[Data, Data],
    negative_pair: tuple[Data, Data],
    epochs: int = 100
):
    """
    Runs the training loop for the Graph Matching Model.
    """
    print("\n--- Starting Training Loop ---")
    model.train() # Set model to training mode

    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Positive Pair ---
        pred_sim_pos = model(positive_pair[0], positive_pair[1])
        target_sim_pos = torch.tensor([[1.0]], device=pred_sim_pos.device) # Similar graphs
        loss_pos = loss_fn(pred_sim_pos, target_sim_pos)

        # --- Negative Pair ---
        pred_sim_neg = model(negative_pair[0], negative_pair[1])
        target_sim_neg = torch.tensor([[0.0]], device=pred_sim_neg.device) # Dissimilar graphs
        loss_neg = loss_fn(pred_sim_neg, target_sim_neg)
        
        # --- Total Loss ---
        total_loss = loss_pos + loss_neg

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f} (Pos: {loss_pos.item():.4f}, Neg: {loss_neg.item():.4f})")
            
    print("--- Training Loop Finished ---\n")


def run_evaluation_pipeline(
    model: GraphMatchingModel,
    positive_pair: tuple[Data, Data],
    negative_pair: tuple[Data, Data]
):
    """
    Runs the evaluation pipeline for the Graph Matching Model.
    """
    print("\n--- Starting Evaluation ---")
    model.eval() # Set model to evaluation mode
    
    with torch.no_grad():
        # Evaluate on positive pair
        pred_sim_pos = model(positive_pair[0], positive_pair[1])
        print(f"Predicted similarity for POSITIVE pair (exp vs gt): {pred_sim_pos.item():.4f}")
        
        # Evaluate on negative pair
        pred_sim_neg = model(negative_pair[0], negative_pair[1])
        print(f"Predicted similarity for NEGATIVE pair (exp vs random): {pred_sim_neg.item():.4f}")

    print("--- Evaluation Finished ---\n")

def run_inference_pipeline(
    model: GraphMatchingModel,
    hmqc_path: str,
    cch_path: str,
    pdb_path: str,
    output_path: str
):
    """
    Runs inference on a new data sample and saves the assignments to a file.
    """
    print("\n--- Starting Inference ---")
    model.eval()

    # 1. Load and preprocess inference data with filtering
    print("Loading and preprocessing inference NMR data (with methyl type filter)...")
    hmqc_peaks, cch_peaks = load_dataset_from_files(hmqc_path, cch_path, filter_specific_types=True)
    if not hmqc_peaks or not cch_peaks:
        print("Failed to load inference data. Exiting.")
        return

    experimental_graph = preprocess_data(hmqc_peaks, cch_peaks)
    print(f"Inference experimental graph constructed with {experimental_graph.number_of_nodes()} nodes and {experimental_graph.number_of_edges()} edges.")

    # 2. Generate ground truth graph for inference with filtering
    print("Preprocessing PDB file for inference (with methyl type filter)...")
    preprocessed_pdb_path = pdb_path.replace(".pdb", "_preprocessed.pdb")
    preprocess_pdb_file(pdb_path, preprocessed_pdb_path)
    ground_truth_graph = calculate_noe_network(preprocessed_pdb_path, filter_specific_methyl_types=True)
    os.remove(preprocessed_pdb_path) # Clean up
    print(f"Inference ground truth graph constructed with {ground_truth_graph.number_of_nodes()} nodes and {ground_truth_graph.number_of_edges()} edges.")
    
    # 3. Get assignments
    if experimental_graph.number_of_nodes() > 0 and ground_truth_graph.number_of_nodes() > 0:
        exp_data_pyg = nx_to_pyg_data(experimental_graph)
        gt_data_pyg = nx_to_pyg_data(ground_truth_graph)
        
        assignments = model.get_assignments(exp_data_pyg, gt_data_pyg)
        
        # 4. Save assignments to file
        with open(output_path, 'w') as f:
            f.write("Experimental_Peak_ID\tAssigned_PDB_Label\n")
            for exp_node_idx, gt_node_idx in assignments.items():
                exp_peak_id = experimental_graph.nodes[exp_node_idx]['id']
                gt_label = ground_truth_graph.nodes[gt_node_idx]['label']
                f.write(f"{exp_peak_id}\t{gt_label}\n")
        print(f"\nInference assignments saved to {output_path}")

    else:
        print("One of the graphs for inference is empty.")

    print("--- Inference Finished ---\n")

def load_dataset_from_files(hmqc_path: str, cch_path: str, filter_specific_types: bool = False):
    """
    Helper function to load a dataset from specific HMQC and CCH files.
    """
    hmqc_peaks = load_hmqc_peak_list(hmqc_path, filter_specific_types)
    cch_peaks = load_cch_noesy_peak_list(cch_path) # CCH list doesn't need filtering by type
    return hmqc_peaks, cch_peaks


if __name__ == "__main__":
    # --- Training Phase ---
    training_data_dir = "data/train"
    training_pdb_path = "data/1IEP.pdb"
    
    print("--- Preparing for Training ---")
    # For training, we still use all methyl types in the training data
    hmqc_peaks_train, cch_peaks_train = load_dataset(training_data_dir, filter_specific_types=False)
    
    if not hmqc_peaks_train or not cch_peaks_train:
        print("Failed to load training data. Exiting.")
    else:
        exp_graph_train = preprocess_data(hmqc_peaks_train, cch_peaks_train)
        
        preprocessed_pdb_path_train = training_pdb_path.replace(".pdb", "_preprocessed.pdb")
        preprocess_pdb_file(training_pdb_path, preprocessed_pdb_path_train)
        gt_graph_train = calculate_noe_network(preprocessed_pdb_path_train, filter_specific_methyl_types=False)
        os.remove(preprocessed_pdb_path_train)

        random_graph_train = nx.gnp_random_graph(n=50, p=0.3, seed=42)
        for node in random_graph_train.nodes():
            random_graph_train.nodes[node]['c_shift'] = torch.randn(1).item()
            random_graph_train.nodes[node]['h_shift'] = torch.randn(1).item()
        
        if exp_graph_train.number_of_nodes() > 0 and gt_graph_train.number_of_nodes() > 0:
            exp_data_pyg_train = nx_to_pyg_data(exp_graph_train)
            gt_data_pyg_train = nx_to_pyg_data(gt_graph_train)
            random_data_pyg_train = nx_to_pyg_data(random_graph_train)
            
            positive_pair_train = (exp_data_pyg_train, gt_data_pyg_train)
            negative_pair_train = (exp_data_pyg_train, random_data_pyg_train)

            feature_dim = 2
            model = GraphMatchingModel(in_channels=feature_dim, hidden_channels=32, out_channels=16, num_layers=3)
            loss_fn = GraphMatchingLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            run_training_pipeline(model, optimizer, loss_fn, positive_pair_train, negative_pair_train, epochs=100)
            run_evaluation_pipeline(model, positive_pair_train, negative_pair_train)

            # --- Inference Phase ---
            inference_hmqc_path = "data/inference/yme1l_201/hmqc.list"
            inference_cch_path = "data/inference/yme1l_201/20210716-YME1L_AAA-800-3d_methyl_nosey_manual_pick_all_wIntensity_nohead_purged.list"
            inference_pdb_path = "data/inference/yme1l_201/AF3_yme1l_201_cordinates_only.pdb"
            inference_output_path = "data/inference/yme1l_201/result.txt"

            run_inference_pipeline(model, inference_hmqc_path, inference_cch_path, inference_pdb_path, inference_output_path)
        else:
            print("One of the training graphs is empty, cannot proceed.")
