#!/usr/bin/env python
"""Prediction script for NMR methyl assignment."""

import argparse
import torch
import yaml
from pathlib import Path

from nmr_graph_matching import (
    DGMCModel,
    PDBParser,
    HMQCParser,
    NOESYParser,
    MethylNetworkBuilder,
    PeakNetworkBuilder,
    OutputFormatter
)


def load_model(checkpoint_path: str, config: dict, device: str) -> DGMCModel:
    """Load trained model from checkpoint."""
    model_config = config['model']

    model = DGMCModel(
        peak_feature_dim=model_config.get('peak_feature_dim', 4),
        methyl_feature_dim=model_config.get('methyl_feature_dim', 11),
        embedding_dim=model_config.get('embedding_dim', 64),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_encoder_layers=model_config.get('num_encoder_layers', 3),
        num_consensus_layers=model_config.get('num_consensus_layers', 2),
        gnn_type=model_config.get('gnn_type', 'gcn'),
        dropout=model_config.get('dropout', 0.1),
        sinkhorn_iterations=model_config.get('sinkhorn_iterations', 20),
        sinkhorn_tau=model_config.get('sinkhorn_tau', 0.05)
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Predict NMR methyl assignments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--pdb', type=str, required=True,
                       help='Path to PDB file')
    parser.add_argument('--hmqc', type=str, required=True,
                       help='Path to HMQC peak list')
    parser.add_argument('--noesy', type=str, required=True,
                       help='Path to NOESY peak list')
    parser.add_argument('--output', type=str, default='assignments.txt',
                       help='Output file path')
    parser.add_argument('--format', type=str, default='text',
                       choices=['text', 'csv', 'both'],
                       help='Output format')
    parser.add_argument('--pymol', action='store_true',
                       help='Generate PyMOL visualization script')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for inference (cuda or cpu)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, config, device)

    print("\n" + "="*60)
    print("Building graphs from input data...")
    print("="*60)

    # Build methyl network
    data_config = config.get('data', {})
    methyl_builder = MethylNetworkBuilder(
        distance_cutoff=data_config.get('distance_cutoff', 10.0)
    )
    methyl_graph = methyl_builder.build_from_pdb(args.pdb)
    methyl_graph = methyl_graph.to(device)

    print(f"Methyl network: {methyl_graph.num_nodes} nodes, "
          f"{methyl_graph.edge_index.size(1)} edges")

    # Parse NMR data
    hmqc_parser = HMQCParser()
    hmqc_peaks = hmqc_parser.parse(args.hmqc)

    noesy_parser = NOESYParser()
    noesy_peaks = noesy_parser.parse(args.noesy)
    crosspeaks = noesy_parser.match_with_hmqc(hmqc_peaks)

    # Build peak network
    peak_builder = PeakNetworkBuilder(
        shift_normalization=data_config.get('shift_normalization', 'standard'),
        confidence_threshold=data_config.get('confidence_threshold', 0.0)
    )
    peak_graph = peak_builder.build(hmqc_peaks, crosspeaks)
    peak_graph = peak_graph.to(device)

    print(f"Peak network: {peak_graph.num_nodes} nodes, "
          f"{peak_graph.edge_index.size(1)} edges")

    # Predict assignments
    print("\n" + "="*60)
    print("Running prediction...")
    print("="*60)

    with torch.no_grad():
        matching_matrix, _ = model(peak_graph, methyl_graph)

    # Format output
    formatter = OutputFormatter(
        confidence_threshold=0.5,
        top_k_alternatives=3
    )

    # Get peak shifts
    peak_shifts = torch.stack([
        peak_graph.x[:, 0],  # H_shift
        peak_graph.x[:, 1]   # C_shift
    ], dim=1).cpu().numpy()

    # Get adjacency matrices for NOE completeness
    peak_adj = peak_builder.get_adjacency_matrix(peak_graph)
    methyl_adj = methyl_builder.get_adjacency_matrix(methyl_graph)

    results = formatter.format_assignments(
        matching_matrix=matching_matrix,
        peak_ids=peak_graph.peak_ids,
        peak_shifts=peak_shifts,
        methyl_names=methyl_graph.methyl_names,
        peak_adjacency=peak_adj,
        methyl_adjacency=methyl_adj
    )

    print(f"\nGenerated {len(results)} assignments")

    # Save output
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)

    if args.format in ['text', 'both']:
        text_output = formatter.to_text(results, include_alternatives=True)
        text_file = args.output if args.output.endswith('.txt') else args.output + '.txt'

        with open(text_file, 'w') as f:
            f.write(text_output)

        print(f"Text output saved to {text_file}")
        print("\n" + text_output)

    if args.format in ['csv', 'both']:
        csv_file = args.output.replace('.txt', '.csv') if args.output.endswith('.txt') else args.output + '.csv'
        formatter.to_csv(results, csv_file, include_alternatives=True)

    # Generate PyMOL script if requested
    if args.pymol:
        pymol_file = args.output.replace('.txt', '.pml').replace('.csv', '.pml')
        if not pymol_file.endswith('.pml'):
            pymol_file = pymol_file + '.pml'

        formatter.generate_pymol_script(
            results=results,
            pdb_file=args.pdb,
            output_file=pymol_file
        )

    print("\nPrediction completed successfully!")


if __name__ == '__main__':
    main()
