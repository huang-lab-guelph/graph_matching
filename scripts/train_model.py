#!/usr/bin/env python
"""Training script for NMR graph matching model."""

import argparse
import yaml
import torch
from pathlib import Path

from nmr_graph_matching import DGMCModel, Trainer, NMRDataset


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> DGMCModel:
    """Create model from configuration."""
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

    return model


def main():
    parser = argparse.ArgumentParser(description='Train NMR graph matching model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--val-data-dir', type=str, default=None,
                       help='Directory containing validation data')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for training (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    # Create model
    model = create_model(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Prepare data list for training
    data_dir = Path(args.data_dir)
    train_data_list = []

    # Look for triplets of (PDB, HMQC, NOESY) files
    for pdb_file in sorted(data_dir.glob("*.pdb")):
        base_name = pdb_file.stem

        # Find corresponding HMQC and NOESY files
        hmqc_file = data_dir / f"{base_name}_hmqc.txt"
        noesy_file = data_dir / f"{base_name}_noesy.txt"

        if hmqc_file.exists() and noesy_file.exists():
            train_data_list.append({
                'pdb_file': str(pdb_file),
                'hmqc_file': str(hmqc_file),
                'noesy_file': str(noesy_file),
                'name': base_name
            })

    print(f"Found {len(train_data_list)} training samples")

    if len(train_data_list) == 0:
        print("ERROR: No training data found!")
        print("Expected files: <name>.pdb, <name>_hmqc.txt, <name>_noesy.txt")
        return

    # Create datasets
    data_config = config.get('data', {})
    train_dataset = NMRDataset(
        data_list=train_data_list,
        distance_cutoff=data_config.get('distance_cutoff', 10.0),
        shift_normalization=data_config.get('shift_normalization', 'standard'),
        confidence_threshold=data_config.get('confidence_threshold', 0.0)
    )

    # Validation dataset (if provided)
    val_dataset = None
    if args.val_data_dir:
        val_data_dir = Path(args.val_data_dir)
        val_data_list = []

        for pdb_file in sorted(val_data_dir.glob("*.pdb")):
            base_name = pdb_file.stem
            hmqc_file = val_data_dir / f"{base_name}_hmqc.txt"
            noesy_file = val_data_dir / f"{base_name}_noesy.txt"

            if hmqc_file.exists() and noesy_file.exists():
                val_data_list.append({
                    'pdb_file': str(pdb_file),
                    'hmqc_file': str(hmqc_file),
                    'noesy_file': str(noesy_file),
                    'name': base_name
                })

        print(f"Found {len(val_data_list)} validation samples")

        val_dataset = NMRDataset(
            data_list=val_data_list,
            distance_cutoff=data_config.get('distance_cutoff', 10.0),
            shift_normalization=data_config.get('shift_normalization', 'standard'),
            confidence_threshold=data_config.get('confidence_threshold', 0.0)
        )

    # Create trainer
    training_config = config['training']
    loss_weights = config.get('loss_weights', {})

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=training_config.get('learning_rate', 1e-3),
        weight_decay=training_config.get('weight_decay', 1e-4),
        batch_size=training_config.get('batch_size', 1),
        num_workers=training_config.get('num_workers', 0),
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        loss_weights=loss_weights
    )

    # Train
    num_epochs = args.epochs if args.epochs else training_config.get('num_epochs', 100)
    save_every = training_config.get('save_every', 10)
    validate_every = training_config.get('validate_every', 1)

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {args.device}")
    print(f"Checkpoint directory: {args.checkpoint_dir}\n")

    trainer.train(
        num_epochs=num_epochs,
        save_every=save_every,
        validate_every=validate_every
    )

    print("\nTraining completed!")


if __name__ == '__main__':
    main()
