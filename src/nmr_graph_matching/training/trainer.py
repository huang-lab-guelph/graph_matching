"""Training loop and evaluation for graph matching models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
from pathlib import Path
from tqdm import tqdm
import numpy as np

from ..models.dgmc import DGMCModel
from ..models.matching import HungarianMatching, evaluate_matching
from .losses import CombinedLoss
from .dataset import NMRDataset, collate_fn


class Trainer:
    """Trainer for NMR graph matching models."""

    def __init__(
        self,
        model: DGMCModel,
        train_dataset: NMRDataset,
        val_dataset: Optional[NMRDataset] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1,
        num_workers: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Optional[str] = None,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the trainer.

        Args:
            model: DGMC model to train
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            batch_size: Batch size (typically 1 for graph matching)
            num_workers: Number of data loading workers
            device: Device for training ('cuda' or 'cpu')
            checkpoint_dir: Directory for saving checkpoints
            loss_weights: Dictionary of loss component weights
        """
        self.model = model.to(device)
        self.device = device

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn
            )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss function
        if loss_weights is None:
            loss_weights = {
                'matching_weight': 1.0,
                'distance_weight': 0.1,
                'permutation_weight': 0.01
            }

        self.criterion = CombinedLoss(**loss_weights)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of average losses
        """
        self.model.train()

        epoch_losses = {
            'total': [],
            'matching': [],
            'distance': [],
            'permutation': []
        }

        for batch in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_loss_components = {k: [] for k in epoch_losses.keys()}

            # Process each sample in batch
            for i in range(len(batch['peak_graphs'])):
                peak_graph = batch['peak_graphs'][i].to(self.device)
                methyl_graph = batch['methyl_graphs'][i].to(self.device)
                ground_truth = batch['ground_truths'][i]

                if ground_truth is not None:
                    ground_truth = ground_truth.to(self.device)

                # Forward pass
                matching_matrix, _ = self.model(peak_graph, methyl_graph)

                # Prepare loss inputs
                from ..graphs.peak_network import PeakNetworkBuilder
                from ..graphs.methyl_network import MethylNetworkBuilder

                peak_builder = PeakNetworkBuilder()
                methyl_builder = MethylNetworkBuilder()

                peak_adj = torch.FloatTensor(
                    peak_builder.get_adjacency_matrix(peak_graph)
                ).to(self.device)

                # Compute methyl distances
                methyl_coords = methyl_graph.pos.cpu().numpy()
                methyl_dists = self._compute_distance_matrix(methyl_coords)
                methyl_dists = torch.FloatTensor(methyl_dists).to(self.device)

                # Compute losses
                losses = self.criterion(
                    matching_matrix=matching_matrix,
                    ground_truth=ground_truth,
                    peak_adjacency=peak_adj,
                    methyl_distances=methyl_dists
                )

                batch_loss += losses['total']

                # Track components
                for key in epoch_losses.keys():
                    if key in losses:
                        batch_loss_components[key].append(losses[key].item())

            # Backward pass
            batch_loss.backward()
            self.optimizer.step()

            # Record losses
            for key in epoch_losses.keys():
                if batch_loss_components[key]:
                    epoch_losses[key].append(np.mean(batch_loss_components[key]))

        # Average losses over epoch
        avg_losses = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}

        return avg_losses

    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        val_losses = []
        accuracies = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                for i in range(len(batch['peak_graphs'])):
                    peak_graph = batch['peak_graphs'][i].to(self.device)
                    methyl_graph = batch['methyl_graphs'][i].to(self.device)
                    ground_truth = batch['ground_truths'][i]

                    if ground_truth is None:
                        continue

                    ground_truth = ground_truth.to(self.device)

                    # Forward pass
                    matching_matrix, _ = self.model(peak_graph, methyl_graph)

                    # Compute loss
                    losses = self.criterion(
                        matching_matrix=matching_matrix,
                        ground_truth=ground_truth
                    )
                    val_losses.append(losses['total'].item())

                    # Compute accuracy
                    hungarian = HungarianMatching()
                    pred_assignments, _, _ = hungarian(matching_matrix)
                    true_assignments = ground_truth.argmax(dim=1)

                    metrics = evaluate_matching(pred_assignments, true_assignments)
                    accuracies.append(metrics['accuracy'])

        return {
            'val_loss': np.mean(val_losses) if val_losses else 0.0,
            'val_accuracy': np.mean(accuracies) if accuracies else 0.0
        }

    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        validate_every: int = 1
    ):
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Validate every N epochs
        """
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_losses = self.train_epoch()
            self.history['train_loss'].append(train_losses['total'])

            print(f"Train Loss: {train_losses['total']:.4f}")
            print(f"  Matching: {train_losses.get('matching', 0.0):.4f}, "
                  f"Distance: {train_losses.get('distance', 0.0):.4f}, "
                  f"Permutation: {train_losses.get('permutation', 0.0):.4f}")

            # Validate
            if (epoch + 1) % validate_every == 0:
                val_metrics = self.validate()
                if val_metrics:
                    self.history['val_loss'].append(val_metrics['val_loss'])
                    self.history['val_accuracy'].append(val_metrics['val_accuracy'])

                    print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")

                    # Save best model
                    if val_metrics['val_accuracy'] > self.best_val_accuracy:
                        self.best_val_accuracy = val_metrics['val_accuracy']
                        self.save_checkpoint('best_model.pt')
                        print(f"  New best model saved (accuracy: {self.best_val_accuracy:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_accuracy': self.best_val_accuracy
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")

        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_accuracy = checkpoint['best_val_accuracy']

        print(f"Checkpoint loaded from {path}")

    @staticmethod
    def _compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances
