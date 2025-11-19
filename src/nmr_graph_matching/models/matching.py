"""Matching algorithms for graph node assignment."""

import torch
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linear_sum_assignment


class SinkhornMatching:
    """Sinkhorn algorithm for computing soft matching matrices.

    The Sinkhorn algorithm converts a similarity matrix into a doubly-stochastic
    matrix through iterative row and column normalization.
    """

    def __init__(self, n_iterations: int = 20, tau: float = 0.05, eps: float = 1e-8):
        """Initialize Sinkhorn matching.

        Args:
            n_iterations: Number of Sinkhorn iterations
            tau: Temperature parameter (lower = more discrete)
            eps: Small constant for numerical stability
        """
        self.n_iterations = n_iterations
        self.tau = tau
        self.eps = eps

    def __call__(
        self,
        similarity: torch.Tensor,
        dummy_row: bool = False,
        dummy_col: bool = False
    ) -> torch.Tensor:
        """Apply Sinkhorn algorithm.

        Args:
            similarity: Similarity matrix [n, m]
            dummy_row: Add dummy row for unmatched target nodes
            dummy_col: Add dummy column for unmatched source nodes

        Returns:
            Doubly-stochastic matching matrix [n, m]
        """
        # Apply temperature
        log_alpha = similarity / self.tau

        # Add dummy rows/columns if requested
        if dummy_row:
            dummy = torch.full(
                (1, log_alpha.size(1)),
                -float('inf'),
                dtype=log_alpha.dtype,
                device=log_alpha.device
            )
            log_alpha = torch.cat([log_alpha, dummy], dim=0)

        if dummy_col:
            dummy = torch.full(
                (log_alpha.size(0), 1),
                -float('inf'),
                dtype=log_alpha.dtype,
                device=log_alpha.device
            )
            log_alpha = torch.cat([log_alpha, dummy], dim=1)

        # Sinkhorn iterations
        for _ in range(self.n_iterations):
            # Normalize rows (dim=1)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            # Normalize columns (dim=0)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)

        # Convert back from log space
        matching = torch.exp(log_alpha)

        # Remove dummy rows/columns
        if dummy_row:
            matching = matching[:-1, :]
        if dummy_col:
            matching = matching[:, :-1]

        return matching

    def forward(self, similarity: torch.Tensor) -> torch.Tensor:
        """Alias for __call__."""
        return self(similarity)


class HungarianMatching:
    """Hungarian algorithm for optimal bipartite matching.

    Solves the linear assignment problem to find the optimal one-to-one
    matching that maximizes the total similarity score.
    """

    def __init__(self, maximize: bool = True):
        """Initialize Hungarian matching.

        Args:
            maximize: If True, maximize similarity; if False, minimize cost
        """
        self.maximize = maximize

    def __call__(
        self,
        similarity: torch.Tensor,
        return_cost: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[float]]:
        """Apply Hungarian algorithm.

        Args:
            similarity: Similarity matrix [n, m] where n <= m typically
            return_cost: Whether to return the total matching cost

        Returns:
            Tuple of:
                - row_indices: Source node indices [n]
                - col_indices: Target node indices [n]
                - total_cost: Optional total matching cost
        """
        # Convert to numpy
        if isinstance(similarity, torch.Tensor):
            sim_numpy = similarity.detach().cpu().numpy()
        else:
            sim_numpy = similarity

        # Create cost matrix (negate if maximizing)
        if self.maximize:
            cost_matrix = -sim_numpy
        else:
            cost_matrix = sim_numpy

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Convert back to tensors
        row_indices = torch.tensor(row_ind, dtype=torch.long)
        col_indices = torch.tensor(col_ind, dtype=torch.long)

        # Compute total cost if requested
        total_cost = None
        if return_cost:
            total_cost = float(sim_numpy[row_ind, col_ind].sum())

        return row_indices, col_indices, total_cost

    def create_assignment_matrix(
        self,
        similarity: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Create binary assignment matrix from Hungarian matching.

        Args:
            similarity: Similarity matrix [n, m]
            device: Device for output tensor

        Returns:
            Binary assignment matrix [n, m]
        """
        n, m = similarity.shape
        row_ind, col_ind, _ = self(similarity)

        if device is None:
            device = similarity.device

        # Create binary matrix
        assignment = torch.zeros((n, m), dtype=torch.float32, device=device)
        assignment[row_ind, col_ind] = 1.0

        return assignment


class GreedyMatching:
    """Greedy matching algorithm.

    Iteratively assigns each source node to its highest-scoring target node,
    with optional constraints to ensure one-to-one matching.
    """

    def __init__(self, one_to_one: bool = True):
        """Initialize greedy matching.

        Args:
            one_to_one: If True, enforce one-to-one matching constraint
        """
        self.one_to_one = one_to_one

    def __call__(
        self,
        similarity: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply greedy matching.

        Args:
            similarity: Similarity matrix [n, m]

        Returns:
            Tuple of:
                - assignments: Target index for each source [n]
                - confidences: Confidence scores [n]
        """
        n, m = similarity.shape

        if not self.one_to_one:
            # Simple: assign each source to best target
            confidences, assignments = torch.max(similarity, dim=1)
            return assignments, confidences

        # One-to-one matching: iteratively assign highest scores
        remaining_similarity = similarity.clone()
        assignments = torch.full((n,), -1, dtype=torch.long)
        confidences = torch.zeros(n)

        used_targets = set()

        # Flatten and sort all scores
        flat_sim = remaining_similarity.flatten()
        sorted_indices = torch.argsort(flat_sim, descending=True)

        for idx in sorted_indices:
            source_idx = idx // m
            target_idx = idx % m

            # Skip if already assigned
            if assignments[source_idx] != -1:
                continue
            if target_idx.item() in used_targets:
                continue

            # Make assignment
            assignments[source_idx] = target_idx
            confidences[source_idx] = similarity[source_idx, target_idx]
            used_targets.add(target_idx.item())

            # Stop if all sources assigned
            if len(used_targets) == n:
                break

        return assignments, confidences


def evaluate_matching(
    predicted_assignments: torch.Tensor,
    true_assignments: torch.Tensor,
    ignore_index: int = -1
) -> dict:
    """Evaluate matching accuracy.

    Args:
        predicted_assignments: Predicted target indices [n]
        true_assignments: Ground truth target indices [n]
        ignore_index: Index to ignore (unassigned nodes)

    Returns:
        Dictionary with evaluation metrics
    """
    # Filter out ignored indices
    mask = (true_assignments != ignore_index)
    pred = predicted_assignments[mask]
    true = true_assignments[mask]

    if len(true) == 0:
        return {
            "accuracy": 0.0,
            "num_correct": 0,
            "num_total": 0
        }

    # Compute accuracy
    correct = (pred == true).sum().item()
    total = len(true)
    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total
    }


def compute_matching_precision_recall(
    matching_matrix: torch.Tensor,
    ground_truth_edges: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """Compute precision and recall for edge matching.

    Args:
        matching_matrix: Soft matching matrix [n, m]
        ground_truth_edges: Ground truth edges [2, num_edges]
        threshold: Threshold for considering a match

    Returns:
        Dictionary with precision, recall, F1 score
    """
    # Binarize matching matrix
    binary_matching = (matching_matrix > threshold).float()

    # Count matches
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Get predicted edges
    pred_edges = torch.nonzero(binary_matching, as_tuple=False)

    # Convert ground truth to set for fast lookup
    gt_set = set()
    for i in range(ground_truth_edges.size(1)):
        src = ground_truth_edges[0, i].item()
        tgt = ground_truth_edges[1, i].item()
        gt_set.add((src, tgt))

    # Count true positives and false positives
    pred_set = set()
    for i in range(pred_edges.size(0)):
        src = pred_edges[i, 0].item()
        tgt = pred_edges[i, 1].item()
        pred_set.add((src, tgt))

        if (src, tgt) in gt_set:
            true_positives += 1
        else:
            false_positives += 1

    # Count false negatives
    false_negatives = len(gt_set) - true_positives

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }
