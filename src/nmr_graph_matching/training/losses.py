"""Loss functions for graph matching training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class MatchingLoss(nn.Module):
    """Loss function for graph matching based on ground truth assignments.

    Computes cross-entropy loss between predicted matching matrix and
    ground truth assignments.
    """

    def __init__(
        self,
        reduction: str = "mean",
        ignore_unassigned: bool = True
    ):
        """Initialize matching loss.

        Args:
            reduction: Loss reduction method ('mean', 'sum', 'none')
            ignore_unassigned: Whether to ignore peaks without ground truth
        """
        super(MatchingLoss, self).__init__()
        self.reduction = reduction
        self.ignore_unassigned = ignore_unassigned

    def forward(
        self,
        matching_matrix: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Compute matching loss.

        Args:
            matching_matrix: Predicted matching matrix [num_peaks, num_methyls]
            ground_truth: Ground truth assignment matrix [num_peaks, num_methyls]

        Returns:
            Scalar loss value
        """
        # Apply log-softmax for cross-entropy
        log_probs = F.log_softmax(matching_matrix, dim=1)

        # Compute negative log-likelihood
        loss = -(ground_truth * log_probs).sum(dim=1)

        # Mask out unassigned peaks if requested
        if self.ignore_unassigned:
            # Peaks with no assignment have all zeros in ground truth
            has_assignment = ground_truth.sum(dim=1) > 0
            loss = loss[has_assignment]

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean() if len(loss) > 0 else torch.tensor(0.0, device=loss.device)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DistanceConsistencyLoss(nn.Module):
    """Distance consistency loss for graph matching.

    Penalizes assignments that violate NOE-distance correlation.
    If two peaks have a strong NOE, their assigned methyls should be close
    in 3D space.
    """

    def __init__(
        self,
        distance_threshold: float = 10.0,
        margin: float = 2.0,
        reduction: str = "mean"
    ):
        """Initialize distance consistency loss.

        Args:
            distance_threshold: Expected maximum distance for NOE (Å)
            margin: Margin for the constraint violation
            reduction: Loss reduction method
        """
        super(DistanceConsistencyLoss, self).__init__()
        self.distance_threshold = distance_threshold
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        matching_matrix: torch.Tensor,
        peak_adjacency: torch.Tensor,
        methyl_distances: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance consistency loss.

        Args:
            matching_matrix: Predicted matching matrix [num_peaks, num_methyls]
            peak_adjacency: Peak network adjacency [num_peaks, num_peaks]
            methyl_distances: Methyl distance matrix [num_methyls, num_methyls]

        Returns:
            Scalar loss value
        """
        num_peaks = matching_matrix.size(0)

        # Compute expected distance matrix for peaks based on matching
        # D_expected[i,j] = sum_k sum_l M[i,k] * M[j,l] * D_methyl[k,l]
        expected_distances = torch.mm(
            torch.mm(matching_matrix, methyl_distances),
            matching_matrix.t()
        )

        # For pairs with strong NOE correlation, penalize if distance is too large
        # Find peak pairs with NOE correlations
        noe_mask = peak_adjacency > 0

        # Compute violations: expected distance > threshold + margin
        violations = F.relu(
            expected_distances - (self.distance_threshold + self.margin)
        )

        # Weight by NOE strength
        weighted_violations = violations * peak_adjacency * noe_mask.float()

        # Sum over all pairs and normalize
        loss = weighted_violations.sum() / (noe_mask.sum() + 1e-6)

        return loss


class PermutationLoss(nn.Module):
    """Permutation loss based on doubly-stochastic constraint.

    Encourages the matching matrix to be doubly-stochastic (rows and columns
    sum to 1), which is a relaxation of the permutation matrix constraint.
    """

    def __init__(self, reduction: str = "mean"):
        """Initialize permutation loss.

        Args:
            reduction: Loss reduction method
        """
        super(PermutationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, matching_matrix: torch.Tensor) -> torch.Tensor:
        """Compute permutation loss.

        Args:
            matching_matrix: Predicted matching matrix [num_peaks, num_methyls]

        Returns:
            Scalar loss value
        """
        # Row-wise L1 deviation from sum=1
        row_sums = matching_matrix.sum(dim=1)
        row_loss = F.l1_loss(row_sums, torch.ones_like(row_sums))

        # Column-wise L1 deviation from sum=1
        col_sums = matching_matrix.sum(dim=0)
        col_loss = F.l1_loss(col_sums, torch.ones_like(col_sums))

        return row_loss + col_loss


class CombinedLoss(nn.Module):
    """Combined loss function for graph matching training.

    Combines multiple loss components:
    - Matching loss (cross-entropy with ground truth)
    - Distance consistency loss
    - Permutation loss
    """

    def __init__(
        self,
        matching_weight: float = 1.0,
        distance_weight: float = 0.1,
        permutation_weight: float = 0.01,
        distance_threshold: float = 10.0
    ):
        """Initialize combined loss.

        Args:
            matching_weight: Weight for matching loss
            distance_weight: Weight for distance consistency loss
            permutation_weight: Weight for permutation loss
            distance_threshold: Distance threshold for consistency loss
        """
        super(CombinedLoss, self).__init__()

        self.matching_weight = matching_weight
        self.distance_weight = distance_weight
        self.permutation_weight = permutation_weight

        self.matching_loss = MatchingLoss()
        self.distance_loss = DistanceConsistencyLoss(
            distance_threshold=distance_threshold
        )
        self.permutation_loss = PermutationLoss()

    def forward(
        self,
        matching_matrix: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        peak_adjacency: Optional[torch.Tensor] = None,
        methyl_distances: Optional[torch.Tensor] = None
    ) -> dict:
        """Compute combined loss.

        Args:
            matching_matrix: Predicted matching matrix [num_peaks, num_methyls]
            ground_truth: Ground truth assignments [num_peaks, num_methyls]
            peak_adjacency: Peak network adjacency [num_peaks, num_peaks]
            methyl_distances: Methyl distance matrix [num_methyls, num_methyls]

        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        total_loss = 0.0

        # Matching loss (if ground truth available)
        if ground_truth is not None:
            match_loss = self.matching_loss(matching_matrix, ground_truth)
            losses['matching'] = match_loss
            total_loss += self.matching_weight * match_loss

        # Distance consistency loss
        if peak_adjacency is not None and methyl_distances is not None:
            dist_loss = self.distance_loss(
                matching_matrix,
                peak_adjacency,
                methyl_distances
            )
            losses['distance'] = dist_loss
            total_loss += self.distance_weight * dist_loss

        # Permutation loss
        perm_loss = self.permutation_loss(matching_matrix)
        losses['permutation'] = perm_loss
        total_loss += self.permutation_weight * perm_loss

        losses['total'] = total_loss

        return losses


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning node embeddings.

    Encourages embeddings of matched nodes to be similar and embeddings
    of non-matched nodes to be dissimilar.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0
    ):
        """Initialize contrastive loss.

        Args:
            temperature: Temperature for similarity scaling
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        embeddings_s: torch.Tensor,
        embeddings_t: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            embeddings_s: Source node embeddings [num_source, embedding_dim]
            embeddings_t: Target node embeddings [num_target, embedding_dim]
            ground_truth: Ground truth matching [num_source, num_target]

        Returns:
            Scalar loss value
        """
        # Compute similarity matrix
        similarity = torch.mm(embeddings_s, embeddings_t.t()) / self.temperature

        # Positive pairs (matched nodes)
        pos_mask = ground_truth > 0
        pos_similarity = similarity[pos_mask]

        # Negative pairs (non-matched nodes)
        neg_mask = ~pos_mask
        neg_similarity = similarity[neg_mask]

        # Loss: maximize positive similarity, minimize negative similarity
        pos_loss = -pos_similarity.mean() if pos_mask.sum() > 0 else 0.0
        neg_loss = F.relu(neg_similarity - self.margin).mean() if neg_mask.sum() > 0 else 0.0

        return pos_loss + neg_loss
