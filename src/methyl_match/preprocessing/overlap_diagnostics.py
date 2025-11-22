"""
Overlap diagnostics for analyzing chemical shift ambiguity in NMR data.

This module provides utilities to quantify and report on chemical shift overlap
in HMQC peak lists, which directly impacts the ambiguity of NOESY peak matching.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from methyl_match.reading.hmqc_parser import HMQCPeak
from methyl_match.reading.noesy_parser import NOEPeak

logger = logging.getLogger(__name__)


def analyze_hmqc_overlap(
    hmqc_peaks: List[HMQCPeak],
    c_tolerance: float = 0.5,
    h_tolerance: float = 0.05
) -> Dict[str, any]:
    """
    Analyze chemical shift overlap in HMQC peak list.

    This function computes an overlap matrix showing how many other peaks
    each HMQC peak overlaps with within given tolerances.

    Args:
        hmqc_peaks: List of HMQCPeak objects
        c_tolerance: 13C chemical shift tolerance (ppm)
        h_tolerance: 1H chemical shift tolerance (ppm)

    Returns:
        Dictionary containing:
            - overlap_matrix_1d: n×n matrix of 1D (13C only) overlap
            - overlap_matrix_2d: n×n matrix of 2D (13C + 1H) overlap
            - overlapping_peaks_1d: List of (peak_idx, num_overlaps)
            - overlapping_peaks_2d: List of (peak_idx, num_overlaps)
            - overlap_stats_1d: Statistics dict
            - overlap_stats_2d: Statistics dict
            - recommendations: List of recommendations based on analysis

    Example:
        >>> from methyl_match.reading import HMQCParser
        >>> parser = HMQCParser("hmqc.txt")
        >>> peaks = parser.parse()
        >>> report = analyze_hmqc_overlap(peaks)
        >>> print(f"Severe overlap: {report['overlap_stats_1d']['severe_overlap_pct']:.1f}%")
    """
    n = len(hmqc_peaks)

    # Initialize overlap matrices
    overlap_1d = np.zeros((n, n), dtype=int)
    overlap_2d = np.zeros((n, n), dtype=int)

    # Compute pairwise overlaps
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # 1D overlap (13C only)
            c_diff = abs(hmqc_peaks[i].c_shift - hmqc_peaks[j].c_shift)
            if c_diff <= c_tolerance:
                overlap_1d[i, j] = 1

            # 2D overlap (13C and 1H)
            h_diff = abs(hmqc_peaks[i].h_shift - hmqc_peaks[j].h_shift)
            if c_diff <= c_tolerance and h_diff <= h_tolerance:
                overlap_2d[i, j] = 1

    # Count overlaps per peak
    overlapping_peaks_1d = [(i, np.sum(overlap_1d[i, :])) for i in range(n)]
    overlapping_peaks_2d = [(i, np.sum(overlap_2d[i, :])) for i in range(n)]

    # Sort by number of overlaps
    overlapping_peaks_1d.sort(key=lambda x: x[1], reverse=True)
    overlapping_peaks_2d.sort(key=lambda x: x[1], reverse=True)

    # Compute statistics
    overlap_counts_1d = [count for _, count in overlapping_peaks_1d]
    overlap_counts_2d = [count for _, count in overlapping_peaks_2d]

    peaks_with_overlap_1d = sum(1 for count in overlap_counts_1d if count > 0)
    peaks_with_overlap_2d = sum(1 for count in overlap_counts_2d if count > 0)

    # Categorize severity (1D)
    mild_1d = sum(1 for count in overlap_counts_1d if 1 <= count <= 2)
    moderate_1d = sum(1 for count in overlap_counts_1d if 3 <= count <= 5)
    severe_1d = sum(1 for count in overlap_counts_1d if count > 5)

    # Categorize severity (2D)
    mild_2d = sum(1 for count in overlap_counts_2d if 1 <= count <= 2)
    moderate_2d = sum(1 for count in overlap_counts_2d if 3 <= count <= 5)
    severe_2d = sum(1 for count in overlap_counts_2d if count > 5)

    overlap_stats_1d = {
        'total_peaks': n,
        'peaks_with_overlap': peaks_with_overlap_1d,
        'overlap_pct': 100.0 * peaks_with_overlap_1d / n if n > 0 else 0.0,
        'avg_overlaps_per_peak': np.mean(overlap_counts_1d) if overlap_counts_1d else 0.0,
        'max_overlaps': max(overlap_counts_1d) if overlap_counts_1d else 0,
        'mild_overlap': mild_1d,
        'moderate_overlap': moderate_1d,
        'severe_overlap': severe_1d,
        'severe_overlap_pct': 100.0 * severe_1d / n if n > 0 else 0.0,
    }

    overlap_stats_2d = {
        'total_peaks': n,
        'peaks_with_overlap': peaks_with_overlap_2d,
        'overlap_pct': 100.0 * peaks_with_overlap_2d / n if n > 0 else 0.0,
        'avg_overlaps_per_peak': np.mean(overlap_counts_2d) if overlap_counts_2d else 0.0,
        'max_overlaps': max(overlap_counts_2d) if overlap_counts_2d else 0,
        'mild_overlap': mild_2d,
        'moderate_overlap': moderate_2d,
        'severe_overlap': severe_2d,
        'severe_overlap_pct': 100.0 * severe_2d / n if n > 0 else 0.0,
    }

    # Generate recommendations
    recommendations = _generate_recommendations(overlap_stats_1d, overlap_stats_2d)

    return {
        'overlap_matrix_1d': overlap_1d,
        'overlap_matrix_2d': overlap_2d,
        'overlapping_peaks_1d': overlapping_peaks_1d,
        'overlapping_peaks_2d': overlapping_peaks_2d,
        'overlap_stats_1d': overlap_stats_1d,
        'overlap_stats_2d': overlap_stats_2d,
        'recommendations': recommendations,
    }


def analyze_noesy_ambiguity(
    hmqc_peaks: List[HMQCPeak],
    noe_peaks: List[NOEPeak],
    c_tolerance: float = 0.5,
    intensity_threshold: float = 0.0
) -> Dict[str, any]:
    """
    Predict ambiguity in NOESY peak matching before building the network.

    This function analyzes how many NOESY peaks will have ambiguous matches
    based on chemical shift overlap in the HMQC peak list.

    Args:
        hmqc_peaks: List of HMQCPeak objects
        noe_peaks: List of NOEPeak objects
        c_tolerance: 13C chemical shift tolerance (ppm)
        intensity_threshold: Minimum NOE intensity to consider

    Returns:
        Dictionary containing:
            - total_noe_peaks: Total number of NOE peaks
            - above_threshold: Number of NOE peaks above intensity threshold
            - predicted_matched: Predicted number of matched edges
            - predicted_unmatched_w1: Predicted unmatched w1 dimensions
            - predicted_unmatched_w2: Predicted unmatched w2 dimensions
            - predicted_ambiguous_w1: Predicted ambiguous w1 matches
            - predicted_ambiguous_w2: Predicted ambiguous w2 matches
            - ambiguity_distribution: Distribution of (w1_cand, w2_cand) pairs

    Example:
        >>> report = analyze_noesy_ambiguity(hmqc_peaks, noe_peaks)
        >>> print(f"Predicted ambiguous edges: {report['predicted_ambiguous_w1']}")
    """
    # Build 13C shift lookup for quick matching
    c_shifts = np.array([p.c_shift for p in hmqc_peaks])

    predicted_matched = 0
    predicted_unmatched_w1 = 0
    predicted_unmatched_w2 = 0
    predicted_ambiguous_w1 = 0
    predicted_ambiguous_w2 = 0

    ambiguity_distribution = {}

    for noe in noe_peaks:
        if noe.intensity < intensity_threshold:
            continue

        # Find candidates for w1
        w1_diffs = np.abs(c_shifts - noe.w1)
        w1_candidates = np.sum(w1_diffs <= c_tolerance)

        # Find candidates for w2
        w2_diffs = np.abs(c_shifts - noe.w2)
        w2_candidates = np.sum(w2_diffs <= c_tolerance)

        # Track statistics
        if w1_candidates == 0:
            predicted_unmatched_w1 += 1
        elif w1_candidates > 1:
            predicted_ambiguous_w1 += 1

        if w2_candidates == 0:
            predicted_unmatched_w2 += 1
        elif w2_candidates > 1:
            predicted_ambiguous_w2 += 1

        # Count as matched if both dimensions have at least one candidate
        if w1_candidates > 0 and w2_candidates > 0:
            predicted_matched += 1

            # Track ambiguity distribution
            key = (w1_candidates, w2_candidates)
            ambiguity_distribution[key] = ambiguity_distribution.get(key, 0) + 1

    above_threshold = sum(1 for noe in noe_peaks if noe.intensity >= intensity_threshold)

    return {
        'total_noe_peaks': len(noe_peaks),
        'above_threshold': above_threshold,
        'predicted_matched': predicted_matched,
        'predicted_unmatched_w1': predicted_unmatched_w1,
        'predicted_unmatched_w2': predicted_unmatched_w2,
        'predicted_ambiguous_w1': predicted_ambiguous_w1,
        'predicted_ambiguous_w2': predicted_ambiguous_w2,
        'ambiguity_distribution': ambiguity_distribution,
    }


def generate_overlap_report(
    hmqc_peaks: List[HMQCPeak],
    noe_peaks: Optional[List[NOEPeak]] = None,
    c_tolerance: float = 0.5,
    h_tolerance: float = 0.05,
    intensity_threshold: float = 0.0
) -> str:
    """
    Generate a comprehensive human-readable overlap diagnostic report.

    Args:
        hmqc_peaks: List of HMQCPeak objects
        noe_peaks: Optional list of NOEPeak objects
        c_tolerance: 13C chemical shift tolerance (ppm)
        h_tolerance: 1H chemical shift tolerance (ppm)
        intensity_threshold: Minimum NOE intensity to consider

    Returns:
        Formatted string report

    Example:
        >>> report = generate_overlap_report(hmqc_peaks, noe_peaks)
        >>> print(report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CHEMICAL SHIFT OVERLAP DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Analyze HMQC overlap
    hmqc_analysis = analyze_hmqc_overlap(hmqc_peaks, c_tolerance, h_tolerance)
    stats_1d = hmqc_analysis['overlap_stats_1d']
    stats_2d = hmqc_analysis['overlap_stats_2d']

    lines.append(f"HMQC Peak List: {stats_1d['total_peaks']} peaks")
    lines.append(f"Tolerance: 13C ±{c_tolerance} ppm, 1H ±{h_tolerance} ppm")
    lines.append("")

    lines.append("1D Overlap Analysis (13C only):")
    lines.append(f"  Peaks with overlap: {stats_1d['peaks_with_overlap']} ({stats_1d['overlap_pct']:.1f}%)")
    lines.append(f"  Average overlaps per peak: {stats_1d['avg_overlaps_per_peak']:.2f}")
    lines.append(f"  Maximum overlaps: {stats_1d['max_overlaps']}")
    lines.append(f"  Severity breakdown:")
    lines.append(f"    Mild (1-2 overlaps):     {stats_1d['mild_overlap']} peaks")
    lines.append(f"    Moderate (3-5 overlaps): {stats_1d['moderate_overlap']} peaks")
    lines.append(f"    Severe (>5 overlaps):    {stats_1d['severe_overlap']} peaks ({stats_1d['severe_overlap_pct']:.1f}%)")
    lines.append("")

    lines.append("2D Overlap Analysis (13C + 1H):")
    lines.append(f"  Peaks with overlap: {stats_2d['peaks_with_overlap']} ({stats_2d['overlap_pct']:.1f}%)")
    lines.append(f"  Average overlaps per peak: {stats_2d['avg_overlaps_per_peak']:.2f}")
    lines.append(f"  Maximum overlaps: {stats_2d['max_overlaps']}")
    lines.append(f"  Severity breakdown:")
    lines.append(f"    Mild (1-2 overlaps):     {stats_2d['mild_overlap']} peaks")
    lines.append(f"    Moderate (3-5 overlaps): {stats_2d['moderate_overlap']} peaks")
    lines.append(f"    Severe (>5 overlaps):    {stats_2d['severe_overlap']} peaks ({stats_2d['severe_overlap_pct']:.1f}%)")
    lines.append("")

    # Analyze NOESY ambiguity if provided
    if noe_peaks:
        noesy_analysis = analyze_noesy_ambiguity(
            hmqc_peaks, noe_peaks, c_tolerance, intensity_threshold
        )

        lines.append(f"NOESY Peak Matching Prediction:")
        lines.append(f"  Total NOE peaks: {noesy_analysis['total_noe_peaks']}")
        lines.append(f"  Above intensity threshold: {noesy_analysis['above_threshold']}")
        lines.append(f"  Predicted matched edges: {noesy_analysis['predicted_matched']}")
        lines.append(f"  Predicted unmatched w1: {noesy_analysis['predicted_unmatched_w1']}")
        lines.append(f"  Predicted unmatched w2: {noesy_analysis['predicted_unmatched_w2']}")
        lines.append(f"  Predicted ambiguous w1: {noesy_analysis['predicted_ambiguous_w1']} " +
                    f"({100*noesy_analysis['predicted_ambiguous_w1']/max(noesy_analysis['above_threshold'],1):.1f}%)")
        lines.append(f"  Predicted ambiguous w2: {noesy_analysis['predicted_ambiguous_w2']} " +
                    f"({100*noesy_analysis['predicted_ambiguous_w2']/max(noesy_analysis['above_threshold'],1):.1f}%)")
        lines.append("")

        # Show ambiguity distribution
        lines.append("  Ambiguity distribution (w1_candidates, w2_candidates): count")
        for key, count in sorted(noesy_analysis['ambiguity_distribution'].items(),
                                 key=lambda x: x[1], reverse=True)[:10]:
            lines.append(f"    {key}: {count} edges")
        lines.append("")

    # Recommendations
    lines.append("RECOMMENDATIONS:")
    for i, rec in enumerate(hmqc_analysis['recommendations'], 1):
        lines.append(f"  {i}. {rec}")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def _generate_recommendations(
    stats_1d: Dict[str, float],
    stats_2d: Dict[str, float]
) -> List[str]:
    """
    Generate recommendations based on overlap analysis.

    Args:
        stats_1d: 1D overlap statistics
        stats_2d: 2D overlap statistics

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Check 1D overlap severity
    if stats_1d['overlap_pct'] < 20:
        recommendations.append("Low overlap detected. Default 1D matching should work well.")
    elif stats_1d['overlap_pct'] < 50:
        recommendations.append("Moderate overlap detected. Consider using 'best match by distance' mode.")
    else:
        recommendations.append("High overlap detected (>50% of peaks). Strongly recommend advanced matching strategies.")

    # Check if 2D helps
    improvement = stats_1d['overlap_pct'] - stats_2d['overlap_pct']
    if improvement > 20:
        recommendations.append(f"2D matching significantly reduces overlap by {improvement:.1f}%. Enable use_2d_matching=True.")
    elif improvement > 10:
        recommendations.append(f"2D matching moderately reduces overlap by {improvement:.1f}%. Consider enabling use_2d_matching=True.")
    else:
        recommendations.append("2D matching provides minimal improvement. 1D matching may be sufficient.")

    # Check for severe cases
    if stats_1d['severe_overlap_pct'] > 30:
        recommendations.append("Severe overlap (>30% of peaks with >5 overlaps). Consider create_ambiguous_edges=True for multiple edge hypotheses.")

    # Check maximum overlaps
    if stats_1d['max_overlaps'] > 10:
        recommendations.append(f"Some peaks have extreme overlap ({stats_1d['max_overlaps']} overlaps). Review data quality and consider adjusting c_tolerance.")

    return recommendations


def log_overlap_diagnostics(
    hmqc_peaks: List[HMQCPeak],
    noe_peaks: Optional[List[NOEPeak]] = None,
    c_tolerance: float = 0.5,
    h_tolerance: float = 0.05,
    intensity_threshold: float = 0.0
):
    """
    Log overlap diagnostics to the logger.

    Args:
        hmqc_peaks: List of HMQCPeak objects
        noe_peaks: Optional list of NOEPeak objects
        c_tolerance: 13C chemical shift tolerance (ppm)
        h_tolerance: 1H chemical shift tolerance (ppm)
        intensity_threshold: Minimum NOE intensity to consider
    """
    report = generate_overlap_report(
        hmqc_peaks, noe_peaks, c_tolerance, h_tolerance, intensity_threshold
    )
    logger.info("\n" + report)
