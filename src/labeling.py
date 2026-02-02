"""Error classification with full distribution tracking for sensitivity analysis."""

from typing import List, Dict, Literal

from .schemas import EquivalenceStats, EquivalenceJudgment, ErrorLabel


def compute_equivalence_stats(
    judgments: List[EquivalenceJudgment]
) -> EquivalenceStats:
    """
    Compute statistics from a list of equivalence judgments.
    
    Args:
        judgments: List of "same", "different", "unclear" judgments
    
    Returns:
        EquivalenceStats with counts and total
    """
    num_same = sum(1 for j in judgments if j == "same")
    num_different = sum(1 for j in judgments if j == "different")
    num_unclear = sum(1 for j in judgments if j == "unclear")
    
    return EquivalenceStats(
        num_same=num_same,
        num_different=num_different,
        num_unclear=num_unclear,
        total=len(judgments)
    )


def classify_error(
    is_correct: bool,
    equivalence_stats: EquivalenceStats,
    threshold: float = 0.9,
    unclear_treatment: Literal["exclude", "count_as_different"] = "exclude"
) -> ErrorLabel:
    """
    Classify an error based on correctness and semantic equivalence.
    
    Args:
        is_correct: Whether the greedy answer was correct
        equivalence_stats: Statistics about equivalence across samples
        threshold: Minimum equivalence ratio for self-consistent classification
        unclear_treatment: How to handle "unclear" judgments:
            - "exclude": Don't count unclear in the ratio (default)
            - "count_as_different": Treat unclear as different
    
    Returns:
        ErrorLabel with classification and stats
    """
    if is_correct:
        return ErrorLabel(
            label="correct",
            equivalence_stats=equivalence_stats,
            threshold_used=threshold
        )
    
    # Compute equivalence ratio based on treatment
    if unclear_treatment == "exclude":
        ratio = equivalence_stats.equivalence_ratio
    else:  # count_as_different
        ratio = equivalence_stats.equivalence_ratio_with_unclear
    
    # Classify based on threshold
    if ratio >= threshold:
        label = "self_consistent_error"
    else:
        label = "inconsistent_error"
    
    return ErrorLabel(
        label=label,
        equivalence_stats=equivalence_stats,
        threshold_used=threshold
    )


def classify_at_multiple_thresholds(
    is_correct: bool,
    equivalence_stats: EquivalenceStats,
    thresholds: List[float] = [1.0, 0.9, 0.8, 0.7],
    unclear_treatment: Literal["exclude", "count_as_different"] = "exclude"
) -> Dict[float, str]:
    """
    Classify an error at multiple thresholds for sensitivity analysis.
    
    Args:
        is_correct: Whether the greedy answer was correct
        equivalence_stats: Statistics about equivalence across samples
        thresholds: List of thresholds to evaluate
        unclear_treatment: How to handle "unclear" judgments
    
    Returns:
        Dictionary mapping threshold to label
    """
    results = {}
    
    for threshold in thresholds:
        error_label = classify_error(
            is_correct,
            equivalence_stats,
            threshold,
            unclear_treatment
        )
        results[threshold] = error_label.label
    
    return results


if __name__ == "__main__":
    # Test the labeling module
    print("Testing labeling module...\n")
    
    # Test cases with different distributions
    test_cases = [
        # (judgments, is_correct, expected_label_at_0.9)
        (["same"] * 10, True, "correct"),
        (["same"] * 10, False, "self_consistent_error"),
        (["same"] * 9 + ["different"], False, "self_consistent_error"),  # 90%
        (["same"] * 8 + ["different"] * 2, False, "inconsistent_error"),  # 80%
        (["same"] * 7 + ["unclear"] * 3, False, "self_consistent_error"),  # 100% excluding unclear
        (["different"] * 10, False, "inconsistent_error"),
    ]
    
    print("Classification tests (threshold=0.9, unclear=exclude):")
    for judgments, is_correct, expected in test_cases:
        stats = compute_equivalence_stats(judgments)
        result = classify_error(is_correct, stats, threshold=0.9)
        status = "PASS" if result.label == expected else "FAIL"
        print(f"  [{status}] {stats.num_same}s/{stats.num_different}d/{stats.num_unclear}u, "
              f"correct={is_correct} -> {result.label} (ratio={stats.equivalence_ratio:.2f})")
    
    print("\nMulti-threshold test:")
    stats = compute_equivalence_stats(["same"] * 8 + ["different"] * 2)
    labels = classify_at_multiple_thresholds(False, stats)
    print(f"  Stats: {stats.num_same}s/{stats.num_different}d/{stats.num_unclear}u")
    for threshold, label in sorted(labels.items(), reverse=True):
        print(f"  threshold={threshold}: {label}")
