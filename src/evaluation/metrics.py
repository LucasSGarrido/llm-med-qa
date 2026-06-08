"""Evaluation metrics for medical QA: ROUGE-L and exact match."""

from __future__ import annotations

from rouge_score import rouge_scorer


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """Compute mean ROUGE-L F1 score across prediction-reference pairs.

    Args:
        predictions: Model-generated answers.
        references: Ground-truth answers.

    Returns:
        Mean ROUGE-L F1 in [0, 1]. Returns 0.0 for empty input.
    """
    if not predictions:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Compute proportion of exact (case-insensitive, stripped) matches.

    Args:
        predictions: Model-generated answers.
        references: Ground-truth answers.

    Returns:
        Fraction of exact matches in [0, 1]. Returns 0.0 for empty input.
    """
    if not predictions:
        return 0.0
    matches = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return matches / len(predictions)
