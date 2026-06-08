import pytest

from src.evaluation.metrics import compute_exact_match, compute_rouge_l


def test_rouge_l_perfect_match():
    """Match perfeito → ROUGE-L = 1.0."""
    score = compute_rouge_l(
        ["Aspirin inhibits COX enzymes."],
        ["Aspirin inhibits COX enzymes."],
    )
    assert score == pytest.approx(1.0, abs=1e-3)


def test_rouge_l_no_overlap():
    """Sem sobreposição → ROUGE-L próximo de 0."""
    score = compute_rouge_l(
        ["The weather is sunny today."],
        ["Aspirin inhibits COX enzymes."],
    )
    assert score < 0.3


def test_rouge_l_partial_match():
    """Sobreposição parcial → 0 < ROUGE-L < 1."""
    score = compute_rouge_l(
        ["Aspirin inhibits COX-1 enzymes."],
        ["Aspirin irreversibly inhibits COX-1 and COX-2 enzymes."],
    )
    assert 0.0 < score < 1.0


def test_rouge_l_empty_input():
    """Lista vazia → 0.0 sem exceção."""
    assert compute_rouge_l([], []) == 0.0


def test_exact_match_case_insensitive():
    """Exact match é case-insensitive e ignora espaços nas bordas."""
    assert compute_exact_match(["aspirin", "Metformin"], ["Aspirin", "METFORMIN"]) == pytest.approx(1.0)
