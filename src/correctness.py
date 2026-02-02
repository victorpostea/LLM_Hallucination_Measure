"""Correctness checking for answer verification with robust TriviaQA matching."""

import re
import string
from typing import List, Tuple, Optional

from .schemas import CorrectnessResult


# Articles to strip
ARTICLES = {"a", "an", "the"}


def normalize(text: str, strip_articles: bool = True) -> str:
    """
    Normalize text for comparison.
    
    Operations:
    - Lowercase
    - Strip articles (the, a, an)
    - Remove punctuation
    - Remove extra whitespace
    - Handle "X (something)" formatting -> extract "X"
    
    Args:
        text: Input text to normalize
        strip_articles: Whether to remove articles
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower().strip()
    
    # Handle "X (something)" formatting - extract the part before parentheses
    # This handles cases like "Paris (city)" -> "Paris"
    paren_match = re.match(r'^([^(]+)\s*\(.*\)$', text)
    if paren_match:
        text = paren_match.group(1).strip()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Split into words for article removal
    words = text.split()
    
    # Strip articles if requested
    if strip_articles:
        words = [w for w in words if w not in ARTICLES]
    
    # Rejoin and normalize whitespace
    text = " ".join(words)
    
    return text


def check_correctness(
    prediction: str,
    ground_truths: List[str],
    strip_articles: bool = True,
    max_length_ratio: float = 3.0
) -> CorrectnessResult:
    """
    Check if a prediction matches any of the ground truth answers.
    
    Uses containment matching with length guards to handle TriviaQA's
    varied answer formats.
    
    Matching rules:
    1. Exact match after normalization
    2. Gold answer contained in prediction (with length guard)
    3. Prediction contained in gold answer (with length guard)
    
    Args:
        prediction: The model's predicted answer
        ground_truths: List of acceptable ground truth answers
        strip_articles: Whether to strip articles during normalization
        max_length_ratio: Maximum length ratio for containment matching
                         (prediction length / gold length must be < this)
    
    Returns:
        CorrectnessResult with is_correct, match_type, and matched_answer
    """
    # Normalize prediction
    norm_pred = normalize(prediction, strip_articles)
    
    if not norm_pred:
        return CorrectnessResult(
            is_correct=False,
            match_type=None,
            matched_answer=None
        )
    
    pred_len = len(norm_pred)
    
    for gold in ground_truths:
        norm_gold = normalize(gold, strip_articles)
        
        if not norm_gold:
            continue
        
        gold_len = len(norm_gold)
        
        # 1. Exact match
        if norm_pred == norm_gold:
            return CorrectnessResult(
                is_correct=True,
                match_type="exact",
                matched_answer=gold
            )
        
        # 2. Gold contained in prediction (with length guard)
        # e.g., prediction="The capital is Paris" matches gold="Paris"
        if norm_gold in norm_pred:
            # Length guard: prediction shouldn't be too much longer than gold
            if pred_len <= gold_len * max_length_ratio:
                return CorrectnessResult(
                    is_correct=True,
                    match_type="prediction_contains_gold",
                    matched_answer=gold
                )
        
        # 3. Prediction contained in gold (with length guard)
        # e.g., prediction="Paris" matches gold="Paris, France"
        if norm_pred in norm_gold:
            # Length guard: gold shouldn't be too much longer than prediction
            if gold_len <= pred_len * max_length_ratio:
                return CorrectnessResult(
                    is_correct=True,
                    match_type="gold_contains_prediction",
                    matched_answer=gold
                )
    
    # No match found
    return CorrectnessResult(
        is_correct=False,
        match_type=None,
        matched_answer=None
    )


def extract_answer_from_response(response: str) -> str:
    """
    Extract the answer from a model response.
    
    Handles common response patterns:
    - Direct answer
    - "The answer is X"
    - "X is the answer"
    - Responses with explanation after the answer
    
    Args:
        response: Raw model response
    
    Returns:
        Extracted answer string
    """
    if not response:
        return ""
    
    # Clean up the response
    text = response.strip()
    
    # Take first line if multi-line (often the answer is first)
    lines = text.split('\n')
    text = lines[0].strip()
    
    # Try to extract if wrapped in common patterns
    # Pattern: "The answer is X" or "Answer: X"
    answer_patterns = [
        r"(?:the\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)",
        r"answer:\s*(.+?)(?:\.|$)",
        r"^(.+?)(?:\s+is the answer)",
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matched, take the text up to the first period
    # (to remove explanations)
    if "." in text:
        text = text.split(".")[0].strip()
    
    return text


if __name__ == "__main__":
    # Test cases
    print("Testing correctness module...\n")
    
    test_cases = [
        # (prediction, ground_truths, expected_correct)
        ("Paris", ["Paris", "paris"], True),
        ("The answer is Paris", ["Paris"], True),
        ("Paris, France", ["Paris"], True),
        ("Paris", ["Paris, the capital of France"], True),
        ("Lyon", ["Paris"], False),
        ("The capital is definitely Paris", ["Paris"], True),
        ("I think the capital might be Paris but I'm not sure about that", ["Paris"], False),  # Too long
        ("Barack Obama", ["Barack Hussein Obama", "Obama"], True),
        ("the Beatles", ["Beatles", "The Beatles"], True),
    ]
    
    print("Normalization tests:")
    print(f"  'The Beatles' -> '{normalize('The Beatles')}'")
    print(f"  'Paris (city)' -> '{normalize('Paris (city)')}'")
    print(f"  'A Tale of Two Cities' -> '{normalize('A Tale of Two Cities')}'")
    
    print("\nCorrectness tests:")
    for pred, truths, expected in test_cases:
        result = check_correctness(pred, truths)
        status = "PASS" if result.is_correct == expected else "FAIL"
        print(f"  [{status}] '{pred}' vs {truths}: {result.is_correct} ({result.match_type})")
