"""Central dataclasses for type consistency across all modules."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal
from datetime import datetime


@dataclass
class Question:
    """Represents a question from the dataset."""
    id: str
    text: str
    ground_truths: List[str]  # All acceptable answers
    category: Optional[str] = None


@dataclass
class GenerationParams:
    """Parameters used for text generation."""
    do_sample: bool
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    max_new_tokens: int


@dataclass
class GenerationResult:
    """Result of a single generation."""
    text: str
    params: GenerationParams
    logprobs: Optional[List[float]] = None  # Optional, not all APIs provide


@dataclass
class CorrectnessResult:
    """Result of correctness checking."""
    is_correct: bool
    match_type: Optional[str] = None  # "exact", "prediction_contains_gold", "gold_contains_prediction"
    matched_answer: Optional[str] = None


@dataclass
class EquivalenceStats:
    """Statistics about semantic equivalence across samples."""
    num_same: int
    num_different: int
    num_unclear: int
    total: int
    
    @property
    def equivalence_ratio(self) -> float:
        """Ratio of 'same' judgments, excluding unclear (default behavior)."""
        denominator = self.num_same + self.num_different
        return self.num_same / denominator if denominator > 0 else 0.0
    
    @property
    def equivalence_ratio_with_unclear(self) -> float:
        """Ratio treating unclear as different."""
        return self.num_same / self.total if self.total > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_same": self.num_same,
            "num_different": self.num_different,
            "num_unclear": self.num_unclear,
            "total": self.total
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EquivalenceStats":
        """Create from dictionary."""
        return cls(
            num_same=data["num_same"],
            num_different=data["num_different"],
            num_unclear=data["num_unclear"],
            total=data["total"]
        )


EquivalenceJudgment = Literal["same", "different", "unclear"]


@dataclass
class ErrorLabel:
    """Classification of an error."""
    label: Literal["correct", "self_consistent_error", "inconsistent_error"]
    equivalence_stats: EquivalenceStats
    threshold_used: float


@dataclass
class ResultRecord:
    """Complete result record for a question-model pair."""
    question_id: str
    question: str
    ground_truth: List[str]
    model: str
    greedy_answer: str
    greedy_correct: bool
    correctness_match_type: Optional[str]
    stochastic_answers: Optional[List[str]]
    equivalence_results: Optional[List[str]]  # "same", "different", "unclear"
    equivalence_stats: Optional[EquivalenceStats]
    equivalence_ratio: Optional[float]
    # Labels at different thresholds for sensitivity analysis
    error_label_1_0: Optional[str] = None
    error_label_0_9: Optional[str] = None
    error_label_0_8: Optional[str] = None
    error_label_0_7: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "model": self.model,
            "greedy_answer": self.greedy_answer,
            "greedy_correct": self.greedy_correct,
            "correctness_match_type": self.correctness_match_type,
            "stochastic_answers": self.stochastic_answers,
            "equivalence_results": self.equivalence_results,
            "equivalence_stats": self.equivalence_stats.to_dict() if self.equivalence_stats else None,
            "equivalence_ratio": self.equivalence_ratio,
            "error_label_1.0": self.error_label_1_0,
            "error_label_0.9": self.error_label_0_9,
            "error_label_0.8": self.error_label_0_8,
            "error_label_0.7": self.error_label_0_7,
            "timestamp": self.timestamp
        }
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "ResultRecord":
        """Create from dictionary."""
        equiv_stats = None
        if data.get("equivalence_stats"):
            equiv_stats = EquivalenceStats.from_dict(data["equivalence_stats"])
        
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            ground_truth=data["ground_truth"],
            model=data["model"],
            greedy_answer=data["greedy_answer"],
            greedy_correct=data["greedy_correct"],
            correctness_match_type=data.get("correctness_match_type"),
            stochastic_answers=data.get("stochastic_answers"),
            equivalence_results=data.get("equivalence_results"),
            equivalence_stats=equiv_stats,
            equivalence_ratio=data.get("equivalence_ratio"),
            error_label_1_0=data.get("error_label_1.0"),
            error_label_0_9=data.get("error_label_0.9"),
            error_label_0_8=data.get("error_label_0.8"),
            error_label_0_7=data.get("error_label_0.7"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z")
        )
