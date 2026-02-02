"""Storage utilities for saving and loading results."""

import json
import logging
import os
from pathlib import Path
from typing import List, Iterator, Optional, Set, Tuple

import pandas as pd

from .schemas import ResultRecord

logger = logging.getLogger(__name__)


class ResultStorage:
    """Handles saving and loading of pipeline results."""
    
    def __init__(self, results_dir: str, results_file: str = "results.jsonl"):
        """
        Initialize the storage handler.
        
        Args:
            results_dir: Directory to store results
            results_file: Name of the JSON lines file
        """
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / results_file
        
        # Ensure directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_record(self, record: ResultRecord) -> None:
        """
        Append a single result record to the JSON lines file.
        
        Args:
            record: ResultRecord to save
        """
        with open(self.results_file, "a", encoding="utf-8") as f:
            json_str = json.dumps(record.to_dict(), ensure_ascii=False)
            f.write(json_str + "\n")
        
        logger.debug(f"Saved record for question {record.question_id}, model {record.model}")
    
    def load_records(self) -> Iterator[ResultRecord]:
        """
        Load all result records from the JSON lines file.
        
        Yields:
            ResultRecord objects
        """
        if not self.results_file.exists():
            return
        
        with open(self.results_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    yield ResultRecord.from_dict(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error creating record from line {line_num}: {e}")
                    continue
    
    def get_completed_pairs(self) -> Set[Tuple[str, str]]:
        """
        Get set of (question_id, model) pairs that have been completed.
        
        Used for checkpoint/resume functionality.
        
        Returns:
            Set of (question_id, model) tuples
        """
        completed = set()
        
        for record in self.load_records():
            completed.add((record.question_id, record.model))
        
        return completed
    
    def count_records(self) -> int:
        """
        Count total number of records.
        
        Returns:
            Number of records in the file
        """
        count = 0
        for _ in self.load_records():
            count += 1
        return count
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Load all records into a pandas DataFrame.
        
        Returns:
            DataFrame with one row per record
        """
        records = []
        
        for record in self.load_records():
            record_dict = record.to_dict()
            
            # Flatten equivalence_stats
            if record_dict.get("equivalence_stats"):
                stats = record_dict.pop("equivalence_stats")
                record_dict["equiv_num_same"] = stats["num_same"]
                record_dict["equiv_num_different"] = stats["num_different"]
                record_dict["equiv_num_unclear"] = stats["num_unclear"]
                record_dict["equiv_total"] = stats["total"]
            
            records.append(record_dict)
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records)
    
    def export_to_parquet(self, parquet_file: Optional[str] = None) -> str:
        """
        Export results to Parquet format for efficient analysis.
        
        Args:
            parquet_file: Output file name (default: results.parquet)
        
        Returns:
            Path to the created Parquet file
        """
        if parquet_file is None:
            parquet_file = "results.parquet"
        
        output_path = self.results_dir / parquet_file
        
        df = self.to_dataframe()
        
        if df.empty:
            logger.warning("No records to export")
            return str(output_path)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")
        
        return str(output_path)
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics about stored results.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.to_dataframe()
        
        if df.empty:
            return {"total_records": 0}
        
        stats = {
            "total_records": len(df),
            "unique_questions": df["question_id"].nunique(),
            "unique_models": df["model"].nunique(),
            "models": df["model"].unique().tolist(),
        }
        
        # Correctness stats
        stats["correct_count"] = df["greedy_correct"].sum()
        stats["incorrect_count"] = (~df["greedy_correct"]).sum()
        stats["accuracy"] = stats["correct_count"] / len(df) if len(df) > 0 else 0
        
        # Error type stats (for incorrect answers)
        incorrect_df = df[~df["greedy_correct"]]
        if not incorrect_df.empty and "error_label_0.9" in incorrect_df.columns:
            label_counts = incorrect_df["error_label_0.9"].value_counts().to_dict()
            stats["error_labels_0.9"] = label_counts
        
        return stats


def load_results_from_file(filepath: str) -> List[ResultRecord]:
    """
    Load results from a JSON lines file.
    
    Args:
        filepath: Path to the JSON lines file
    
    Returns:
        List of ResultRecord objects
    """
    records = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append(ResultRecord.from_dict(data))
    
    return records


if __name__ == "__main__":
    import tempfile
    from datetime import datetime
    
    from .schemas import EquivalenceStats
    
    print("Testing storage module...\n")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ResultStorage(tmpdir)
        
        # Create test records
        test_record = ResultRecord(
            question_id="test_001",
            question="What is the capital of France?",
            ground_truth=["Paris", "paris"],
            model="test-model",
            greedy_answer="Lyon",
            greedy_correct=False,
            correctness_match_type=None,
            stochastic_answers=["Lyon", "Lyon", "Paris"],
            equivalence_results=["same", "same", "different"],
            equivalence_stats=EquivalenceStats(
                num_same=2,
                num_different=1,
                num_unclear=0,
                total=3
            ),
            equivalence_ratio=0.67,
            error_label_1_0="inconsistent_error",
            error_label_0_9="inconsistent_error",
            error_label_0_8="inconsistent_error",
            error_label_0_7="inconsistent_error",
        )
        
        # Save record
        storage.save_record(test_record)
        print(f"Saved record to {storage.results_file}")
        
        # Load records
        loaded = list(storage.load_records())
        print(f"Loaded {len(loaded)} records")
        
        # Check completed pairs
        completed = storage.get_completed_pairs()
        print(f"Completed pairs: {completed}")
        
        # Get stats
        stats = storage.get_summary_stats()
        print(f"Summary stats: {stats}")
        
        # Test DataFrame export
        df = storage.to_dataframe()
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
