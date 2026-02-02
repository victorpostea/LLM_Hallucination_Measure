#!/usr/bin/env python3
"""
Main pipeline script for LLM Self-Consistent Error Measurement.

This script orchestrates the full pipeline:
1. Load questions from TriviaQA
2. For each question and model:
   - Generate greedy answer
   - Check correctness
   - If incorrect, generate stochastic samples
   - Judge semantic equivalence
   - Label the error
   - Save results
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_trivia_qa
from src.inference import HFInferenceClient, build_qa_prompt
from src.correctness import check_correctness, extract_answer_from_response
from src.semantic import SemanticJudge
from src.labeling import compute_equivalence_stats, classify_at_multiple_thresholds
from src.storage import ResultStorage
from src.schemas import ResultRecord, EquivalenceStats

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str, dry_run: bool = False):
    """
    Run the full measurement pipeline.
    
    Args:
        config_path: Path to config.yaml
        dry_run: If True, only load data and validate without making API calls
    """
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Initialize components
    models_to_test = config["models_to_test"]
    judge_model = config["judge_model"]
    
    # Inference settings
    inference_config = config["inference"]
    greedy_config = inference_config["greedy"]
    stochastic_config = inference_config["stochastic"]
    
    # Correctness settings
    correctness_config = config["correctness"]
    
    # Semantic settings
    semantic_config = config["semantic"]
    
    # Dataset settings
    dataset_config = config["dataset"]
    
    # Output settings
    output_config = config["output"]
    
    # Rate limit settings
    rate_config = config.get("rate_limit", {})
    
    # Initialize storage
    storage = ResultStorage(
        results_dir=output_config["results_dir"],
        results_file=output_config["results_file"]
    )
    
    # Get already completed pairs for resume functionality
    completed_pairs = storage.get_completed_pairs()
    if completed_pairs:
        logger.info(f"Found {len(completed_pairs)} completed question-model pairs. Will resume.")
    
    # Initialize inference client
    inference_client = HFInferenceClient(
        initial_delay=rate_config.get("initial_delay", 2.0),
        max_delay=rate_config.get("max_delay", 60.0),
        backoff_factor=rate_config.get("backoff_factor", 2.0)
    )
    
    # Initialize semantic judge
    semantic_judge = SemanticJudge(inference_client, judge_model)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_config['name']}")
    questions = list(load_trivia_qa(
        subset=dataset_config.get("subset", "rc"),
        split=dataset_config.get("split", "validation"),
        max_questions=dataset_config.get("max_questions", 50)
    ))
    logger.info(f"Loaded {len(questions)} questions")
    
    if dry_run:
        logger.info("Dry run mode - validating setup without API calls")
        logger.info(f"Models to test: {models_to_test}")
        logger.info(f"Judge model: {judge_model}")
        logger.info(f"Sample question: {questions[0].text[:100]}...")
        logger.info(f"Sample answers: {questions[0].ground_truths[:3]}")
        return
    
    # Statistics tracking
    stats = {
        "total_processed": 0,
        "skipped_completed": 0,
        "correct": 0,
        "incorrect": 0,
        "self_consistent_errors": 0,
        "inconsistent_errors": 0,
        "errors": 0
    }
    
    # Main processing loop
    total_iterations = len(questions) * len(models_to_test)
    pbar = tqdm(total=total_iterations, desc="Processing")
    
    for question in questions:
        for model in models_to_test:
            pbar.set_description(f"Q:{question.id[:20]}... M:{model.split('/')[-1]}")
            
            # Check if already completed
            if (question.id, model) in completed_pairs:
                logger.debug(f"Skipping completed pair: {question.id}, {model}")
                stats["skipped_completed"] += 1
                pbar.update(1)
                continue
            
            try:
                # Build prompt
                prompt = build_qa_prompt(question.text, model)
                
                # Generate greedy answer
                logger.debug(f"Generating greedy answer for {question.id}")
                greedy_result = inference_client.generate_greedy(
                    model,
                    prompt,
                    max_new_tokens=greedy_config.get("max_new_tokens", 50)
                )
                greedy_answer = extract_answer_from_response(greedy_result.text)
                
                # Check correctness
                correctness_result = check_correctness(
                    greedy_answer,
                    question.ground_truths,
                    strip_articles=correctness_config.get("strip_articles", True),
                    max_length_ratio=correctness_config.get("max_length_ratio", 3.0)
                )
                
                if correctness_result.is_correct:
                    # Save correct result and continue
                    record = ResultRecord(
                        question_id=question.id,
                        question=question.text,
                        ground_truth=question.ground_truths,
                        model=model,
                        greedy_answer=greedy_answer,
                        greedy_correct=True,
                        correctness_match_type=correctness_result.match_type,
                        stochastic_answers=None,
                        equivalence_results=None,
                        equivalence_stats=None,
                        equivalence_ratio=None,
                        error_label_1_0="correct",
                        error_label_0_9="correct",
                        error_label_0_8="correct",
                        error_label_0_7="correct",
                    )
                    storage.save_record(record)
                    stats["correct"] += 1
                    stats["total_processed"] += 1
                    pbar.update(1)
                    continue
                
                # Incorrect answer - generate stochastic samples
                stats["incorrect"] += 1
                logger.debug(f"Incorrect answer, generating {stochastic_config['num_samples']} samples")
                
                stochastic_results = inference_client.generate_stochastic(
                    model,
                    prompt,
                    num_samples=stochastic_config.get("num_samples", 10),
                    temperature=stochastic_config.get("temperature", 0.7),
                    top_p=stochastic_config.get("top_p", 0.9),
                    max_new_tokens=stochastic_config.get("max_new_tokens", 50)
                )
                stochastic_answers = [
                    extract_answer_from_response(r.text) for r in stochastic_results
                ]
                
                # Judge semantic equivalence
                logger.debug(f"Judging semantic equivalence for {len(stochastic_answers)} samples")
                equivalence_results = semantic_judge.judge_all_samples(
                    question.text,
                    greedy_answer,
                    stochastic_answers
                )
                
                # Compute statistics
                equiv_stats = compute_equivalence_stats(equivalence_results)
                
                # Classify at multiple thresholds
                unclear_treatment = semantic_config.get("unclear_treatment", "exclude")
                labels = classify_at_multiple_thresholds(
                    is_correct=False,
                    equivalence_stats=equiv_stats,
                    thresholds=[1.0, 0.9, 0.8, 0.7],
                    unclear_treatment=unclear_treatment
                )
                
                # Track self-consistent vs inconsistent at 0.9 threshold
                if labels[0.9] == "self_consistent_error":
                    stats["self_consistent_errors"] += 1
                else:
                    stats["inconsistent_errors"] += 1
                
                # Create and save record
                record = ResultRecord(
                    question_id=question.id,
                    question=question.text,
                    ground_truth=question.ground_truths,
                    model=model,
                    greedy_answer=greedy_answer,
                    greedy_correct=False,
                    correctness_match_type=None,
                    stochastic_answers=stochastic_answers,
                    equivalence_results=equivalence_results,
                    equivalence_stats=equiv_stats,
                    equivalence_ratio=equiv_stats.equivalence_ratio,
                    error_label_1_0=labels[1.0],
                    error_label_0_9=labels[0.9],
                    error_label_0_8=labels[0.8],
                    error_label_0_7=labels[0.7],
                )
                storage.save_record(record)
                stats["total_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {question.id} with {model}: {e}")
                stats["errors"] += 1
            
            pbar.update(1)
    
    pbar.close()
    
    # Print final statistics
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total processed: {stats['total_processed']}")
    logger.info(f"Skipped (completed): {stats['skipped_completed']}")
    logger.info(f"Correct answers: {stats['correct']}")
    logger.info(f"Incorrect answers: {stats['incorrect']}")
    logger.info(f"  - Self-consistent errors: {stats['self_consistent_errors']}")
    logger.info(f"  - Inconsistent errors: {stats['inconsistent_errors']}")
    logger.info(f"API errors: {stats['errors']}")
    
    # Export to Parquet
    parquet_path = storage.export_to_parquet(output_config.get("parquet_file", "results.parquet"))
    logger.info(f"Exported results to {parquet_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Self-Consistent Error Measurement Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without making API calls"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find config file
    config_path = args.config
    if not os.path.exists(config_path):
        # Try looking in parent directory
        parent_config = Path(__file__).parent.parent / config_path
        if parent_config.exists():
            config_path = str(parent_config)
        else:
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
    
    try:
        run_pipeline(config_path, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
