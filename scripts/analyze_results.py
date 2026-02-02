#!/usr/bin/env python3
"""
Analysis script for LLM Self-Consistent Error Measurement results.

Generates:
- Table 1: % self-consistent vs inconsistent errors per model
- Table 2: Error rates by category (if available)
- Table 3: Threshold sensitivity (% self-consistent at different thresholds)
- Table 4: Unclear rate per model (judge reliability check)
- Plot 1: Bar chart comparing models
- Plot 2: Heatmap of error overlap across models
- Plot 3: Distribution of equivalence ratios (histogram)
- Examples: Top most frequent self-consistent errors
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import ResultStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data(results_dir: str, results_file: str = "results.jsonl") -> pd.DataFrame:
    """Load results into a DataFrame."""
    storage = ResultStorage(results_dir, results_file)
    df = storage.to_dataframe()
    
    if df.empty:
        logger.warning("No data found in results file")
        return df
    
    logger.info(f"Loaded {len(df)} records from {results_dir}/{results_file}")
    return df


def table_1_error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: % self-consistent vs inconsistent errors per model.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to incorrect answers only
    incorrect_df = df[~df["greedy_correct"]]
    
    if incorrect_df.empty:
        logger.warning("No incorrect answers found")
        return pd.DataFrame()
    
    results = []
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        model_incorrect = incorrect_df[incorrect_df["model"] == model]
        
        total = len(model_df)
        correct = model_df["greedy_correct"].sum()
        incorrect = len(model_incorrect)
        
        # Count error types at 0.9 threshold
        self_consistent = (model_incorrect["error_label_0.9"] == "self_consistent_error").sum()
        inconsistent = (model_incorrect["error_label_0.9"] == "inconsistent_error").sum()
        
        results.append({
            "Model": model.split("/")[-1],  # Short name
            "Total": total,
            "Correct": correct,
            "Incorrect": incorrect,
            "Accuracy": f"{100*correct/total:.1f}%",
            "Self-Consistent Errors": self_consistent,
            "Inconsistent Errors": inconsistent,
            "% Self-Consistent (of errors)": f"{100*self_consistent/incorrect:.1f}%" if incorrect > 0 else "N/A"
        })
    
    return pd.DataFrame(results)


def table_3_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Threshold sensitivity analysis.
    Shows % of errors classified as self-consistent at different thresholds.
    """
    if df.empty:
        return pd.DataFrame()
    
    incorrect_df = df[~df["greedy_correct"]]
    
    if incorrect_df.empty:
        return pd.DataFrame()
    
    thresholds = ["1.0", "0.9", "0.8", "0.7"]
    results = []
    
    for model in df["model"].unique():
        model_incorrect = incorrect_df[incorrect_df["model"] == model]
        total_errors = len(model_incorrect)
        
        row = {"Model": model.split("/")[-1]}
        
        for threshold in thresholds:
            col = f"error_label_{threshold}"
            if col in model_incorrect.columns:
                self_consistent = (model_incorrect[col] == "self_consistent_error").sum()
                row[f"Threshold {threshold}"] = f"{100*self_consistent/total_errors:.1f}%" if total_errors > 0 else "N/A"
            else:
                row[f"Threshold {threshold}"] = "N/A"
        
        results.append(row)
    
    return pd.DataFrame(results)


def table_4_unclear_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 4: Unclear rate per model (judge reliability check).
    """
    if df.empty:
        return pd.DataFrame()
    
    # Only look at incorrect answers (which have equivalence stats)
    incorrect_df = df[~df["greedy_correct"]]
    
    if incorrect_df.empty or "equiv_num_unclear" not in incorrect_df.columns:
        return pd.DataFrame()
    
    results = []
    
    for model in df["model"].unique():
        model_incorrect = incorrect_df[incorrect_df["model"] == model]
        
        if model_incorrect.empty:
            continue
        
        total_judgments = model_incorrect["equiv_total"].sum()
        unclear_judgments = model_incorrect["equiv_num_unclear"].sum()
        same_judgments = model_incorrect["equiv_num_same"].sum()
        different_judgments = model_incorrect["equiv_num_different"].sum()
        
        results.append({
            "Model": model.split("/")[-1],
            "Total Judgments": int(total_judgments),
            "Same": int(same_judgments),
            "Different": int(different_judgments),
            "Unclear": int(unclear_judgments),
            "Unclear Rate": f"{100*unclear_judgments/total_judgments:.1f}%" if total_judgments > 0 else "N/A"
        })
    
    return pd.DataFrame(results)


def plot_1_model_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot 1: Bar chart comparing models - error type breakdown.
    """
    if df.empty:
        return
    
    incorrect_df = df[~df["greedy_correct"]]
    
    if incorrect_df.empty:
        logger.warning("No incorrect answers to plot")
        return
    
    # Prepare data
    plot_data = []
    for model in df["model"].unique():
        model_incorrect = incorrect_df[incorrect_df["model"] == model]
        short_name = model.split("/")[-1]
        
        self_consistent = (model_incorrect["error_label_0.9"] == "self_consistent_error").sum()
        inconsistent = (model_incorrect["error_label_0.9"] == "inconsistent_error").sum()
        
        plot_data.append({"Model": short_name, "Type": "Self-Consistent", "Count": self_consistent})
        plot_data.append({"Model": short_name, "Type": "Inconsistent", "Count": inconsistent})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(data=plot_df, x="Model", y="Count", hue="Type", ax=ax)
    
    ax.set_title("Error Type Breakdown by Model (threshold=0.9)", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Number of Errors", fontsize=12)
    ax.legend(title="Error Type")
    
    plt.tight_layout()
    output_path = Path(output_dir) / "plot_1_model_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved plot to {output_path}")


def plot_3_equivalence_distribution(df: pd.DataFrame, output_dir: str):
    """
    Plot 3: Distribution of equivalence ratios (histogram).
    """
    if df.empty:
        return
    
    incorrect_df = df[~df["greedy_correct"]]
    
    if incorrect_df.empty or "equivalence_ratio" not in incorrect_df.columns:
        return
    
    # Filter out None values
    ratios = incorrect_df["equivalence_ratio"].dropna()
    
    if ratios.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(ratios, bins=20, edgecolor="black", alpha=0.7)
    
    # Add vertical lines for thresholds
    for threshold, color, label in [(0.9, "red", "0.9"), (0.8, "orange", "0.8"), (0.7, "green", "0.7")]:
        ax.axvline(x=threshold, color=color, linestyle="--", linewidth=2, label=f"Threshold {label}")
    
    ax.set_title("Distribution of Equivalence Ratios (Incorrect Answers)", fontsize=14)
    ax.set_xlabel("Equivalence Ratio (fraction of samples equivalent to greedy)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "plot_3_equivalence_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved plot to {output_path}")


def find_common_self_consistent_errors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Find the most common self-consistent errors (same wrong answer repeated).
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to self-consistent errors
    sc_errors = df[df["error_label_0.9"] == "self_consistent_error"]
    
    if sc_errors.empty:
        return pd.DataFrame()
    
    # Count occurrences of each (question, wrong_answer) pair
    error_counts = Counter()
    examples = {}
    
    for _, row in sc_errors.iterrows():
        key = (row["question"][:100], row["greedy_answer"])
        error_counts[key] += 1
        if key not in examples:
            examples[key] = {
                "Question": row["question"][:150] + "...",
                "Wrong Answer": row["greedy_answer"],
                "Correct Answer(s)": str(row["ground_truth"][:3]),
                "Model(s)": row["model"].split("/")[-1],
                "Equiv Ratio": row.get("equivalence_ratio", "N/A")
            }
        else:
            # Append model if different
            existing_models = examples[key]["Model(s)"]
            new_model = row["model"].split("/")[-1]
            if new_model not in existing_models:
                examples[key]["Model(s)"] += f", {new_model}"
    
    # Get top N
    top_errors = error_counts.most_common(n)
    
    results = []
    for (question, answer), count in top_errors:
        key = (question, answer)
        example = examples[key]
        example["Occurrence"] = count
        results.append(example)
    
    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame, output_dir: str):
    """Generate a full analysis report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("LLM SELF-CONSISTENT ERROR MEASUREMENT - ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Basic stats
    report_lines.append("DATASET SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total records: {len(df)}")
    report_lines.append(f"Unique questions: {df['question_id'].nunique()}")
    report_lines.append(f"Models tested: {df['model'].nunique()}")
    report_lines.append(f"Models: {', '.join(m.split('/')[-1] for m in df['model'].unique())}")
    report_lines.append("")
    
    # Table 1
    report_lines.append("TABLE 1: ERROR BREAKDOWN BY MODEL")
    report_lines.append("-" * 40)
    table1 = table_1_error_breakdown(df)
    if not table1.empty:
        report_lines.append(table1.to_string(index=False))
    else:
        report_lines.append("No data available")
    report_lines.append("")
    
    # Table 3
    report_lines.append("TABLE 3: THRESHOLD SENSITIVITY")
    report_lines.append("-" * 40)
    report_lines.append("(% of errors classified as self-consistent at each threshold)")
    table3 = table_3_threshold_sensitivity(df)
    if not table3.empty:
        report_lines.append(table3.to_string(index=False))
    else:
        report_lines.append("No data available")
    report_lines.append("")
    
    # Table 4
    report_lines.append("TABLE 4: SEMANTIC JUDGE RELIABILITY (UNCLEAR RATE)")
    report_lines.append("-" * 40)
    table4 = table_4_unclear_rate(df)
    if not table4.empty:
        report_lines.append(table4.to_string(index=False))
    else:
        report_lines.append("No data available")
    report_lines.append("")
    
    # Examples
    report_lines.append("TOP SELF-CONSISTENT ERRORS")
    report_lines.append("-" * 40)
    examples = find_common_self_consistent_errors(df, n=10)
    if not examples.empty:
        for i, row in examples.iterrows():
            report_lines.append(f"\n{i+1}. Question: {row['Question']}")
            report_lines.append(f"   Wrong Answer: {row['Wrong Answer']}")
            report_lines.append(f"   Correct: {row['Correct Answer(s)']}")
            report_lines.append(f"   Model(s): {row['Model(s)']}")
            report_lines.append(f"   Equivalence Ratio: {row['Equiv Ratio']}")
    else:
        report_lines.append("No self-consistent errors found")
    
    # Save report
    report_text = "\n".join(report_lines)
    report_file = output_path / "analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info(f"Saved report to {report_file}")
    
    # Print to console
    print("\n" + report_text)
    
    # Save tables as CSV
    if not table1.empty:
        table1.to_csv(output_path / "table_1_error_breakdown.csv", index=False)
    if not table3.empty:
        table3.to_csv(output_path / "table_3_threshold_sensitivity.csv", index=False)
    if not table4.empty:
        table4.to_csv(output_path / "table_4_unclear_rate.csv", index=False)
    if not examples.empty:
        examples.to_csv(output_path / "top_self_consistent_errors.csv", index=False)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM Self-Consistent Error Measurement results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory containing results (default: data/results)"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results.jsonl",
        help="Results file name (default: results.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for analysis output (default: same as results-dir)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Handle paths relative to script location
    script_dir = Path(__file__).parent.parent
    
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = str(script_dir / results_dir)
    
    output_dir = args.output_dir or results_dir
    if not os.path.isabs(output_dir):
        output_dir = str(script_dir / output_dir)
    
    # Load data
    df = load_data(results_dir, args.results_file)
    
    if df.empty:
        logger.error("No data to analyze. Run the pipeline first.")
        sys.exit(1)
    
    # Generate report
    generate_report(df, output_dir)
    
    # Generate plots
    if not args.no_plots:
        logger.info("Generating plots...")
        plot_1_model_comparison(df, output_dir)
        plot_3_equivalence_distribution(df, output_dir)
    
    logger.info(f"Analysis complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
