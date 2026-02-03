#!/usr/bin/env python3
"""
Compare self-consistent errors between models.
Shows which questions are self-consistent errors for both/either model.
Saves results to data/results/model_error_overlap.txt and model_error_overlap.csv
"""

import sys
from pathlib import Path

import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main():
    results_dir = Path(__file__).parent.parent / "data" / "results"
    parquet_file = results_dir / "results.parquet"
    output_txt = results_dir / "model_error_overlap.txt"
    output_csv = results_dir / "model_error_overlap.csv"
    
    lines = []  # Collect output for saving
    
    def log(text=""):
        print(text)
        lines.append(text)
    
    log("=" * 70)
    log("SELF-CONSISTENT ERROR OVERLAP BETWEEN MODELS")
    log("=" * 70)
    log()
    
    # Load data
    df = pd.read_parquet(parquet_file)
    log(f"Loaded {len(df)} records")
    
    models = df["model"].unique()
    log(f"Models: {[m.split('/')[-1] for m in models]}")
    log()
    
    # Get self-consistent errors for each model at threshold 0.9
    threshold = "0.9"
    col_name = f"error_label_{threshold}"
    
    model_errors = {}
    for model in models:
        model_df = df[df["model"] == model]
        sc_errors = model_df[model_df[col_name] == "self_consistent_error"]
        model_errors[model] = set(sc_errors["question_id"].unique())
        short_name = model.split("/")[-1]
        log(f"{short_name}: {len(model_errors[model])} self-consistent errors")
    
    log()
    
    # Calculate overlap
    model_list = list(models)
    if len(model_list) >= 2:
        m1, m2 = model_list[0], model_list[1]
        m1_short = m1.split("/")[-1]
        m2_short = m2.split("/")[-1]
        
        overlap = model_errors[m1] & model_errors[m2]
        only_m1 = model_errors[m1] - model_errors[m2]
        only_m2 = model_errors[m2] - model_errors[m1]
        
        log("OVERLAP ANALYSIS")
        log("-" * 40)
        log(f"{'Both models:':<30} {len(overlap):>5}")
        log(f"{'Only ' + m1_short + ':':<30} {len(only_m1):>5}")
        log(f"{'Only ' + m2_short + ':':<30} {len(only_m2):>5}")
        log()
        
        # Show overlapping questions with details
        if overlap:
            log(f"QUESTIONS WITH SELF-CONSISTENT ERRORS IN BOTH MODELS ({len(overlap)}):")
            log("-" * 70)
            for qid in sorted(overlap):
                row = df[df["question_id"] == qid].iloc[0]
                q_text = row["question"][:70] + "..." if len(row["question"]) > 70 else row["question"]
                log(f"\n  [{qid}]")
                log(f"  Q: {q_text}")
                log(f"  Ground Truth: {row['ground_truth']}")
                
                # Show each model's answer
                for model in models:
                    model_row = df[(df["question_id"] == qid) & (df["model"] == model)].iloc[0]
                    short_name = model.split("/")[-1]
                    log(f"  {short_name}: {model_row['greedy_answer']}")
            log()
        
        # Show questions only in one model
        if only_m1:
            log(f"\nQUESTIONS WITH SELF-CONSISTENT ERRORS ONLY IN {m1_short} ({len(only_m1)}):")
            log("-" * 70)
            for qid in sorted(only_m1):
                row = df[(df["question_id"] == qid) & (df["model"] == m1)].iloc[0]
                q_text = row["question"][:70] + "..." if len(row["question"]) > 70 else row["question"]
                log(f"\n  [{qid}]")
                log(f"  Q: {q_text}")
                log(f"  Ground Truth: {row['ground_truth']}")
                log(f"  {m1_short} answer: {row['greedy_answer']}")
                # Show what the other model got
                other_row = df[(df["question_id"] == qid) & (df["model"] == m2)]
                if not other_row.empty:
                    other_row = other_row.iloc[0]
                    log(f"  {m2_short} answer: {other_row['greedy_answer']} (correct: {other_row['greedy_correct']})")
            log()
        
        if only_m2:
            log(f"\nQUESTIONS WITH SELF-CONSISTENT ERRORS ONLY IN {m2_short} ({len(only_m2)}):")
            log("-" * 70)
            for qid in sorted(only_m2):
                row = df[(df["question_id"] == qid) & (df["model"] == m2)].iloc[0]
                q_text = row["question"][:70] + "..." if len(row["question"]) > 70 else row["question"]
                log(f"\n  [{qid}]")
                log(f"  Q: {q_text}")
                log(f"  Ground Truth: {row['ground_truth']}")
                log(f"  {m2_short} answer: {row['greedy_answer']}")
                # Show what the other model got
                other_row = df[(df["question_id"] == qid) & (df["model"] == m1)]
                if not other_row.empty:
                    other_row = other_row.iloc[0]
                    log(f"  {m1_short} answer: {other_row['greedy_answer']} (correct: {other_row['greedy_correct']})")
            log()
        
        # Build CSV data
        csv_rows = []
        for qid in sorted(overlap):
            csv_rows.append({"question_id": qid, "category": "both_models"})
        for qid in sorted(only_m1):
            csv_rows.append({"question_id": qid, "category": f"only_{m1_short}"})
        for qid in sorted(only_m2):
            csv_rows.append({"question_id": qid, "category": f"only_{m2_short}"})
        
        # Save CSV
        csv_df = pd.DataFrame(csv_rows)
        csv_df.to_csv(output_csv, index=False)
        log(f"Saved CSV to {output_csv}")
    
    log("=" * 70)
    log("Done!")
    
    # Save text report
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved report to {output_txt}")


if __name__ == "__main__":
    main()
