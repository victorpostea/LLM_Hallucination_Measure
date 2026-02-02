# LLM Self-Consistent Error Measurement

A pipeline for measuring and classifying LLM hallucinations by distinguishing between **self-consistent errors** (confidently wrong) and **inconsistent errors** (uncertain/varying answers).

## Overview

When LLMs make mistakes, they can fail in two fundamentally different ways:

- **Self-Consistent Errors**: The model confidently produces the same wrong answer repeatedly across multiple samples. This suggests a systematic knowledge gap or "confident hallucination."

- **Inconsistent Errors**: The model produces varying answers across samples, indicating uncertainty rather than confident misinformation.

This pipeline measures these error types by:
1. Asking trivia questions from TriviaQA
2. Generating a deterministic (greedy) answer
3. If incorrect, generating multiple stochastic samples
4. Using a semantic judge to determine if samples are equivalent
5. Classifying the error based on the equivalence ratio

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LLM_Hallucination_Measure.git
cd LLM_Hallucination_Measure

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Copy the example environment file and add your HuggingFace token:

```bash
cp .env.example .env
```

Edit `.env` and add your token:
```
HF_TOKEN=your_huggingface_token_here
```

### Pipeline Configuration

Edit `config.yaml` to customize:

- **Models to test**: List of HuggingFace model IDs
- **Judge model**: Model used for semantic equivalence judgments
- **Inference settings**: Greedy and stochastic sampling parameters
- **Correctness settings**: Answer matching configuration
- **Semantic settings**: Equivalence threshold and unclear treatment
- **Dataset settings**: TriviaQA subset, split, and question count

## Usage

### Running the Pipeline

```bash
# Validate setup without making API calls
python scripts/run_pipeline.py --dry-run

# Run the full pipeline
python scripts/run_pipeline.py

# Run with custom config
python scripts/run_pipeline.py --config path/to/config.yaml

# Enable verbose logging
python scripts/run_pipeline.py --verbose
```

The pipeline:
- Supports automatic resumption if interrupted
- Saves results incrementally to `data/results/results.jsonl`
- Exports final results to Parquet format

### Analyzing Results

```bash
# Generate analysis report and plots
python scripts/analyze_results.py

# Specify custom results directory
python scripts/analyze_results.py --results-dir data/results

# Skip plot generation
python scripts/analyze_results.py --no-plots
```

The analysis generates:
- **Table 1**: Error breakdown by model (accuracy, self-consistent vs inconsistent)
- **Table 3**: Threshold sensitivity analysis
- **Table 4**: Semantic judge reliability (unclear rate)
- **Plot 1**: Bar chart comparing error types across models
- **Plot 3**: Distribution of equivalence ratios
- **Examples**: Top self-consistent errors with context

## Project Structure

```
LLM_Hallucination_Measure/
├── config.yaml              # Pipeline configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
├── scripts/
│   ├── run_pipeline.py      # Main pipeline orchestration
│   └── analyze_results.py   # Results analysis and visualization
├── src/
│   ├── __init__.py
│   ├── dataset.py           # TriviaQA data loading
│   ├── inference.py         # HuggingFace inference client
│   ├── correctness.py       # Answer correctness checking
│   ├── semantic.py          # Semantic equivalence judge
│   ├── labeling.py          # Error classification logic
│   ├── storage.py           # Results persistence
│   └── schemas.py           # Data type definitions
└── data/
    └── results/             # Output directory (created at runtime)
```

## How It Works

### 1. Question Sampling
Questions are loaded from the TriviaQA dataset (RC subset by default), which provides factual questions with multiple accepted ground truth answers.

### 2. Greedy Generation
For each question, a deterministic answer is generated using greedy decoding (`do_sample=false`).

### 3. Correctness Check
The greedy answer is compared against ground truths using containment matching with article stripping ("the", "a", "an" removed).

### 4. Stochastic Sampling
If the greedy answer is incorrect, multiple samples are generated with temperature sampling to assess consistency.

### 5. Semantic Judgment
Each stochastic sample is compared to the greedy answer using an LLM judge that outputs:
- **Same**: Semantically equivalent answers
- **Different**: Meaningfully different answers
- **Unclear**: Cannot determine equivalence

### 6. Error Classification
The equivalence ratio (same / total) determines classification:
- **Self-Consistent Error**: ratio ≥ threshold (default 0.9)
- **Inconsistent Error**: ratio < threshold

Multiple thresholds (1.0, 0.9, 0.8, 0.7) are computed for sensitivity analysis.

## Output Format

Results are saved in JSONL format with fields:
- `question_id`, `question`, `ground_truth`
- `model`, `greedy_answer`, `greedy_correct`
- `stochastic_answers`, `equivalence_results`
- `equivalence_ratio`, `error_label_*` (at each threshold)

## Requirements

- Python 3.10+
- HuggingFace account with API access
- Models accessible via HuggingFace Inference API

## License

MIT License
