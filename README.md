# LLM Collaboration and Debate for Natural Language Inference

## Project Overview

This project investigates whether using multiple Large Language Model (LLM) agents improves Natural Language Inference (NLI) classification.

Four approaches are compared:

1. Single-agent reasoning
2. Independent multi-agent reasoning
3. Constructive collaboration
4. Multi-agent debate

The experiments use the ANLI Round 1 dataset and two language models: GPT OSS 120B and Gemma 4 31B.

## Dataset

Two balanced subsets of ANLI Round 1 were used:

- Development: 60 examples (20 per class)
- Evaluation: 300 examples (100 per class)

The three classes are entailment, neutral, and contradiction.

## Models

Two configurations were tested:

- **Same-model:** GPT OSS 120B for all agents and the judge.
- **Mixed-model:** GPT OSS 120B as Agent A, Gemma 4 31B as Agent B, and GPT OSS 120B as the final judge. Gemma was used for the single-agent baseline.

The models were accessed through the University of Oulu's Lehmus AI platform.

## Final Evaluation Results

| Method | GPT-only accuracy | Mixed-model accuracy |
|---|---:|---:|
| Single-agent baseline | 77.67% | 83.00% |
| Independent reasoning | 78.67% | 81.33% |
| Collaboration | 78.67% | 79.33% |
| Debate | 77.67% | 77.67% |

The best overall result was achieved by Gemma alone, with 83.00% accuracy and a macro-F1 score of 0.8304.

## Implementation

The experiments were implemented in Python using an OpenAI-compatible API.

Main scripts:

- `experiment.py` — original GPT-only experiment.
- `experiment_multimodel.py` — GPT + Gemma experiment.
- `evaluate.py` — evaluation metrics and confusion matrices.
- `analyze_agents.py` — analysis of agent predictions.
- `statistical_analysis.py` — statistical comparison.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Configure the Lehmus AI credentials locally in a `.env` file. The `.env` file must not be committed to GitHub.

## Running the Experiments

Run the mixed-model evaluation:

```bash
python experiment_multimodel.py --dataset data/evaluation.csv --output results/mixed_gpt_gemma_evaluation.jsonl
```

Evaluate the predictions:

```bash
python evaluate.py --input results/mixed_gpt_gemma_evaluation.jsonl --output results/mixed_gpt_gemma_evaluation_metrics
```

The evaluation script produces accuracy, macro-F1 scores, classification reports, confusion matrices, and a performance comparison chart.

## Main Findings

The experiments show that using multiple LLM agents does not always improve NLI classification. Although using different models improved some multi-agent results compared with the GPT-only configuration, none of the multi-agent methods outperformed Gemma alone.

## Security

API credentials are stored in a local `.env` file and are excluded from version control.
