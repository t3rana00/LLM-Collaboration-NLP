import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("results/evaluation_analysis/agent_behavior.csv")
OUTPUT = Path("results/evaluation_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

conditions = ["single", "independent", "collaboration", "debate"]

rng = np.random.default_rng(42)
n_bootstrap = 10000

results = []

baseline = df["single_correct"].astype(int).to_numpy()

for condition in conditions:
    predictions = df[f"{condition}_correct"].astype(int).to_numpy()

    accuracy = predictions.mean()

    # Paired bootstrap: resample the same examples for both methods.
    differences = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(df), len(df))

        difference = (
            predictions[indices].mean()
            - baseline[indices].mean()
        )

        differences.append(difference)

    lower, upper = np.percentile(differences, [2.5, 97.5])

    results.append({
        "condition": condition,
        "accuracy": accuracy,
        "difference_vs_single": accuracy - baseline.mean(),
        "ci_lower": lower,
        "ci_upper": upper,
    })

result_df = pd.DataFrame(results)

print("\nPAIRED BOOTSTRAP RESULTS")
print(result_df.to_string(index=False))

result_df.to_csv(
    OUTPUT / "statistical_results.csv",
    index=False
)

print("\nSaved to:", OUTPUT / "statistical_results.csv")