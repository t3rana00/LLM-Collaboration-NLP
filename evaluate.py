import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

CONDITIONS = [
    "single",
    "independent",
    "collaboration",
    "debate",
]

LABELS = [
    "entailment",
    "neutral",
    "contradiction",
]


def evaluate(input_file, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not records:
        print("No experiment results found.")
        return

    gold = [r["gold_label"] for r in records]

    summary = []

    for condition in CONDITIONS:
        predictions = []

        for record in records:
            result = record["conditions"][condition]

            if condition == "single":
                prediction = result["label"]
            else:
                prediction = result["final"]["label"]

            predictions.append(prediction)

        accuracy = accuracy_score(gold, predictions)

        macro_f1 = f1_score(
            gold,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        summary.append({
            "condition": condition,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        print(f"\n{'=' * 50}")
        print(condition.upper())
        print(f"{'=' * 50}")
        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))

        print(
            classification_report(
                gold,
                predictions,
                labels=LABELS,
                zero_division=0,
            )
        )

        cm = confusion_matrix(
            gold,
            predictions,
            labels=LABELS,
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS,
        )

        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix: {condition.title()}")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            f"{output_dir}/confusion_{condition}.png",
            dpi=200,
        )
        plt.close()

    df = pd.DataFrame(summary)

    df.to_csv(
        f"{output_dir}/metrics.csv",
        index=False,
    )

    print("\nOVERALL RESULTS")
    print(df.to_string(index=False))

    ax = df.plot(
        x="condition",
        y=["accuracy", "macro_f1"],
        kind="bar",
        figsize=(9, 5),
        rot=0,
    )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("NLI Performance Across Agent Conditions")

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/performance_comparison.png",
        dpi=200,
    )

    plt.close()

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/development_results.jsonl",
    )

    parser.add_argument(
        "--output",
        default="results/development_analysis",
    )

    args = parser.parse_args()

    evaluate(args.input, args.output)