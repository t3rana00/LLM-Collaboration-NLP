import json
from pathlib import Path

import pandas as pd

INPUT = "results/evaluation_results.jsonl"
OUTPUT = Path("results/evaluation_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

with open(INPUT, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

conditions = ["single", "independent", "collaboration", "debate"]


def prediction(record, condition):
    result = record["conditions"][condition]
    return result["label"] if condition == "single" else result["final"]["label"]


rows = []

for record in records:
    gold = record["gold_label"]
    preds = {condition: prediction(record, condition) for condition in conditions}

    independent = record["conditions"]["independent"]
    collaboration = record["conditions"]["collaboration"]
    debate = record["conditions"]["debate"]

    a = independent["agent_a"]["label"]
    b = independent["agent_b"]["label"]

    rows.append({
        "uid": record["uid"],
        "gold": gold,
        **{f"{c}_prediction": preds[c] for c in conditions},
        **{f"{c}_correct": preds[c] == gold for c in conditions},
        "initial_agreement": a == b,
        "initial_consensus_wrong": a == b and a != gold,
        "collab_a_changed": (
            collaboration["agent_a_initial"]["label"]
            != collaboration["agent_a_revised"]["label"]
        ),
        "collab_b_changed": (
            collaboration["agent_b_initial"]["label"]
            != collaboration["agent_b_revised"]["label"]
        ),
        "debate_a_changed": (
            debate["agent_a_initial"]["label"]
            != debate["agent_a_revised"]["label"]
        ),
        "debate_b_changed": (
            debate["agent_b_initial"]["label"]
            != debate["agent_b_revised"]["label"]
        ),
    })

df = pd.DataFrame(rows)

print("\nAGENT AGREEMENT")
print("Examples:", len(df))
print("Initial A/B agreement:", int(df["initial_agreement"].sum()))
print("Initial incorrect consensus:", int(df["initial_consensus_wrong"].sum()))

print("\nAGENT REVISIONS")
for col in [
    "collab_a_changed",
    "collab_b_changed",
    "debate_a_changed",
    "debate_b_changed",
]:
    print(col, int(df[col].sum()))

print("\nCOMPARISON AGAINST SINGLE AGENT")
for condition in ["independent", "collaboration", "debate"]:
    corrected = (
        (~df["single_correct"]) & df[f"{condition}_correct"]
    ).sum()

    degraded = (
        df["single_correct"] & (~df[f"{condition}_correct"])
    ).sum()

    same_prediction = (
        df["single_prediction"] == df[f"{condition}_prediction"]
    ).sum()

    print(f"\n{condition.upper()}")
    print("Corrected single-agent errors:", int(corrected))
    print("Introduced new errors:", int(degraded))
    print("Same final prediction as single:", int(same_prediction))

print("\nCOLLABORATION VS INDEPENDENT")
print(
    "Same final prediction:",
    int(
        (
            df["collaboration_prediction"]
            == df["independent_prediction"]
        ).sum()
    ),
)

df.to_csv(OUTPUT / "agent_behavior.csv", index=False)

# Save interesting examples for qualitative analysis.
interesting = df[
    (~df["single_correct"] & df["collaboration_correct"])
    | (df["single_correct"] & ~df["debate_correct"])
    | df["initial_consensus_wrong"]
]

interesting.to_csv(
    OUTPUT / "interesting_examples.csv",
    index=False,
)

print("\nSaved:")
print(OUTPUT / "agent_behavior.csv")
print(OUTPUT / "interesting_examples.csv")