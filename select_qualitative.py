import json
import pandas as pd
from pathlib import Path

INPUT = Path("results/evaluation_results.jsonl")
OUTPUT = Path("results/evaluation_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUTPUT / "agent_behavior.csv")

categories = {
    "collaboration_correction": (
        (~df["single_correct"]) & df["collaboration_correct"]
    ),
    "debate_correction": (
        (~df["single_correct"]) & df["debate_correct"]
    ),
    "debate_degradation": (
        df["single_correct"] & (~df["debate_correct"])
    ),
    "incorrect_consensus": df["initial_consensus_wrong"],
    "all_correct": (
        df["single_correct"]
        & df["independent_correct"]
        & df["collaboration_correct"]
        & df["debate_correct"]
    ),
}

selected = []
used_ids = set()

for category, mask in categories.items():

    candidates = df[mask].sample(
        frac=1,
        random_state=42
    )

    count = 0

    for _, row in candidates.iterrows():

        if row["uid"] in used_ids:
            continue

        selected.append({
            "uid": row["uid"],
            "category": category,
        })

        used_ids.add(row["uid"])
        count += 1

        if count == 8:
            break

selection = pd.DataFrame(selected)

print("\nSELECTED EXAMPLES")
print(selection["category"].value_counts())
print("\nTotal unique examples:", len(selection))

selection.to_csv(
    OUTPUT / "qualitative_selection.csv",
    index=False
)

selected_ids = set(selection["uid"])

with open(INPUT, "r", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

record_lookup = {
    record["uid"]: record
    for record in records
}

with open(
    OUTPUT / "qualitative_40_examples.txt",
    "w",
    encoding="utf-8"
) as f:

    for _, row in selection.iterrows():

        record = record_lookup[row["uid"]]

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"CATEGORY: {row['category']}\n")
        f.write(f"UID: {row['uid']}\n")
        f.write("=" * 80 + "\n")

        f.write(
            json.dumps(
                record,
                indent=2,
                ensure_ascii=False
            )
        )

        f.write("\n")

print("\nSaved:")
print(OUTPUT / "qualitative_selection.csv")
print(OUTPUT / "qualitative_40_examples.txt")