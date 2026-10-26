from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Create data folder
Path("data").mkdir(exist_ok=True)

print("Downloading ANLI dataset...")

dataset = load_dataset("facebook/anli")

# Check label definitions
print("Dataset labels:")
print(dataset["dev_r1"].features["label"])

def prepare_data(split, n, filename):
    df = dataset[split].to_pandas()

    # Keep required columns
    df = df[["uid", "premise", "hypothesis", "label"]]

    # Sample equal numbers from each class
    per_class = n // 3

    sampled = (
        df.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=42)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    sampled.to_csv(
        f"data/{filename}",
        index=False
    )

    print(f"\nCreated {filename}")
    print("Total examples:", len(sampled))
    print(sampled["label"].value_counts())

prepare_data("dev_r1", 60, "development.csv")
prepare_data("test_r1", 300, "evaluation.csv")

print("\nDataset preparation completed!")