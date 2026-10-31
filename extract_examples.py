import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

INPUT = "results/evaluation_results.jsonl"

selected_ids = {
    "70e3a244-138b-4cd9-ab53-706ecc052f5d":
        "Successful correction",

    "32914b6d-0b6c-4179-87ee-f7ff79acb53a":
        "Debate introduced an error",

    "1eba8b1a-7084-4f51-bb7a-dbc4fb30ebc3":
        "Debate changed entailment to neutral",

    "ca3b55f0-b6a7-489b-b52d-3f3f759250f8":
        "Incorrect consensus",
}

with open(INPUT, "r", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

for record in records:
    uid = record["uid"]

    if uid not in selected_ids:
        continue

    print("\n" + "=" * 75)
    print("CASE:", selected_ids[uid])
    print("UID:", uid)
    print("=" * 75)

    print("\nPREMISE:")
    print(record.get("premise", "Not found"))

    print("\nHYPOTHESIS:")
    print(record.get("hypothesis", "Not found"))

    print("\nGOLD LABEL:")
    print(record["gold_label"])

    for condition in [
        "single",
        "independent",
        "collaboration",
        "debate",
    ]:
        print(f"\n{condition.upper()}")

        result = record["conditions"][condition]

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )