import os
import json
import time
import argparse
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LEHMUS_API_KEY"),
    base_url=os.getenv("LEHMUS_BASE_URL"),
)

MODEL = os.getenv("LEHMUS_MODEL_ID")

LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

VALID_LABELS = set(LABELS.values())

SYSTEM_PROMPT = """
You are an expert in Natural Language Inference (NLI).

Classify the relationship between a premise and a hypothesis.

Labels:
- entailment: the premise logically supports the hypothesis.
- contradiction: the premise conflicts with the hypothesis.
- neutral: the premise neither entails nor contradicts the hypothesis.

Use only the information provided.
Do not assume unsupported facts.

Return ONLY a complete, valid JSON object with these fields:

{
  "label": "neutral",
  "evidence": "brief exact quote or empty string",
  "justification": "one short sentence",
  "changed_answer": false,
  "peer_assessment": "brief assessment"
}

Use the correct label: entailment, contradiction, or neutral.
Keep evidence, justification, and peer_assessment under 20 words each.
Do not include Markdown, extra text, or explanations outside JSON.
"""


def call_llm(user_prompt):
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1200,
    )

    latency = time.time() - start
    content = response.choices[0].message.content or ""

    # Handle models that wrap JSON in Markdown code fences.
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON: {content[:500]}")

    label = str(result.get("label", "")).lower().strip()

    if label not in VALID_LABELS:
        raise ValueError(f"Invalid NLI label: {label}")

    result["label"] = label

    usage = response.usage

    return {
        "answer": result,
        "latency": latency,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
    }


def format_task(premise, hypothesis):
    return f"""
Premise:
{premise}

Hypothesis:
{hypothesis}

Classify the relationship.
"""


def agent_answer(premise, hypothesis, agent_name):
    prompt = f"""
You are Agent {agent_name}.
Reason independently. You cannot see other agents' answers.

{format_task(premise, hypothesis)}
"""
    return call_llm(prompt)


def revise_answer(premise, hypothesis, own, peer, mode):
    if mode == "collaboration":
        instruction = """
Review the other agent's answer constructively.
Consider whether its evidence improves your original reasoning.
Revise your answer only if justified.
"""
    else:
        instruction = """
Act as a critical debater.
Challenge unsupported assumptions and weak evidence
in the other agent's reasoning.
Defend your original position when justified.
Revise only if the challenge reveals a genuine mistake.
"""

    prompt = f"""
{format_task(premise, hypothesis)}

Your previous answer:
{json.dumps(own)}

Other agent's answer:
{json.dumps(peer)}

{instruction}

Return your final JSON answer.
Set changed_answer to true if your label changed.
Explain your assessment of the peer in peer_assessment.
"""

    return call_llm(prompt)


def decision_agent(premise, hypothesis, answer_a, answer_b, condition):
    prompt = f"""
You are the final decision agent in the {condition} condition.

{format_task(premise, hypothesis)}

Agent A:
{json.dumps(answer_a)}

Agent B:
{json.dumps(answer_b)}

Evaluate both arguments independently.
Do not simply follow the majority.
Choose the label best supported by the premise.

Return your final JSON answer.
"""
    return call_llm(prompt)


def run_example(row):
    premise = str(row["premise"])
    hypothesis = str(row["hypothesis"])

    record = {
        "uid": str(row["uid"]),
        "gold_label": LABELS[int(row["label"])],
        "premise": premise,
        "hypothesis": hypothesis,
        "conditions": {},
    }

    all_calls = []

    # Condition 1: Single agent
    single = agent_answer(premise, hypothesis, "Single")
    all_calls.append(single)

    record["conditions"]["single"] = single["answer"]

    # Shared initial answers
    a = agent_answer(premise, hypothesis, "A")
    b = agent_answer(premise, hypothesis, "B")
    all_calls.extend([a, b])

    # Condition 2: Independent agents
    independent = decision_agent(
        premise,
        hypothesis,
        a["answer"],
        b["answer"],
        "independent",
    )
    all_calls.append(independent)

    record["conditions"]["independent"] = {
        "agent_a": a["answer"],
        "agent_b": b["answer"],
        "final": independent["answer"],
    }

    # Condition 3: Collaboration
    collab_a = revise_answer(
        premise, hypothesis, a["answer"], b["answer"], "collaboration"
    )
    collab_b = revise_answer(
        premise, hypothesis, b["answer"], a["answer"], "collaboration"
    )
    all_calls.extend([collab_a, collab_b])

    collab_final = decision_agent(
        premise,
        hypothesis,
        collab_a["answer"],
        collab_b["answer"],
        "collaboration",
    )
    all_calls.append(collab_final)

    record["conditions"]["collaboration"] = {
        "agent_a_initial": a["answer"],
        "agent_b_initial": b["answer"],
        "agent_a_revised": collab_a["answer"],
        "agent_b_revised": collab_b["answer"],
        "final": collab_final["answer"],
    }

    # Condition 4: Debate
    debate_a = revise_answer(
        premise, hypothesis, a["answer"], b["answer"], "debate"
    )
    debate_b = revise_answer(
        premise, hypothesis, b["answer"], a["answer"], "debate"
    )
    all_calls.extend([debate_a, debate_b])

    debate_final = decision_agent(
        premise,
        hypothesis,
        debate_a["answer"],
        debate_b["answer"],
        "debate",
    )
    all_calls.append(debate_final)

    record["conditions"]["debate"] = {
        "agent_a_initial": a["answer"],
        "agent_b_initial": b["answer"],
        "agent_a_revised": debate_a["answer"],
        "agent_b_revised": debate_b["answer"],
        "final": debate_final["answer"],
    }

    record["api_calls"] = len(all_calls)
    record["latency_seconds"] = sum(c["latency"] for c in all_calls)
    record["prompt_tokens"] = sum(
        c["prompt_tokens"] or 0 for c in all_calls
    )
    record["completion_tokens"] = sum(
        c["completion_tokens"] or 0 for c in all_calls
    )

    return record


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data/development.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output",
        default="results/development_results.jsonl",
    )

    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)

    if args.limit is not None:
        df = df.head(args.limit)

    # Resume previously completed examples.
    completed = set()

    if Path(args.output).exists():
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    completed.add(json.loads(line)["uid"])

    print(f"Dataset examples: {len(df)}", flush=True)
    print(f"Already completed: {len(completed)}", flush=True)

    for index, (_, row) in enumerate(df.iterrows(), start=1):
        if str(row["uid"]) in completed:
            continue

        print(
            f"\nProcessing example {index}/{len(df)}: {row['uid']}",
            flush=True,
        )

        try:
            result = run_example(row)

            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(
                "Gold:",
                result["gold_label"],
                "| Single:",
                result["conditions"]["single"]["label"],
                "| Independent:",
                result["conditions"]["independent"]["final"]["label"],
                "| Collaboration:",
                result["conditions"]["collaboration"]["final"]["label"],
                "| Debate:",
                result["conditions"]["debate"]["final"]["label"],
                flush=True,
            )

            print(
                "API calls:",
                result["api_calls"],
                "| Time:",
                round(result["latency_seconds"], 1),
                "seconds",
                flush=True,
            )

        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}", flush=True)
            print("Stopping to avoid skipping an example.", flush=True)
            break

    print("\nExperiment run finished.", flush=True)


if __name__ == "__main__":
    main()