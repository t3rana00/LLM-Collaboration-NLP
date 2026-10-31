"""
Mixed-model Natural Language Inference experiment.

Models:
    Agent A: GPT OSS 120B
    Agent B: Gemma 4 31B
    Judge: GPT OSS 120B

Conditions:
    1. Gemma single-agent baseline
    2. Independent agents
    3. Collaboration
    4. Debate
"""

import argparse
import json
import os
import time

from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

MODEL_GPT = os.getenv("LEHMUS_MODEL_ID", "").strip()
MODEL_GEMMA = os.getenv("GEMMA_MODEL_ID", "").strip()

REQUIRED = [
    "LEHMUS_API_KEY",
    "LEHMUS_BASE_URL",
    "LEHMUS_MODEL_ID",
    "GEMMA_API_KEY",
    "GEMMA_BASE_URL",
    "GEMMA_MODEL_ID",
]

missing = [
    name
    for name in REQUIRED
    if not os.getenv(name, "").strip()
]

if missing:
    raise ValueError(
        f"Missing .env settings: {', '.join(missing)}"
    )

if MODEL_GPT == MODEL_GEMMA:
    raise ValueError(
        "GPT and Gemma model IDs must be different."
    )


# ============================================================
# 2. CREATE SEPARATE API CLIENTS
# ============================================================

client_gpt = OpenAI(
    api_key=os.environ["LEHMUS_API_KEY"],
    base_url=os.environ["LEHMUS_BASE_URL"],
    timeout=120.0,
    max_retries=0,
)

client_gemma = OpenAI(
    api_key=os.environ["GEMMA_API_KEY"],
    base_url=os.environ["GEMMA_BASE_URL"],
    timeout=120.0,
    max_retries=0,
)


# ============================================================
# 3. LABELS AND SYSTEM PROMPT
# ============================================================

LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

VALID_LABELS = set(LABELS.values())

SYSTEM_PROMPT = """
You are a careful expert in natural language inference (NLI).

Determine whether the hypothesis is:
- entailment
- neutral
- contradiction

Base your judgment only on the premise and hypothesis.

Return ONLY a valid JSON object.

Required keys:
"label"
"evidence"
"justification"
"changed_answer"
"peer_assessment"

The label must be one of:
"entailment", "neutral", "contradiction"

changed_answer must be a boolean.

Do not include markdown formatting.
"""


# ============================================================
# 4. PARSE MODEL RESPONSE
# ============================================================

def parse_answer(text):

    if not text or not text.strip():
        raise ValueError(
            "Model returned an empty response."
        )

    content = text.strip()

    # Remove markdown code fences if present
    if content.startswith("```"):

        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    try:
        obj = json.loads(content)

    except json.JSONDecodeError:

        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end <= start:
            raise ValueError(
                f"Response is not valid JSON: {content[:300]}"
            )

        obj = json.loads(
            content[start:end + 1]
        )

    if not isinstance(obj, dict):
        raise ValueError(
            "Model response must be a JSON object."
        )

    label = str(
        obj.get("label", "")
    ).strip().lower()

    if label not in VALID_LABELS:
        raise ValueError(
            f"Invalid NLI label: {label}"
        )

    obj["label"] = label

    obj.setdefault("evidence", "")
    obj.setdefault("justification", "")
    obj.setdefault("changed_answer", False)
    obj.setdefault("peer_assessment", "")

    return obj


# ============================================================
# 5. CALL GPT OR GEMMA
# ============================================================

def call_llm(user_prompt, model=None):

    model = model or MODEL_GPT

    if model == MODEL_GPT:
        selected_client = client_gpt

    elif model == MODEL_GEMMA:
        selected_client = client_gemma

    else:
        raise ValueError(
            f"Unknown model ID: {model}"
        )

    for attempt in range(1, 5):

        try:

            start = time.perf_counter()

            response = selected_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0,
                max_tokens=1600,
            )

            latency = time.perf_counter() - start

            answer = parse_answer(
                response.choices[0].message.content
            )

            usage = response.usage

            return {
                "answer": answer,
                "latency": latency,
                "prompt_tokens": getattr(
                    usage,
                    "prompt_tokens",
                    None,
                ),
                "completion_tokens": getattr(
                    usage,
                    "completion_tokens",
                    None,
                ),
            }

        except Exception as error:

            print(
                f"API attempt {attempt}/4 failed "
                f"({model}): "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

            if attempt == 4:
                raise

            time.sleep(3 * attempt)


# ============================================================
# 6. FORMAT NLI TASK
# ============================================================

def format_task(premise, hypothesis):

    return f"""
Premise:
{premise}

Hypothesis:
{hypothesis}

Classify the relationship as:
entailment, neutral, or contradiction.
"""


# ============================================================
# 7. INITIAL AGENT ANSWER
# ============================================================

def agent_answer(
    premise,
    hypothesis,
    agent_name,
    model=None,
):

    prompt = f"""
You are Agent {agent_name}.

Reason independently.

{format_task(premise, hypothesis)}

Give your best-supported label and concise evidence.

Return your answer as the required JSON object.
"""

    return call_llm(
        prompt,
        model=model,
    )


# ============================================================
# 8. COLLABORATION AND DEBATE
# ============================================================

def revise_answer(
    premise,
    hypothesis,
    own,
    peer,
    mode,
    model=None,
):

    if mode == "collaboration":

        instruction = """
Act as a constructive collaborator.

Carefully compare your reasoning with the other agent.

Correct genuine errors.

Do not change a well-supported answer merely to agree.
"""

    elif mode == "debate":

        instruction = """
Act as a critical debater.

Challenge unsupported assumptions and weak evidence
in the other agent's reasoning.

Defend your original position when justified.

Revise only if the challenge reveals a genuine mistake.
"""

    else:
        raise ValueError(
            f"Unknown revision mode: {mode}"
        )

    prompt = f"""
{format_task(premise, hypothesis)}

Your previous answer:
{json.dumps(own, ensure_ascii=False)}

Other agent's answer:
{json.dumps(peer, ensure_ascii=False)}

{instruction}

Return your final JSON answer.

Set changed_answer to true if your label changed.

Explain your assessment of the peer in peer_assessment.
"""

    return call_llm(
        prompt,
        model=model,
    )


# ============================================================
# 9. FINAL DECISION AGENT
# ============================================================

def decision_agent(
    premise,
    hypothesis,
    answer_a,
    answer_b,
    condition,
    model=None,
):

    prompt = f"""
You are the final decision agent in the {condition} condition.

{format_task(premise, hypothesis)}

Agent A:
{json.dumps(answer_a, ensure_ascii=False)}

Agent B:
{json.dumps(answer_b, ensure_ascii=False)}

Evaluate both arguments independently.

Do not simply follow the majority.

Choose the label best supported by the premise.

Return your final JSON answer.
"""

    return call_llm(
        prompt,
        model=model,
    )


# ============================================================
# 10. RUN ONE DATASET EXAMPLE
# ============================================================

def run_example(row):

    premise = str(row["premise"])
    hypothesis = str(row["hypothesis"])

    record = {
        "uid": str(row["uid"]),
        "gold_label": LABELS[int(row["label"])],
        "premise": premise,
        "hypothesis": hypothesis,
        "experiment_type": "mixed_model_gpt_gemma",
        "models": {
            "single": MODEL_GEMMA,
            "agent_a": MODEL_GPT,
            "agent_b": MODEL_GEMMA,
            "judge": MODEL_GPT,
        },
        "conditions": {},
    }

    all_calls = []

    # --------------------------------------------------------
    # CONDITION 1: GEMMA SINGLE AGENT
    # --------------------------------------------------------

    single = agent_answer(
        premise,
        hypothesis,
        "Gemma Single",
        model=MODEL_GEMMA,
    )

    all_calls.append(single)

    record["conditions"]["single"] = single["answer"]

    # --------------------------------------------------------
    # INITIAL AGENTS
    # Agent A = GPT
    # Agent B = Gemma
    # --------------------------------------------------------

    a = agent_answer(
        premise,
        hypothesis,
        "A",
        model=MODEL_GPT,
    )

    b = agent_answer(
        premise,
        hypothesis,
        "B",
        model=MODEL_GEMMA,
    )

    all_calls.extend([a, b])

    # --------------------------------------------------------
    # CONDITION 2: INDEPENDENT AGENTS
    # --------------------------------------------------------

    independent = decision_agent(
        premise,
        hypothesis,
        a["answer"],
        b["answer"],
        "independent",
        model=MODEL_GPT,
    )

    all_calls.append(independent)

    record["conditions"]["independent"] = {
        "agent_a": a["answer"],
        "agent_b": b["answer"],
        "final": independent["answer"],
    }

    # --------------------------------------------------------
    # CONDITION 3: COLLABORATION
    # --------------------------------------------------------

    collab_a = revise_answer(
        premise,
        hypothesis,
        a["answer"],
        b["answer"],
        "collaboration",
        model=MODEL_GPT,
    )

    collab_b = revise_answer(
        premise,
        hypothesis,
        b["answer"],
        a["answer"],
        "collaboration",
        model=MODEL_GEMMA,
    )

    all_calls.extend([
        collab_a,
        collab_b,
    ])

    collab_final = decision_agent(
        premise,
        hypothesis,
        collab_a["answer"],
        collab_b["answer"],
        "collaboration",
        model=MODEL_GPT,
    )

    all_calls.append(collab_final)

    record["conditions"]["collaboration"] = {
        "agent_a_initial": a["answer"],
        "agent_b_initial": b["answer"],
        "agent_a_revised": collab_a["answer"],
        "agent_b_revised": collab_b["answer"],
        "final": collab_final["answer"],
    }

    # --------------------------------------------------------
    # CONDITION 4: DEBATE
    # --------------------------------------------------------

    debate_a = revise_answer(
        premise,
        hypothesis,
        a["answer"],
        b["answer"],
        "debate",
        model=MODEL_GPT,
    )

    debate_b = revise_answer(
        premise,
        hypothesis,
        b["answer"],
        a["answer"],
        "debate",
        model=MODEL_GEMMA,
    )

    all_calls.extend([
        debate_a,
        debate_b,
    ])

    debate_final = decision_agent(
        premise,
        hypothesis,
        debate_a["answer"],
        debate_b["answer"],
        "debate",
        model=MODEL_GPT,
    )

    all_calls.append(debate_final)

    record["conditions"]["debate"] = {
        "agent_a_initial": a["answer"],
        "agent_b_initial": b["answer"],
        "agent_a_revised": debate_a["answer"],
        "agent_b_revised": debate_b["answer"],
        "final": debate_final["answer"],
    }

    # --------------------------------------------------------
    # COMPUTE EXPERIMENT COSTS
    # --------------------------------------------------------

    record["api_calls"] = len(all_calls)

    record["latency_seconds"] = sum(
        c["latency"]
        for c in all_calls
    )

    record["prompt_tokens"] = sum(
        c["prompt_tokens"] or 0
        for c in all_calls
    )

    record["completion_tokens"] = sum(
        c["completion_tokens"] or 0
        for c in all_calls
    )

    return record


# ============================================================
# 11. MAIN EXPERIMENT
# ============================================================

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
        default="results/mixed_gpt_gemma_development.jsonl",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.dataset)

    required_columns = {
        "uid",
        "premise",
        "hypothesis",
        "label",
    }

    if not required_columns.issubset(df.columns):

        missing_columns = (
            required_columns - set(df.columns)
        )

        raise ValueError(
            f"Dataset missing columns: {sorted(missing_columns)}"
        )

    if args.limit is not None:
        df = df.head(args.limit)

    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    completed = set()

    if output_path.exists():

        with output_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.strip():

                    completed.add(
                        str(json.loads(line)["uid"])
                    )

    print(
        f"GPT model ID: {MODEL_GPT}",
        flush=True,
    )

    print(
        f"Gemma model ID: {MODEL_GEMMA}",
        flush=True,
    )

    print(
        f"Dataset examples: {len(df)}",
        flush=True,
    )

    print(
        f"Already completed: {len(completed)}",
        flush=True,
    )

    # --------------------------------------------------------
    # PROCESS DATASET
    # --------------------------------------------------------

    for index, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        if str(row["uid"]) in completed:
            continue

        print(
            f"\nProcessing example "
            f"{index}/{len(df)}: {row['uid']}",
            flush=True,
        )

        try:

            result = run_example(row)

            with output_path.open(
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    ) + "\n"
                )

            print(
                "Gold:",
                result["gold_label"],

                "| Gemma Single:",
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
                round(
                    result["latency_seconds"],
                    1,
                ),

                "seconds",

                flush=True,
            )

        except Exception as error:

            print(
                f"ERROR: {type(error).__name__}: {error}",
                flush=True,
            )

            print(
                "Stopping to avoid skipping an example.",
                flush=True,
            )

            break

    print(
        "\nExperiment run finished.",
        flush=True,
    )


# ============================================================
# 12. START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()