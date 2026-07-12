"""
Exit Advisor Prompt Engineering Module
---------------------------------------
Evaluates and compares different versions of the Exit Advisor prompt ONLY.
(Info Advisor and Scheduling Advisor are not covered here — they were never
fine-tuned, so they have no "before/after" version to compare.)

Replaces the previous fine-tuning approach. Instead of training a custom model,
we improve Exit Advisor performance through Prompt Engineering techniques:
  - Chain-of-Thought (step-by-step reasoning instruction)
  - Few-Shot examples (embedded directly in the system prompt)
  - Explicit edge-case handling (hesitation, short replies, competing offers)

Steps:
  1. load_test_cases       - reads labeled SMS data from sms_conversations.json
  2. evaluate_prompt       - runs a prompt on all test cases, returns metrics
  3. compare_prompts       - compares PROMPT_V1 (basic) vs PROMPT_V2 (engineered)

Run: python -m app.modules.exit_advisor_prompt_engineering.exit_advisor_prompt_engineering
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from app.modules.prompts.prompts import EXIT_ADVISOR_SYSTEM_PROMPT

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
JSON_PATH   = os.path.join(DATA_DIR, "sms_conversations.json")


# ═══════════════════════════════════════════════════════════════════
# PROMPT VERSIONS
# ═══════════════════════════════════════════════════════════════════

# V1 — Basic prompt (no examples, no reasoning steps)
PROMPT_V1 = """
You are a Conversation Exit Specialist for a recruitment chatbot.

Your ONLY job: decide whether the conversation should END.

Analyze the conversation and return ONLY valid JSON:
{"should_exit": true, "reason": "..."} or {"should_exit": false, "reason": "..."}

End when: candidate found another job, is not interested, is rude, or asks to stop.
Do NOT end when: candidate has questions, needs to reschedule, or is still engaging.
"""

# V2 — Improved prompt (Chain-of-Thought + Few-Shot + edge cases)
PROMPT_V2 = EXIT_ADVISOR_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def _build_context(turns: list, up_to_index: int) -> str:
    """Build a conversation string from turns before up_to_index."""
    lines = []
    for turn in turns[:up_to_index]:
        speaker = "Candidate" if turn["speaker"] == "candidate" else "Bot"
        if turn.get("text"):
            lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def load_test_cases(json_path: str = JSON_PATH) -> list:
    """
    Load labeled test cases from sms_conversations.json.

    For each labeled recruiter turn:
      label == "end"                  -> expected should_exit = True
      label == "continue"/"schedule"  -> expected should_exit = False

    Returns
    -------
    list of {"context": str, "expected": bool}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    cases = []
    for conv in conversations:
        turns = conv["turns"]
        for i, turn in enumerate(turns):
            if turn["speaker"] != "recruiter" or turn.get("label") is None:
                continue
            context = _build_context(turns, up_to_index=i)
            cases.append({
                "context":  context,
                "expected": turn["label"] == "end",
            })
    return cases


# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def run_prompt(prompt: str, context: str, client: OpenAI) -> bool:
    """
    Run a single prompt against a conversation context.
    Returns the predicted should_exit value (bool).
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": prompt.strip()},
            {
                "role": "user",
                "content": f"Conversation:\n{context}\n\nShould we end this conversation?",
            },
        ],
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return bool(result.get("should_exit", False))
    except (json.JSONDecodeError, AttributeError):
        return False


def evaluate_prompt(prompt: str, test_cases: list, client: OpenAI) -> dict:
    """
    Evaluate a prompt version on all labeled test cases.

    Returns
    -------
    dict: {
        "accuracy":    float,
        "tp": int, "fp": int, "tn": int, "fn": int,
        "predictions": list of {"predicted": bool, "expected": bool}
    }
    """
    tp = fp = tn = fn = 0
    predictions = []

    for i, case in enumerate(test_cases):
        predicted = run_prompt(prompt, case["context"], client)
        expected  = case["expected"]
        predictions.append({"predicted": predicted, "expected": expected})

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        else:
            fn += 1

        print(f"  [{i+1}/{len(test_cases)}] expected={expected} predicted={predicted}")

    total    = len(test_cases)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "accuracy":    round(accuracy, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "predictions": predictions,
    }


# ═══════════════════════════════════════════════════════════════════
# COMPARISON PIPELINE
# ═══════════════════════════════════════════════════════════════════

def compare_prompts(json_path: str = JSON_PATH) -> dict:
    """
    Compare PROMPT_V1 (basic) vs PROMPT_V2 (prompt engineering) on labeled SMS data.

    Prints accuracy, TP/FP/TN/FN, and improvement for each version.
    Returns both result dicts for use in test_evals.ipynb.
    """
    print("=" * 50)
    print("Exit Advisor - Prompt Engineering Evaluation")
    print("=" * 50)

    client     = OpenAI(api_key=OPENAI_API_KEY)
    test_cases = load_test_cases(json_path)
    print(f"Loaded {len(test_cases)} test cases\n")

    print("Evaluating V1 (basic prompt)...")
    v1 = evaluate_prompt(PROMPT_V1, test_cases, client)
    print(f"\nV1 Accuracy : {v1['accuracy']*100:.1f}%")
    print(f"  TP={v1['tp']}  FP={v1['fp']}  TN={v1['tn']}  FN={v1['fn']}\n")

    print("Evaluating V2 (prompt engineering)...")
    v2 = evaluate_prompt(PROMPT_V2, test_cases, client)
    print(f"\nV2 Accuracy : {v2['accuracy']*100:.1f}%")
    print(f"  TP={v2['tp']}  FP={v2['fp']}  TN={v2['tn']}  FN={v2['fn']}\n")

    improvement = (v2["accuracy"] - v1["accuracy"]) * 100
    print(f"Improvement : {improvement:+.1f} percentage points")
    print("=" * 50)

    return {"v1": v1, "v2": v2}


# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    compare_prompts()
