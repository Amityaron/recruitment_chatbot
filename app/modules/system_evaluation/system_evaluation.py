"""
System Evaluation Module
-------------------------
Evaluates the multi-agent system's end-to-end routing decision: given a
conversation, does it correctly classify the next action as
end / continue / schedule?

Design — classify_action makes a 3-step decision:
  1. Exit check - reuses the REAL Exit Advisor (run_exit_advisor from
     agents.py) — production code, including the Prompt Engineering
     improvements (see exit_advisor_prompt_engineering/). Catches candidate
     disengagement/rejection.
  2. Slot-confirmation check - the labeled dataset shows that most "end"
     turns are NOT rejections but successful booking wrap-ups (candidate
     just accepted a proposed time). Production handles this via
     is_booking_done() after book_interview runs; since historical data
     has no live booking state, this is detected directly from the
     conversation text instead.
  3. Schedule-readiness check - lightweight, DB-free. The real Scheduling
     Advisor depends on live schedule DB availability, which is not
     reproducible against historical conversations (the dataset's dates
     are fixed in the past). This checks the same readiness criteria
     defined in MAIN_AGENT_SYSTEM_PROMPT (enough background gathered to
     propose slots), without touching the DB.

Steps:
  1. load_test_cases  - reads ALL 3 labels (end/continue/schedule) from sms_conversations.json
  2. classify_action  - 3-step decision described above
  3. evaluate_system  - runs classify_action on every test case

Run: python -m app.modules.system_evaluation.system_evaluation
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import confusion_matrix, classification_report

from app.modules.agents.agents import run_exit_advisor

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
JSON_PATH   = os.path.join(DATA_DIR, "sms_conversations.json")


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
    Load labeled test cases from sms_conversations.json, keeping all 3
    labels as-is (end/continue/schedule) — no collapsing to binary.

    Returns
    -------
    list of {"context": str, "expected": "end"|"continue"|"schedule"}
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
                "expected": turn["label"],
            })
    return cases


# ═══════════════════════════════════════════════════════════════════
# SCHEDULE-READINESS CLASSIFIER (DB-free)
# ═══════════════════════════════════════════════════════════════════

SCHEDULE_READINESS_PROMPT = """
You are a Recruitment Coordinator deciding whether enough background has
been gathered from a candidate to justify proposing interview slots.

Propose interview slots (should_schedule: true) if EITHER:
- The candidate has shared at least some background (years of experience,
  tech stack, or availability) AND has expressed interest in moving
  forward, OR
- Interview times were already proposed earlier in the conversation and
  the candidate rejected them, said they don't work, or asked for
  different times — propose new slots rather than going back to
  gathering background.

Otherwise (should_schedule: false), more information should be gathered first.

Example:
Bot: We could do Monday at 2 PM or Tuesday at 10 AM — does either work?
Candidate: Neither of those works for me, sorry.
should_schedule: true  (rejected the proposed times, not the idea of scheduling)

Return ONLY valid JSON: {"should_schedule": true/false}
"""


def should_propose_schedule(context: str, client: OpenAI) -> bool:
    """
    Lightweight, DB-free check: has enough background been gathered to
    propose interview slots? Mirrors the scheduling-readiness criteria
    from MAIN_AGENT_SYSTEM_PROMPT without querying the live schedule DB.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SCHEDULE_READINESS_PROMPT.strip()},
            {"role": "user", "content": f"Conversation:\n{context}\n\nShould we propose interview slots now?"},
        ],
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return bool(result.get("should_schedule", False))
    except (json.JSONDecodeError, AttributeError):
        return False


# ═══════════════════════════════════════════════════════════════════
# SLOT-CONFIRMATION CLASSIFIER (mirrors is_booking_done in production)
# ═══════════════════════════════════════════════════════════════════
#
# In production (agents.py), once a candidate confirms an offered slot,
# book_interview runs and is_booking_done() forces "end" on the next turn.
# In this historical dataset there is no live booking state to check, so
# this classifier detects the same moment directly from the conversation
# text: did the candidate just accept a specific day/time that was offered?

SLOT_CONFIRMATION_PROMPT = """
You are a Recruitment Coordinator reviewing a conversation with a job candidate.

Check whether the candidate has accepted a proposed interview time, OR is
simply closing out a conversation where a time was already confirmed
earlier. Answer true for any of these:
- A specific day/time confirmation (e.g. "Monday at 3pm works", "I'll take
  the second option", "Wednesday at 10am is good for me").
- A short affirmative reply to a single proposed day, even without a
  specific time (e.g. Bot: "Does Thursday work for a quick call?"
  Candidate: "Works for me!").
- Accepting one of the offered options approximately — naming the right
  day but a different time than exactly what was offered, or accepting
  an earlier-offered option instead of the most recently offered one.
- A closing/farewell message after the interview was already confirmed
  earlier in the conversation (e.g. "Great, talk soon!").

Return ONLY valid JSON: {"slot_confirmed": true/false}
"""


def has_confirmed_slot(context: str, client: OpenAI) -> bool:
    """
    Detect whether the candidate just confirmed a specific interview slot.
    If true, the conversation should wrap up with a booking confirmation
    and end — same outcome as is_booking_done() in production.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SLOT_CONFIRMATION_PROMPT.strip()},
            {"role": "user", "content": f"Conversation:\n{context}\n\nHas the candidate just confirmed a specific interview time?"},
        ],
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return bool(result.get("slot_confirmed", False))
    except (json.JSONDecodeError, AttributeError):
        return False


# ═══════════════════════════════════════════════════════════════════
# SYSTEM-LEVEL 3-CLASS CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def classify_action(context: str, client: OpenAI) -> str:
    """
    Classify a conversation's next action as end / continue / schedule.

    Step 1 - reuse the real Exit Advisor (production code) - candidate
             disengagement/rejection.
    Step 2 - check slot confirmation - candidate just accepted a proposed
             time, so the next step is to book and wrap up (mirrors
             is_booking_done in production).
    Step 3 - if neither, check schedule-readiness (DB-free).
    """
    exit_result = run_exit_advisor(context)
    if exit_result.get("should_exit"):
        return "end"

    if has_confirmed_slot(context, client):
        return "end"

    if should_propose_schedule(context, client):
        return "schedule"

    return "continue"


def evaluate_system(test_cases: list, client: OpenAI) -> dict:
    """
    Run classify_action on every labeled test case.

    Returns
    -------
    dict: {"y_true": [...], "y_pred": [...]}
    Ready to feed directly into sklearn's accuracy_score / confusion_matrix.
    """
    y_true = []
    y_pred = []

    for i, case in enumerate(test_cases):
        predicted = classify_action(case["context"], client)
        y_true.append(case["expected"])
        y_pred.append(predicted)
        print(f"  [{i+1}/{len(test_cases)}] expected={case['expected']:<10} predicted={predicted}")

    return {"y_true": y_true, "y_pred": y_pred}


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_evaluation(json_path: str = JSON_PATH) -> dict:
    """
    Run the complete system evaluation and print accuracy.
    Returns y_true/y_pred for further analysis (e.g. in test_evals.ipynb).
    """
    print("=" * 50)
    print("Multi-Agent System Evaluation - End / Continue / Schedule")
    print("=" * 50)

    client     = OpenAI(api_key=OPENAI_API_KEY)
    test_cases = load_test_cases(json_path)
    print(f"Loaded {len(test_cases)} labeled test cases\n")

    results  = evaluate_system(test_cases, client)
    correct  = sum(1 for t, p in zip(results["y_true"], results["y_pred"]) if t == p)
    accuracy = correct / len(results["y_true"]) if results["y_true"] else 0.0

    print(f"\nAccuracy: {accuracy*100:.1f}%")

    # Per-class breakdown — overall accuracy alone can't tell you *where*
    # the errors are (e.g. "schedule" being confused with "continue" points
    # at a prompt/threshold issue; errors spread evenly across all 3
    # classes points more at the model's general judgment quality).
    labels = ["end", "continue", "schedule"]
    print("\nConfusion matrix (rows = expected, cols = predicted):")
    print("             " + "  ".join(f"{l:>10}" for l in labels))
    cm = confusion_matrix(results["y_true"], results["y_pred"], labels=labels)
    for label, row in zip(labels, cm):
        print(f"  {label:>10} " + "  ".join(f"{v:>10}" for v in row))

    print("\nPer-class precision/recall/f1:")
    print(classification_report(results["y_true"], results["y_pred"], labels=labels, zero_division=0))
    print("=" * 50)

    return results


# ── Run directly ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()
