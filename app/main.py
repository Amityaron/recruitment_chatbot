"""
Main Entry Point
----------------
Initializes all components and starts the recruitment chatbot.

Run order:
  1. Check / create SQL Server database
  2. Check / create Chroma vector store
  3. Create conversation session
  4. Start the conversation loop

Usage:
  python -m app.main
"""

import os
from datetime import datetime, timezone

from app.modules.database.database import create_and_seed_db
from app.modules.embedding.embedding import run_embedding, CHROMA_DIR
from app.modules.memory.memory import create_session, end_session, add_bot_message
from app.modules.agents.agents import run_main_agent
from app.modules.prompts.prompts import JOB_POSITION


def setup():
    """
    One-time setup: create the database and vector store if they
    do not already exist.
    """
    # ── SQL Server DB ────────────────────────────────────────────
    create_and_seed_db()

    # ── Chroma DB ─────────────────────────────────────────────────
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        print("[Setup] Chroma DB not found — running embedding pipeline...")
        run_embedding()
    else:
        print("[Setup] Chroma DB found.")


def run_conversation():
    """
    Start a single recruitment conversation.
    Loops turn-by-turn until the agent decides to end or schedule.
    """
    session_id = create_session()
    timestamp  = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 50)
    print(f"Recruitment Chatbot — {JOB_POSITION} Position")
    print("Type 'quit' to exit at any time.")
    print("=" * 50 + "\n")

    opening = (
        f"Hi! Thanks for your interest in our {JOB_POSITION} opening. "
        "Could you tell me a bit about your relevant experience?"
    )
    print(f"Bot: {opening}\n")
    add_bot_message(session_id, opening)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("\nSession ended by user.")
            break

        if not user_input:
            continue

        result = run_main_agent(
            session_id=session_id,
            user_message=user_input,
            conversation_timestamp=timestamp,
        )

        print(f"\nBot: {result['response']}\n")

        if result["action"] in ("end", "schedule"):
            if result["action"] == "schedule":
                print("[System] Interview scheduling initiated.")
            else:
                print("[System] Conversation ended.")
            break

    end_session(session_id)


if __name__ == "__main__":
    setup()
    run_conversation()
