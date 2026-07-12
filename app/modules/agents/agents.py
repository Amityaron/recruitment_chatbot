"""
Agents Module
-------------
Implements all five agents using LangChain + OpenAI:

  1. Exit Advisor          - decides if the conversation should end
  2. Info Advisor          - answers candidate questions using Chroma DB (RAG)
  3. Qualification Advisor - checks stated background against the job's minimum requirements (RAG)
  4. Scheduling Advisor    - finds available interview slots via function calling
  5. Main Agent            - orchestrates the advisors and decides the next action
"""

import os
import re
import json
import difflib
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.modules.prompts.prompts import (
    MAIN_AGENT_SYSTEM_PROMPT, MAIN_AGENT_PARAMS, MAIN_AGENT_FEW_SHOT,
    EXIT_ADVISOR_SYSTEM_PROMPT, EXIT_ADVISOR_PARAMS,
    INFO_ADVISOR_SYSTEM_PROMPT, INFO_ADVISOR_PARAMS,
    QUALIFICATION_ADVISOR_SYSTEM_PROMPT, QUALIFICATION_ADVISOR_PARAMS,
    DB_POSITION,
)
from app.modules.memory.memory import (
    ConversationBufferMemory,
    create_memory,
    get_memory,
    get_history_as_text,
    add_bot_message,
    store_offered_slots,
    get_offered_slots,
    set_booking_done,
    is_booking_done,
    get_booked_slot,
    get_turn_count,
    increment_turn_count,
)
from app.modules.embedding.embedding import load_vector_store
from app.modules.database.database import get_available_slots, book_slot, VALID_WEEKDAYS

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Minimum number of candidate turns required before the scheduling_advisor
# tool is made available to the Main Agent. Enforced in code (not just the
# prompt) because the LLM does not reliably follow the "ask 3 questions
# first" instruction on its own — it has been observed calling
# scheduling_advisor on the very first message with no candidate info at all.
MIN_TURNS_BEFORE_SCHEDULING = 3


def _make_llm(params: dict, api_key: str = OPENAI_API_KEY) -> ChatOpenAI:
    """Create a ChatOpenAI instance with the given parameters."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=params.get("temperature", 0.5),
        max_tokens=params.get("max_tokens", 200),
    )


# ═══════════════════════════════════════════════════════════════════
# 1. EXIT ADVISOR
# ═══════════════════════════════════════════════════════════════════

def run_exit_advisor(history_text: str) -> dict:
    """
    Decide whether the conversation should end.

    Parameters
    ----------
    history_text : full conversation history as plain text

    Returns
    -------
    dict: {"should_exit": bool, "reason": str}
    """
    llm = _make_llm(EXIT_ADVISOR_PARAMS)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXIT_ADVISOR_SYSTEM_PROMPT),
        ("human", "Conversation:\n{history}\n\nShould we end this conversation?"),
    ])

    chain  = prompt | llm
    result = chain.invoke({"history": history_text})

    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        # Fallback if model returns non-JSON
        return {"should_exit": False, "reason": "parse error - defaulting to continue"}


# ═══════════════════════════════════════════════════════════════════
# 2. INFO ADVISOR
# ═══════════════════════════════════════════════════════════════════

def run_info_advisor(question: str, history_text: str) -> str:
    """
    Answer a candidate question using RAG (Chroma DB).

    Parameters
    ----------
    question     : the candidate's question
    history_text : conversation history for context

    Returns
    -------
    str: answer to pass back to the candidate
    """
    llm          = _make_llm(INFO_ADVISOR_PARAMS)
    vector_store = load_vector_store()
    retriever    = vector_store.as_retriever(search_kwargs={"k": 3})

    # Retrieve relevant chunks from the job description PDF
    docs    = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", INFO_ADVISOR_SYSTEM_PROMPT),
        ("human",
         "Job description context:\n{context}\n\n"
         "Conversation so far:\n{history}\n\n"
         "Candidate question: {question}\n\n"
         "Your answer:"),
    ])

    chain  = prompt | llm
    result = chain.invoke({
        "context":  context,
        "history":  history_text,
        "question": question,
    })
    return result.content


# ═══════════════════════════════════════════════════════════════════
# 3. QUALIFICATION ADVISOR
# ═══════════════════════════════════════════════════════════════════

def run_qualification_advisor(history_text: str) -> dict:
    """
    Check whether the candidate's stated background meets the job's
    minimum requirements, using RAG on the job description PDF.

    Parameters
    ----------
    history_text : full conversation history as plain text

    Returns
    -------
    dict: {"meets_requirements": bool, "reason": str}
    """
    llm          = _make_llm(QUALIFICATION_ADVISOR_PARAMS)
    vector_store = load_vector_store()
    retriever    = vector_store.as_retriever(search_kwargs={"k": 3})

    docs    = retriever.invoke("required skills and qualifications, minimum years of experience")
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUALIFICATION_ADVISOR_SYSTEM_PROMPT),
        ("human",
         "Job requirements context:\n{context}\n\n"
         "Conversation so far:\n{history}\n\n"
         "Does the candidate meet the minimum requirements?"),
    ])

    chain  = prompt | llm
    result = chain.invoke({"context": context, "history": history_text})

    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        return {"meets_requirements": True, "reason": "parse error - defaulting to meets requirements"}


# ═══════════════════════════════════════════════════════════════════
# 4. SCHEDULING ADVISOR
# ═══════════════════════════════════════════════════════════════════

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Python's date.weekday(): Monday=0 ... Sunday=6.
_HEBREW_WEEKDAYS = {
    "יום ראשון": 6, "יום שני": 0, "יום שלישי": 1, "יום רביעי": 2,
    "יום חמישי": 3, "יום שישי": 4, "שבת": 5,
}


def _parse_weekday_date(text: str, conversation_timestamp: str):
    """
    Deterministically resolve an explicit weekday name (English "Wednesday"
    or Hebrew "יום רביעי") in `text` to a concrete YYYY-MM-DD date on or
    after the conversation's reference date. Returns None if no weekday
    name is found, so the caller can fall back to showing the nearest
    real availability instead of guessing.

    This exists because leaving date arithmetic to an LLM call was
    observed silently falling back to today's date whenever it failed to
    compute the right date from a bare day name — ignoring what the
    candidate actually asked for. A weekday name is unambiguous; there's
    no reason to leave it to an LLM to get the arithmetic right.

    Tolerates common English typos ("wensday", "tuesady") via a fuzzy
    match — an SMS-style chatbot audience will misspell weekday names
    often enough that an exact-match-only regex would silently miss them.
    """
    target_weekday = None

    for phrase, weekday in _HEBREW_WEEKDAYS.items():
        if phrase in text:
            target_weekday = weekday
            break

    if target_weekday is None:
        exact = re.search(
            r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            text, re.IGNORECASE,
        )
        if exact:
            target_weekday = _WEEKDAYS[exact.group(1).lower()]
        else:
            for token in re.findall(r"[a-zA-Z]+", text.lower()):
                close = difflib.get_close_matches(token, _WEEKDAYS.keys(), n=1, cutoff=0.75)
                if close:
                    target_weekday = _WEEKDAYS[close[0]]
                    break

    if target_weekday is None:
        return None

    today = datetime.fromisoformat(conversation_timestamp.replace("Z", "+00:00")).date()
    days_ahead = (target_weekday - today.weekday()) % 7
    return str(today + timedelta(days=days_ahead))


_HEBREW_DAY_COUNT_WORDS = {
    "יום": 1, "יומיים": 2, "שלושה ימים": 3, "ארבעה ימים": 4,
    "חמישה ימים": 5, "שישה ימים": 6, "שבעה ימים": 7,
}


def _parse_relative_date(text: str, conversation_timestamp: str):
    """
    Deterministically resolve simple relative-date phrases — "today"/
    "tomorrow"/"in N days", and their Hebrew equivalents ("היום"/"מחר"/
    "עוד N ימים") — before falling back to weekday-name matching, then
    None if nothing matched (caller then defaults to the nearest real
    availability rather than guessing a date via an LLM — see
    run_scheduling_advisor).

    There used to be an LLM-based fallback here (_infer_target_date) for
    phrasing this function doesn't recognize. It was removed: asking an
    LLM to compute "today + 2 days at noon" was observed silently
    defaulting to today's date on any format mismatch, and separately
    causing the Main Agent to loop trying to construct a tool call and
    hit LangChain's max_iterations limit. If a request can't be parsed
    with confidence, showing the nearest real availability is more
    honest and more reliable than a guess.
    """
    today = datetime.fromisoformat(conversation_timestamp.replace("Z", "+00:00")).date()

    if re.search(r"\btoday\b", text, re.IGNORECASE) or "היום" in text:
        return str(today)
    if re.search(r"\btomorrow\b", text, re.IGNORECASE) or "מחר" in text:
        return str(today + timedelta(days=1))

    match = re.search(r"\bin\s+(\d+)\s+days?\b", text, re.IGNORECASE)
    if match:
        return str(today + timedelta(days=int(match.group(1))))

    match = re.search(r"(?:עוד|בעוד)\s+(\d+)\s+ימים", text)
    if match:
        return str(today + timedelta(days=int(match.group(1))))

    for phrase, n_days in _HEBREW_DAY_COUNT_WORDS.items():
        if re.search(rf"(?:עוד|בעוד)\s+{phrase}", text):
            return str(today + timedelta(days=n_days))

    return _parse_weekday_date(text, conversation_timestamp)


def _build_slot_offer_message(slots: list, requested_date: str = None) -> str:
    """
    Deterministically format the offered slots as a numbered, dated list.

    This used to be phrased freely by an LLM call, but that was observed to
    drop the numbering and/or the real date (e.g. showing "10 AM, 1 PM, or
    3 PM" with no date at all) — which then made the candidate's reply
    harder to map back to a slot number. Building it in code guarantees the
    candidate always sees exactly what's in `slots`, in order.

    requested_date : the date actually asked for (e.g. resolved from
        "Wednesday"), if any. get_available_slots searches on/after this
        date, so if it's fully booked the returned slots silently land on
        a later day — call that out explicitly instead of just switching
        days without explanation (observed confusing candidates who named
        a specific day and got a different one back with no context).
    """
    lines = []
    for i, slot in enumerate(slots):
        day    = datetime.strptime(slot["date"], "%Y-%m-%d")
        hour, minute = map(int, slot["time"].split(":"))
        period = "AM" if hour < 12 else "PM"
        hour12 = hour % 12 or 12
        lines.append(f"{i + 1}. {day.strftime('%A, %B %d')} at {hour12}:{minute:02d} {period}")

    intro = "Great! Here are our available interview slots:"
    if requested_date and slots and slots[0]["date"] != requested_date:
        requested_day_obj = datetime.strptime(requested_date, "%Y-%m-%d")
        if requested_day_obj.weekday() not in VALID_WEEKDAYS:
            # Saturdays/Mondays were never in the schedule at all — "fully
            # booked" would be misleading (implies slots existed and got
            # taken, not that we simply don't interview that day).
            intro = f"We don't hold interviews on {requested_day_obj.strftime('%A')}s, but here's our nearest availability:"
        else:
            requested_day = requested_day_obj.strftime("%A, %B %d")
            intro = f"{requested_day} is fully booked, but here's our nearest availability:"

    return (
        intro + "\n"
        + "\n".join(lines)
        + "\nWhich one works best for you? Just reply with 1, 2, or 3."
    )


def run_scheduling_advisor(
    history_text: str,
    conversation_timestamp: str,
    position: str = DB_POSITION,
    latest_message: str = "",
) -> dict:
    """
    Find 3 available interview slots and compose a message for the candidate.

    Parameters
    ----------
    history_text             : conversation history as plain text
    conversation_timestamp   : ISO timestamp of when the conversation started
    position                 : job position to filter slots (default 'Python Dev')
    latest_message            : the candidate's most recent message, checked
                                 for an explicit weekday name BEFORE the rest
                                 of history — otherwise a day name the bot
                                 itself mentioned earlier (e.g. offering
                                 "Tuesday or Wednesday?") could match first
                                 and silently override what the candidate
                                 actually said.

    Returns
    -------
    dict: {
        "should_schedule": bool,
        "slots": [...],
        "message": str
    }
    """
    # If we can't confidently parse a specific date/day the candidate asked
    # for, don't guess with an LLM — just default to today and let
    # get_available_slots' `>=` search find the nearest real availability.
    # Keep the parsed result separate from the fallback: _build_slot_offer_
    # message should only say "X is unavailable" when the candidate
    # actually asked for X — not when X is just our internal default
    # (observed responding "We don't hold interviews on Saturdays" to a
    # candidate who only said "I'm available this week" and never
    # mentioned Saturday at all).
    parsed_date = (
        _parse_relative_date(latest_message, conversation_timestamp)
        or _parse_relative_date(history_text, conversation_timestamp)
    )
    today = str(datetime.fromisoformat(conversation_timestamp.replace("Z", "+00:00")).date())
    target_date = parsed_date or today
    slots = get_available_slots(target_date, position, n=3)

    if not slots:
        return {
            "should_schedule": False,
            "slots": [],
            "message": "I'm sorry, there are no available slots at that time. Can you suggest another date?",
        }

    return {
        "should_schedule": True,
        "slots": slots,
        "message": _build_slot_offer_message(slots, requested_date=parsed_date),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN AGENT
# ═══════════════════════════════════════════════════════════════════

_SLOT_PICK_KEYWORDS = ("option", "slot", "number", "pick", "choose", "take", "go with", "works")

# Matches a clock time like "10 AM" or "2:30 PM" — used to catch the Main
# Agent inventing plausible-looking interview times in freeform text
# without ever calling scheduling_advisor (see Step 2e in run_main_agent).
_TIME_MENTION_RE = re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", re.IGNORECASE)

# LangChain's own default message when an AgentExecutor's ReAct loop hits
# its iteration cap without producing a final answer — observed when the
# candidate mixes a relative date with a time (e.g. "in 2 days at 12:00"),
# which the Main Agent apparently struggled to turn into a tool call and
# kept retrying. Caught here so the candidate never sees this raw internal
# error string (see Step 2e in run_main_agent).
_AGENT_FAILURE_RE = re.compile(r"agent stopped due to", re.IGNORECASE)

_SCHEDULING_STALL_KEYWORDS = (
    "hold on", "one moment", "get back to you", "shortly", "let me check",
    "let me find", "i'll check", "i'll find", "i'll look",
    "give me a moment", "give me a second",
)
_SCHEDULING_TOPIC_KEYWORDS = ("slot", "interview", "availab", "schedule", "book")


def _looks_like_scheduling_stall(text: str) -> bool:
    """
    True if the Main Agent said something like "I'll find some slots and
    get back to you shortly" — i.e. promised to check scheduling but
    didn't actually call scheduling_advisor this turn, leaving the
    candidate stuck until they send another message. Requires both a
    deferral phrase AND a scheduling-topic word so an unrelated "let me
    check with the team on salary" doesn't get caught.
    """
    lower = text.lower()
    return (
        any(kw in lower for kw in _SCHEDULING_STALL_KEYWORDS)
        and any(kw in lower for kw in _SCHEDULING_TOPIC_KEYWORDS)
    )


def _parse_slot_number(text: str, n_slots: int):
    """
    Return the 0-based index of the slot a short reply picks (e.g. "2",
    "option 2", "I'll take slot 2"), or None if the message doesn't look
    like a slot pick.

    Deliberately conservative: requires exactly one number in the message,
    and either a short overall length or an explicit picking keyword — so
    an unrelated number elsewhere in the conversation (e.g. "I have 2 years
    of experience", "I'm 25 years old") isn't mistaken for a slot pick. The
    out-of-range check below is a second safety net, since real offers only
    ever have 1-3 slots.
    """
    digits = re.findall(r"\d+", text)
    if len(digits) != 1:
        return None
    stripped = text.strip()
    looks_like_pick = len(stripped) <= 15 or any(kw in stripped.lower() for kw in _SLOT_PICK_KEYWORDS)
    if not looks_like_pick:
        return None
    index = int(digits[0]) - 1
    return index if 0 <= index < n_slots else None


def _book_offered_slot(session_id: str, slots: list, index: int) -> str:
    """
    Book slots[index] and record it in session memory.

    Shared by the deterministic slot-pick check in run_main_agent (Step 1.5)
    and the book_interview tool, so both paths apply the exact same
    bounds-check/booking/state-update logic.
    """
    if index < 0 or index >= len(slots):
        return f"Invalid slot number. Choose between 1 and {len(slots)}."
    chosen = slots[index]
    if book_slot(chosen["ScheduleID"]):
        set_booking_done(session_id, chosen)
        return f"Booking confirmed: {chosen['date']} at {chosen['time']} (ID: {chosen['ScheduleID']})"
    return "Sorry, that slot was just taken. Please ask for new options."


def _build_main_agent(session_id: str, user_message: str) -> AgentExecutor:
    """
    Build the Main Agent with its advisor tools.
    Called once per conversation turn.

    user_message is needed here (not just inside run_main_agent) because
    scheduling_tool below has to reconstruct "history so far, including
    what the candidate just said" — get_history_as_text() alone lags one
    turn behind, since the memory buffer isn't updated with this turn's
    exchange until after agent_executor.invoke() returns.
    """
    llm    = _make_llm(MAIN_AGENT_PARAMS)
    memory = get_memory(session_id)

    # NOTE: exit_advisor is intentionally NOT exposed as a Main Agent tool.
    # run_main_agent's Step 1 already calls run_exit_advisor() deterministically
    # on every turn, before the Main Agent ever runs. Binding it again here
    # would let the Main Agent re-run the same (non-deterministic) check a
    # second time and potentially get a different answer than Step 1 already
    # committed to — observed in testing to occasionally contradict Step 1
    # on neutral/ambiguous messages.

    # ── Tool: Info Advisor ────────────────────────────────────────
    def info_tool(question: str) -> str:
        history = get_history_as_text(session_id)
        return run_info_advisor(question, history)

    # ── Tool: Qualification Advisor ────────────────────────────────
    def qualification_tool(history: str) -> str:
        result = run_qualification_advisor(history)
        return json.dumps(result)

    # ── Tool: Scheduling Advisor ──────────────────────────────────
    def scheduling_tool(timestamp: str) -> str:
        # Append user_message explicitly — get_history_as_text() alone
        # would miss it (see docstring above), and _infer_target_date()
        # needs to see exactly what the candidate just asked for (e.g.
        # "how about next Wednesday?") to resolve the right date.
        history = get_history_as_text(session_id) + f"\nCandidate: {user_message}"
        result  = run_scheduling_advisor(history, timestamp, latest_message=user_message)
        if result.get("should_schedule") and result.get("slots"):
            store_offered_slots(session_id, result["slots"])
        return json.dumps(result)

    # ── Tool: Book Interview ──────────────────────────────────────
    def book_tool(slot_number: str) -> str:
        slots = get_offered_slots(session_id)
        if not slots:
            return "No slots available to book. Please run scheduling first."
        index = _parse_slot_number(slot_number, len(slots))
        if index is None:
            return "Could not parse slot number. Please provide a number like 1, 2, or 3."
        return _book_offered_slot(session_id, slots, index)

    tools = [
        Tool(
            name="info_advisor",
            func=info_tool,
            description=(
                "Use to answer a candidate's question about the job. "
                "Input: the candidate's question. "
                "Returns a ready-to-send answer."
            ),
        ),
        Tool(
            name="qualification_check",
            func=qualification_tool,
            description=(
                "Use BEFORE proposing interview slots to confirm the candidate "
                "meets the role's minimum requirements. "
                "Input: full conversation history as text. "
                "Returns JSON with meets_requirements (bool) and reason."
            ),
        ),
        Tool(
            name="scheduling_advisor",
            func=scheduling_tool,
            description=(
                "Use to find available interview slots and propose them. "
                "Input: the conversation start timestamp (ISO format). "
                "Returns JSON with slots and a ready-to-send message."
            ),
        ),
        Tool(
            name="book_interview",
            func=book_tool,
            description=(
                "Use when the candidate confirms a slot choice. "
                "Input: slot number as a string ('1', '2', or '3'). "
                "Returns confirmation or error message."
            ),
        ),
    ]

    # ── Hard gate: scheduling_advisor is not even visible to the LLM ─────
    # until enough turns have passed. The "ask 3 questions first" prompt
    # rule is a suggestion the LLM can ignore; removing the tool from its
    # toolkit entirely is a real constraint it cannot bypass.
    if get_turn_count(session_id) < MIN_TURNS_BEFORE_SCHEDULING:
        tools = [t for t in tools if t.name != "scheduling_advisor"]

    # ── Few-shot examples in system prompt ───────────────────────
    few_shot_text = "\n\n".join([
        f"Example {i+1}:\nConversation: {ex['conversation']}\nAction: {ex['action']}\nResponse: {ex['response']}"
        for i, ex in enumerate(MAIN_AGENT_FEW_SHOT)
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", MAIN_AGENT_SYSTEM_PROMPT + "\n\nExamples:\n" + few_shot_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def run_main_agent(
    session_id: str,
    user_message: str,
    conversation_timestamp: str = None,
) -> dict:
    """
    Process one conversation turn.

    Parameters
    ----------
    session_id               : active session ID
    user_message             : candidate's latest message
    conversation_timestamp   : ISO timestamp (defaults to now if not provided)

    Returns
    -------
    dict: {
        "action":   "continue" | "end" | "schedule",
        "response": str   (the message to send back to the candidate)
    }
    """
    if conversation_timestamp is None:
        conversation_timestamp = datetime.now(timezone.utc).isoformat()

    # ── Step 0: If booking already done, end gracefully ──────────
    if is_booking_done(session_id):
        response = "You're all set! See you at your interview. Have a great day!"
        add_bot_message(session_id, response)
        return {"action": "end", "response": response}

    history_text = get_history_as_text(session_id)

    # ── Step 1: Check if we should exit ──────────────────────────
    exit_result = run_exit_advisor(history_text + f"\nCandidate: {user_message}")
    if exit_result.get("should_exit"):
        response = (
            "Thank you for your time! We wish you all the best. "
            "Feel free to reach out if anything changes. "
        )
        add_bot_message(session_id, response)
        return {"action": "end", "response": response}

    # ── Step 1.5: Deterministically detect & book a slot pick ────
    # book_interview only runs if the Main Agent's LLM decides to call it —
    # observed in testing to sometimes just reply with a confirmation
    # sentence without invoking the tool, leaving the slot `available = 1`
    # in SQL Server forever. Mirrors the MIN_TURNS_BEFORE_SCHEDULING gate
    # above: enforce the real state change in code instead of trusting the
    # LLM to remember to call the tool.
    offered_slots = get_offered_slots(session_id)
    if offered_slots and not is_booking_done(session_id):
        slot_index = _parse_slot_number(user_message, len(offered_slots))
        if slot_index is not None:
            _book_offered_slot(session_id, offered_slots, slot_index)

    # ── Step 2: Run the main agent (may call info or scheduling) ─
    agent_executor = _build_main_agent(session_id, user_message)
    result = agent_executor.invoke({
        "input": (
            f"[Timestamp: {conversation_timestamp}]\n"
            f"Candidate: {user_message}\n\n"
            "Write your next SMS message to the candidate. "
            "Respond with ONLY the message text — never the word "
            "'continue', 'end', or 'schedule' by itself."
        ),
    })

    response = result.get("output", "")
    increment_turn_count(session_id)

    intermediate_steps = result.get("intermediate_steps", [])
    scheduling_step = next(
        (step for step in intermediate_steps if step[0].tool == "scheduling_advisor"),
        None,
    )
    # Set inside Step 2d / Step 2e below when a real (non-tool-call) DB
    # requery produces a fresh offer — Step 3 can't see these from
    # intermediate_steps since they bypass the LangChain tool-calling path.
    stale_offer_refreshed = False
    hallucinated_offer    = False

    # ── Step 2b: If booking just completed this turn, don't trust the ────
    # LLM's own paraphrase of what was booked — it has the full chat_history
    # (including earlier offer text) in context and nothing forces it to
    # quote book_interview's return value verbatim. Build the confirmation
    # deterministically from the slot that was actually written to the DB.
    if is_booking_done(session_id):
        booked = get_booked_slot(session_id)
        if booked:
            response = (
                f"Your interview is confirmed for {booked['date']} at {booked['time']}. "
                "Looking forward to speaking with you then! Best of luck!"
            )
    elif scheduling_step is not None:
        # ── Step 2c: Same reliability gap as booking — the Main Agent still
        # composes its own final SMS text after seeing scheduling_advisor's
        # tool output, and nothing forces it to relay `message` verbatim
        # (observed dropping the numbering / date, same as the old offer
        # bug). Trust the deterministic message the tool actually returned.
        scheduling_result = json.loads(scheduling_step[1])
        if scheduling_result.get("message"):
            response = scheduling_result["message"]
    elif offered_slots and slot_index is None:
        # ── Step 2d: Slots are still pending from an earlier turn and the
        # candidate didn't just pick one this turn — could be a vague reply
        # ("I can do it next week") or an explicit rejection asking for a
        # different day ("is there another day?", "i want another option").
        # Observed bug: just echoing the same cached `offered_slots` kept
        # insisting on the same day even after the candidate explicitly
        # asked for a different one. Don't trust freeform LLM text *or* a
        # stale cache — re-derive fresh from the real schedule DB every
        # time, using the candidate's latest message so a genuine change
        # of day request actually gets picked up. Skip only if the
        # candidate asked an unrelated question this turn (info_advisor
        # was called) — don't clobber the real answer.
        answered_a_question = any(step[0].tool == "info_advisor" for step in intermediate_steps)
        if not answered_a_question:
            fresh_result = run_scheduling_advisor(
                history_text + f"\nCandidate: {user_message}", conversation_timestamp,
                latest_message=user_message,
            )
            if fresh_result.get("should_schedule") and fresh_result.get("slots"):
                store_offered_slots(session_id, fresh_result["slots"])
                stale_offer_refreshed = True
            if fresh_result.get("message"):
                response = fresh_result["message"]

    if (
        not offered_slots
        and not is_booking_done(session_id)
        and scheduling_step is None
        and (
            _TIME_MENTION_RE.search(response)
            or _looks_like_scheduling_stall(response)
            or _AGENT_FAILURE_RE.search(response)
        )
    ):
        # ── Step 2e: No slots have ever actually been offered this session
        # (scheduling_advisor was never called), yet the response either
        # mentions clock times anyway (inventing a plausible-sounding offer
        # like "Wednesday at 10 AM, 1 PM, or 3 PM"), stalls with something
        # like "I'll find some slots and get back to you shortly" instead
        # of actually checking, or is LangChain's own "agent stopped due to
        # max iterations" failure message — in every case the candidate is
        # left with nothing real. Force the real check now, so anything
        # that looks like an offer (or a promise of one, or a failure)
        # always traces back to an actual query in this same turn.
        real_result = run_scheduling_advisor(
            history_text + f"\nCandidate: {user_message}", conversation_timestamp,
            latest_message=user_message,
        )
        if real_result.get("should_schedule") and real_result.get("slots"):
            store_offered_slots(session_id, real_result["slots"])
            hallucinated_offer = True
        if real_result.get("message"):
            response = real_result["message"]

    add_bot_message(session_id, response)

    # ── Step 3: Determine the action label from actual tool calls ────
    # Ground truth, not text pattern matching: a free-text response can
    # accidentally contain schedule-ish substrings (e.g. "available" inside
    # an unrelated clarifying question, or "am"/"pm" inside ordinary words
    # like "framework") and get misclassified. Instead, check which tools
    # the Main Agent actually invoked this turn.
    if is_booking_done(session_id):
        action = "end"
    elif any(
        step[0].tool == "qualification_check" and json.loads(step[1]).get("meets_requirements") is False
        for step in intermediate_steps
    ):
        action = "end"
    elif any(
        step[0].tool == "scheduling_advisor" and json.loads(step[1]).get("should_schedule")
        for step in intermediate_steps
    ):
        action = "schedule"
    elif hallucinated_offer or stale_offer_refreshed:
        action = "schedule"
    else:
        action = "continue"

    return {"action": action, "response": response}
