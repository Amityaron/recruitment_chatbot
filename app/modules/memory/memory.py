"""
Memory Module
-------------
Manages conversation history per candidate session.
Each candidate gets a unique session_id mapped to their own memory.

Sessions dict structure:
    {
        "session_id": ConversationBufferMemory,
        ...
    }
"""

import uuid
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

# Global sessions store — maps session_id -> memory
_sessions: dict = {}

# Offered slots store — maps session_id -> list of slots from scheduling advisor
_offered_slots: dict = {}

# Booking status — maps session_id -> bool (True if interview was booked)
_booking_done: dict = {}

# Booked slot — maps session_id -> the exact slot dict that was actually booked
_booked_slot: dict = {}

# Turn count — maps session_id -> number of completed Main Agent turns.
# Used to hard-gate the scheduling_advisor tool (deterministic, not just a
# prompt instruction the LLM might not always follow).
_turn_count: dict = {}


# ═══════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def create_session() -> str:
    """
    Create a new session for a candidate.
    Returns a unique session_id.
    """
    session_id = str(uuid.uuid4())
    _sessions[session_id] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    return session_id


def get_memory(session_id: str) -> ConversationBufferMemory:
    """
    Return the memory object for a given session_id.
    Raises KeyError if session does not exist.
    """
    if session_id not in _sessions:
        raise KeyError(f"Session '{session_id}' not found.")
    return _sessions[session_id]


def end_session(session_id: str) -> None:
    """
    Remove a session from the store when the conversation ends.
    """
    if session_id in _sessions:
        del _sessions[session_id]
    _offered_slots.pop(session_id, None)
    _booking_done.pop(session_id, None)
    _booked_slot.pop(session_id, None)
    _turn_count.pop(session_id, None)


# ═══════════════════════════════════════════════════════════════════
# SLOTS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def store_offered_slots(session_id: str, slots: list) -> None:
    """Save the slots offered to the candidate so we can book the chosen one."""
    _offered_slots[session_id] = slots


def get_offered_slots(session_id: str) -> list:
    """Return the slots that were offered to the candidate."""
    return _offered_slots.get(session_id, [])


def set_booking_done(session_id: str, slot: dict = None) -> None:
    """
    Mark that an interview was successfully booked for this session.

    Parameters
    ----------
    slot : the exact slot dict that was actually booked (date/time/ScheduleID),
           so the confirmation message can be built from the real booking
           instead of the LLM's own paraphrase of the conversation.
    """
    _booking_done[session_id] = True
    if slot is not None:
        _booked_slot[session_id] = slot


def is_booking_done(session_id: str) -> bool:
    """Return True if an interview was already booked for this session."""
    return _booking_done.get(session_id, False)


def get_booked_slot(session_id: str) -> dict:
    """Return the exact slot that was actually booked for this session, or None."""
    return _booked_slot.get(session_id)


def get_turn_count(session_id: str) -> int:
    """Return how many Main Agent turns have already completed for this session."""
    return _turn_count.get(session_id, 0)


def increment_turn_count(session_id: str) -> int:
    """Increment and return the turn count for this session."""
    _turn_count[session_id] = _turn_count.get(session_id, 0) + 1
    return _turn_count[session_id]


def list_sessions() -> list:
    """
    Return all active session IDs.
    Useful for monitoring how many conversations are running.
    """
    return list(_sessions.keys())


# ═══════════════════════════════════════════════════════════════════
# MESSAGE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def add_user_message(session_id: str, message: str) -> None:
    """Save a candidate message to the session memory."""
    memory = get_memory(session_id)
    memory.chat_memory.add_message(HumanMessage(content=message))


def add_bot_message(session_id: str, message: str) -> None:
    """Save a bot response to the session memory."""
    memory = get_memory(session_id)
    memory.chat_memory.add_message(AIMessage(content=message))


def get_history(session_id: str) -> list:
    """
    Return the full conversation history as a list of Message objects.
    Each item is either a HumanMessage (candidate) or AIMessage (bot).
    """
    memory = get_memory(session_id)
    return memory.chat_memory.messages


def get_history_as_text(session_id: str) -> str:
    """
    Return the conversation history as plain text.
    Used by advisors that expect a text string.

    Format:
        Candidate: ...
        Bot: ...
    """
    lines = []
    for message in get_history(session_id):
        if isinstance(message, HumanMessage):
            lines.append(f"Candidate: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Bot: {message.content}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# BACKWARDS COMPATIBILITY — kept for agents.py
# ═══════════════════════════════════════════════════════════════════

def create_memory() -> ConversationBufferMemory:
    """Create a standalone memory object (used directly by AgentExecutor)."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
