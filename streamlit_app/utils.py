"""
Streamlit Utilities
-------------------
Helper functions for formatting and displaying chat messages.
"""

import re
from datetime import datetime

_NAME_RE     = re.compile(r"^[A-Za-z֐-׿\s'-]+$")
_EMAIL_RE    = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ID_RE       = re.compile(r"^\d{9}$")
_PHONE_RE    = re.compile(r"^(05\d{8}|\+9725\d{8})$")
_LINKEDIN_RE = re.compile(r"^(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?$", re.IGNORECASE)


def is_valid_name_or_location(value: str) -> bool:
    """Letters (incl. Hebrew), spaces, hyphens, apostrophes only."""
    return bool(_NAME_RE.fullmatch(value.strip()))


def is_valid_email(value: str) -> bool:
    """Basic local@domain.tld shape."""
    return bool(_EMAIL_RE.fullmatch(value.strip()))


def is_valid_israeli_id(value: str) -> bool:
    """Exactly 9 digits."""
    return bool(_ID_RE.fullmatch(value.strip()))


def is_valid_israeli_phone(value: str) -> bool:
    """05X-XXXXXXX / 05XXXXXXXX / +972-5X-XXXXXXX, hyphens/spaces optional."""
    normalized = re.sub(r"[\s-]", "", value.strip())
    return bool(_PHONE_RE.fullmatch(normalized))


def has_none_conflict(selected: list) -> bool:
    """True if 'None' is selected together with at least one other option."""
    return "None" in selected and len(selected) > 1


def is_valid_linkedin_url(value: str) -> bool:
    """(https?://)?(www.)?linkedin.com/in/<username>, protocol/www optional."""
    return bool(_LINKEDIN_RE.fullmatch(value.strip()))


def format_timestamp(iso_str: str) -> str:
    """
    Convert an ISO timestamp to a readable time string.
    e.g. "2024-03-15T10:30:00" -> "10:30"
    """
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M")
    except Exception:
        return ""


def format_slot_option(slot: dict) -> str:
    """
    Format a single slot dict into a readable label.
    e.g. {"date": "2024-03-19", "time": "10:00"} -> "Tuesday, Mar 19 at 10:00"
    """
    try:
        dt  = datetime.strptime(slot["date"], "%Y-%m-%d")
        day = dt.strftime("%A, %b %d")
        return f"{day} at {slot['time']}"
    except Exception:
        return f"{slot['date']} at {slot['time']}"


def format_slots_as_text(slots: list) -> str:
    """
    Format a list of slot dicts into a readable, numbered string.

    Input:  [{"date": "2024-03-19", "time": "10:00", "ScheduleID": 5}]
    Output: "1. Tuesday, Mar 19 at 10:00"
    """
    if not slots:
        return "No available slots found."

    return "\n".join(
        f"{i}. {format_slot_option(slot)}" for i, slot in enumerate(slots, start=1)
    )


def format_intake_form_as_text(form_data: dict) -> str:
    """
    Format submitted intake-form data into a single readable message,
    written in first person as if the candidate typed it themselves.

    Input: dict with keys matching the intake form fields in streamlit_main.py
    Output: multi-line string suitable as a candidate chat message.
    """
    labels = [
        ("name", "Full name"),
        ("id_number", "ID number"),
        ("phone", "Phone number"),
        ("email", "Email address"),
        ("location", "Location"),
        ("years_experience", "Years of experience"),
        ("education", "Education"),
        ("python_stack", "Python tools/frameworks"),
        ("frontend_skills", "Front-end skills"),
        ("database_skills", "Database skills"),
        ("problem_solving_summary", "Problem-solving example"),
        ("job_search_status", "Job search status"),
        ("linkedin_url", "LinkedIn profile"),
        ("web_frameworks", "Python web frameworks"),
        ("ml_experience", "Data science / ML experience"),
        ("cloud_platforms", "Cloud platforms"),
        ("open_source", "Open-source involvement"),
        ("additional_notes", "Additional notes"),
    ]

    lines = ["Here's a bit about my background:"]
    for key, label in labels:
        value = form_data.get(key)

        if isinstance(value, bool):
            value = "Yes" if value else "No"
        elif isinstance(value, list):
            value = ", ".join(value)

        if not value and value != 0:
            continue

        lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def get_action_badge(action: str) -> str:
    """
    Return a colored badge string for a given action label.
    Used to show the agent's decision in the UI.
    """
    badges = {
        "continue": "🟢 Continue",
        "schedule": "📅 Schedule",
        "end":      "🔴 End",
    }
    return badges.get(action, action)
