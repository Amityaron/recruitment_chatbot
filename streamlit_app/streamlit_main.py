"""
Streamlit Main App
------------------
Chat interface for the recruitment chatbot.
Simulates an SMS conversation between a recruiter bot and a candidate.

Run:
  streamlit run streamlit_app/streamlit_main.py
"""

import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime, timezone
import streamlit as st

from app.modules.database.database import create_and_seed_db
from app.modules.embedding.embedding import run_embedding, CHROMA_DIR
from app.modules.memory.memory import (
    create_session, get_memory, add_bot_message, end_session,
    get_history_as_text, get_offered_slots, is_booking_done,
)
from app.modules.agents.agents import run_main_agent
from app.modules.prompts.prompts import JOB_POSITION
from streamlit_app.utils import (
    get_action_badge, format_intake_form_as_text, format_slot_option,
    is_valid_name_or_location, is_valid_email, is_valid_israeli_id, is_valid_israeli_phone,
    has_none_conflict, is_valid_linkedin_url,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recruitment Chatbot",
    page_icon="💼",
    layout="wide",
)

# ── One-time setup ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Setting up database and knowledge base...")
def setup():
    """Create DB and Chroma vector store if they don't exist yet."""
    create_and_seed_db()
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        run_embedding()
    return True


setup()

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id     = create_session()
    st.session_state.timestamp      = datetime.now(timezone.utc).isoformat()
    st.session_state.messages       = []   # list of {"role", "text", "action"}
    st.session_state.finished       = False
    st.session_state.form_submitted = False

    # Opening message
    opening = (
        f"Hi! Thanks for your interest in our {JOB_POSITION} opening. "
        "Please fill out the short form below so I can learn about your background."
    )
    add_bot_message(st.session_state.session_id, opening)
    st.session_state.messages.append({
        "role":   "bot",
        "text":   opening,
        "action": "continue",
    })


# ── Shared helper: send a message and record the agent's reply ────────────────
def send_message(user_text: str, spinner_text: str = "Thinking..."):
    st.session_state.messages.append({"role": "user", "text": user_text})
    with st.spinner(spinner_text):
        result = run_main_agent(
            session_id=st.session_state.session_id,
            user_message=user_text,
            conversation_timestamp=st.session_state.timestamp,
        )
    st.session_state.messages.append({
        "role":   "bot",
        "text":   result["response"],
        "action": result["action"],
    })
    if result["action"] == "end":
        st.session_state.finished = True
    return result


# ── Debug Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Debug Panel")

    st.subheader("Last Action")
    last_action = "—"
    if st.session_state.get("messages"):
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "bot":
                last_action = get_action_badge(msg.get("action", "—"))
                break
    st.write(last_action)

    st.subheader("Chat History (Memory)")
    if "session_id" in st.session_state:
        history = get_history_as_text(st.session_state.session_id)
        st.text_area("History", value=history or "(empty)", height=300, disabled=True, label_visibility="collapsed")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💼 Recruitment Chatbot")
st.caption(f"{JOB_POSITION} Position — Powered by LangChain & OpenAI")
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "bot":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["text"])
            if "action" in msg:
                st.caption(get_action_badge(msg["action"]))

            is_latest = idx == len(st.session_state.messages) - 1
            if msg.get("action") == "schedule" and is_latest and not is_booking_done(st.session_state.session_id):
                offered_slots = get_offered_slots(st.session_state.session_id)
                if offered_slots:
                    options = [format_slot_option(slot) for slot in offered_slots]
                    choice = st.selectbox("Pick an interview slot", options, key=f"slot_select_{idx}")
                    if st.button("Confirm a slot", key=f"slot_confirm_{idx}"):
                        send_message(str(options.index(choice) + 1), spinner_text="Booking...")
                        st.rerun()
                    st.caption("Or type a slot number, a different day, or any other request in the message box below.")
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["text"])

# ── Intake form ───────────────────────────────────────────────────────────────
if not st.session_state.form_submitted:
    with st.form("intake_form"):
        st.markdown(f"#### Tell us about your background — {JOB_POSITION} role")

        st.markdown("**Required**")
        name       = st.text_input("Full name", help="Letters, spaces, hyphens, and apostrophes only, e.g. \"John O'Brien\"")
        id_number  = st.text_input("ID Number", help="Exactly 9 digits, e.g. 123456789")
        phone      = st.text_input("Phone Number", help="Israeli format: 05X-XXXXXXX, 05XXXXXXXX, or +972-5X-XXXXXXX")
        email      = st.text_input("Email Address", help="Standard email format, e.g. name@example.com")
        location   = st.text_input("City / Location", help="Letters, spaces, and hyphens only, e.g. \"Tel Aviv\"")
        years_experience = st.number_input(
            f"Years of experience as a {JOB_POSITION}",
            min_value=0, max_value=50, step=1,
        )
        education = st.selectbox(
            "Highest relevant education",
            [
                "Bachelor's in CS/Software Engineering",
                "Bachelor's in a related field",
                "Master's or higher",
                "Other / self-taught",
                "No degree",
            ],
        )
        python_stack = st.multiselect(
            "Which of these Python tools/frameworks have you used?",
            ["None","NumPy", "SciPy", "Pandas", "Dask", "spaCy", "NLTK", "scikit-learn", "PyTorch"],
        )
        frontend_skills = st.multiselect(
            "Front-end technologies you're comfortable with",
            ["None","HTML", "CSS", "JavaScript"],
        )
        database_skills = st.multiselect(
            "Database technologies you've worked with",
            ["None","SQL", "NoSQL"],
        )
        problem_solving_summary = st.text_area(
            "Briefly describe a challenging problem you solved",
        )
        job_search_status = st.selectbox(
            "What's your current job search status?",
            [
                "Actively looking",
                "Not actively looking, but open to opportunities",
                "Looking for a different type of job",
                "In process with other companies",
                "Already accepted another offer",
                "Currently employed, not looking",
            ],
        )

        st.markdown("**Optional**")
        linkedin_url = st.text_input(
            "LinkedIn profile URL (optional)",
            help="e.g. linkedin.com/in/johndoe or https://www.linkedin.com/in/johndoe",
        )
        web_frameworks = st.multiselect(
            "Python web frameworks you've used (optional)",
            ["Django", "Flask", "Pyramid"],
        )
        ml_experience = st.checkbox("I have experience with data science / ML concepts and tools")
        cloud_platforms = st.multiselect(
            "Cloud platforms you've worked with (optional)",
            ["AWS", "Google Cloud", "Azure"],
        )
        open_source = st.checkbox("I contribute to open-source or am active in the Python community")
        additional_notes = st.text_area("Anything else you'd like to add? (optional)")

        submitted = st.form_submit_button("Submit Application Info")

    if submitted:
        missing = []
        invalid = []
        if not name.strip():
            missing.append("Full name")
        elif not is_valid_name_or_location(name):
            invalid.append("Full name (letters only)")
        if not id_number.strip():
            missing.append("ID Number")
        elif not is_valid_israeli_id(id_number):
            invalid.append("ID Number (must be exactly 9 digits)")
        if not phone.strip():
            missing.append("Phone Number")
        elif not is_valid_israeli_phone(phone):
            invalid.append("Phone Number (e.g. 050-1234567 or +972-50-1234567)")
        if not email.strip():
            missing.append("Email Address")
        elif not is_valid_email(email):
            invalid.append("Email Address (e.g. name@example.com)")
        if not location.strip():
            missing.append("City / Location")
        elif not is_valid_name_or_location(location):
            invalid.append("City / Location (letters only)")
        if not python_stack:
            missing.append("Python tools/frameworks")
        elif has_none_conflict(python_stack):
            invalid.append("Python tools/frameworks (cannot select None with other options)")
        if not frontend_skills:
            missing.append("Front-end technologies")
        elif has_none_conflict(frontend_skills):
            invalid.append("Front-end technologies (cannot select None with other options)")
        if not database_skills:
            missing.append("Database technologies")
        elif has_none_conflict(database_skills):
            invalid.append("Database technologies (cannot select None with other options)")
        if not problem_solving_summary.strip():
            missing.append("Problem-solving description")
        if linkedin_url.strip() and not is_valid_linkedin_url(linkedin_url):
            invalid.append("LinkedIn profile URL (must look like linkedin.com/in/username)")

        if missing or invalid:
            if missing:
                st.warning(f"Please fill out the required fields: {', '.join(missing)}")
            if invalid:
                st.warning(f"Please fix the following fields: {', '.join(invalid)}")
        else:
            form_data = {
                "name":                    name,
                "id_number":               id_number,
                "phone":                   phone,
                "email":                   email,
                "location":                location,
                "years_experience":        years_experience,
                "education":               education,
                "python_stack":            python_stack,
                "frontend_skills":         frontend_skills,
                "database_skills":         database_skills,
                "problem_solving_summary": problem_solving_summary,
                "job_search_status":       job_search_status,
                "linkedin_url":            linkedin_url,
                "web_frameworks":          web_frameworks,
                "ml_experience":           ml_experience,
                "cloud_platforms":         cloud_platforms,
                "open_source":             open_source,
                "additional_notes":        additional_notes,
            }
            form_text = format_intake_form_as_text(form_data)

            send_message(form_text)

            st.session_state.form_submitted = True

            st.rerun()

# ── Conversation ended ────────────────────────────────────────────────────────
if st.session_state.finished:
    st.info("The conversation has ended. Refresh the page to start a new one.")
    if st.button("🔄 Start New Conversation"):
        end_session(st.session_state.session_id)
        for key in ["session_id", "timestamp", "messages", "finished", "form_submitted"]:
            del st.session_state[key]
        st.rerun()
    st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
if st.session_state.form_submitted:
    user_input = st.chat_input("Type your message...")

    if user_input:
        send_message(user_input)
        st.rerun()
