<!-- PROJECT LOGO -->
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Logo" width="120" height="120">
</p>

<h1 align="center">SMS Recruitment Chatbot</h1>

<p align="center">
  A multi-agent AI chatbot for recruiting Python Developer candidates<br>
  <a href="#demo">View Demo</a>
  ·
  <a href="#about-the-project">About</a>
  ·
  <a href="#getting-started">Getting Started</a>
</p>

---
<br></br>

## Team

**Course:** Generative AI
**Project:** GenAI Recruitment Chatbot

| Name | ID |
|------|-----|
| Lior Tomer | 206795304 |
| Amit Yaron | 205560618 |
| Lior Mugami | 028698983 |

---
<br></br>


## Table of Contents

- [About The Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Code Examples](#code-examples)
- [Project Structure](#project-structure)
- [To-Do List](#to-do-list)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---
<br></br>


## About The Project

> An SMS-based recruitment chatbot that interacts with job candidates for a Python Developer position.
> The bot gathers information, answers questions, and schedules interviews — all via a Streamlit chat interface.

<div style="background: #272822; color: #f8f8f2; padding: 10px; border-radius: 8px;">
  <b>Technologies:</b> Python, LangChain, OpenAI API, ChromaDB, SQL Server, Streamlit
</div>

### How it works

The system uses a **multi-agent architecture**:

| Agent | Role |
|-------|------|
| **Main Agent** | Orchestrates the conversation and decides the next action |
| **Exit Advisor** | Detects when the conversation should end (prompt engineering) |
| **Info Advisor** | Answers candidate questions using RAG on the job description PDF |
| **Scheduling Advisor** | Finds available interview slots via function calling on SQL Server DB |

At every turn the Main Agent chooses one of three actions:
- `continue` - keep the conversation going
- `schedule` - propose interview time slots
- `end` - politely close the conversation

The Streamlit conversation now opens with a structured intake form instead of an open-ended question. The candidate fills in contact details (name, ID number, phone, email, location), experience and education, a job-description-driven skills checklist (Python stack, front-end, databases, and optional preferred skills like web frameworks, ML, cloud, open-source), a problem-solving example, and their current job search status. On submission the form is formatted into a single message and passed to the Main Agent exactly like a normal candidate reply, kicking off the rest of the conversation.

---
<br></br>


## Features

- [x] Multi-agent architecture with LangChain Agents & Tools
- [x] Structured candidate intake form (JD-driven fields) before the chat begins
- [x] RAG (Retrieval-Augmented Generation) with ChromaDB
- [x] Function calling to query SQL Server interview schedule
- [x] Conversation memory with LangChain ConversationBufferMemory
- [x] Prompt engineering — Roles, Few-Shot, Chain-of-Thought, System Prompts, API Parameters
- [x] Streamlit chat UI simulating SMS
- [x] Evaluation notebook — system-level Accuracy & Confusion Matrix on End/Continue/Schedule, plus Exit Advisor prompt comparison (V1 vs V2)
- [x] Modern Python project structure with Git

---
<br></br>


## Getting Started

### Prerequisites

- Python >= 3.10
- pip
- OpenAI API key
- SQL Server (local instance, Windows Authentication) with ODBC Driver 17 for SQL Server

### Installation

```bash
git clone https://github.com/Lior176/recruitment_chatbot.git
cd recruitment_chatbot

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here

# SQL Server (Windows Authentication) — defaults match a local default-instance install
SQL_SERVER=localhost
SQL_DATABASE=Tech
```

### One-time Setup

```bash
# 1. Create the SQL Server interview schedule database (Tech / dbo.Schedule)
python -m app.modules.database.database

# 2. Embed the job description PDF into ChromaDB
python -m app.modules.embedding.embedding
```

---
<br></br>


## Usage

### Run the Streamlit app

```bash
streamlit run streamlit_app/streamlit_main.py
```

### Run via CLI

```bash
python -m app.main
```

### Compare Prompt Versions (optional)

```bash
python -m app.modules.exit_advisor_prompt_engineering.exit_advisor_prompt_engineering
```

### Run Evaluation

Open and run `tests/test_evals.ipynb` in Jupyter. It covers:

1. **Multi-agent system evaluation** — classifies all 59 labeled turns from `sms_conversations.json` as End/Continue/Schedule and reports Accuracy + Confusion Matrix. The classifier mirrors production decision logic in 3 steps:
   - Exit Advisor check (candidate disengagement/rejection)
   - Slot-confirmation check (candidate just accepted a proposed time — mirrors `is_booking_done` in production)
   - Schedule-readiness check (DB-free — the live schedule DB only has 2026 dates, which wouldn't match this historical dataset)
2. **Exit Advisor prompt comparison** — basic prompt (V1) vs the engineered prompt (V2, live in production) on the `end` decision specifically.

Or run the system evaluation standalone:

```bash
python -m app.modules.system_evaluation.system_evaluation
```

---
<br></br>


## Screenshots

<p float="left">
  <img src="https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/e50214173218977.648c4882a75d6.gif" width="400"/>
</p>

---
<br></br>


## Code Examples

```python
from app.modules.memory.memory import create_session
from app.modules.agents.agents import run_main_agent

# Start a conversation
session_id = create_session()

result = run_main_agent(
    session_id=session_id,
    user_message="I have 5 years of Python experience.",
)

print(result["action"])    # continue / schedule / end
print(result["response"])  # bot's reply
```

> In the Streamlit app, the first `user_message` sent to `run_main_agent` is not free text — it's the candidate's intake form, formatted into a single message by `format_intake_form_as_text` (see `streamlit_app/utils.py`).

---
<br></br>


## Project Structure

```text
recruitment_chatbot/
├── .env                          # Environment variables (not in Git)
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── Python Developer Job Description.pdf
│   ├── sms_conversations.json
│   └── db_Tech.sql                # Reference T-SQL schema (dbo.Schedule) — Tech DB itself lives in SQL Server, created at runtime
├── app/
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   └── modules/
│       ├── agents/
│       │   └── agents.py         # Main Agent + 3 Advisors (LangChain)
│       ├── prompts/
│       │   └── prompts.py        # All system prompts, few-shot, API params
│       ├── memory/
│       │   └── memory.py         # ConversationBufferMemory
│       ├── embedding/
│       │   └── embedding.py      # PDF → ChromaDB pipeline
│       ├── database/
│       │   └── database.py       # SQL Server schedule DB + queries
│       ├── exit_advisor_prompt_engineering/
│       │   └── exit_advisor_prompt_engineering.py  # Exit Advisor prompt comparison (V1 vs V2)
│       └── system_evaluation/
│           └── system_evaluation.py  # Multi-agent system eval (end/continue/schedule)
├── streamlit_app/
│   ├── streamlit_main.py         # Main Streamlit chat UI
│   └── utils.py                  # UI helper functions
└── tests/
    └── test_evals.ipynb          # Accuracy & Confusion Matrix evaluation
```

---
<br></br>


## To-Do List

- [x] Project structure setup
- [x] PDF embedding with ChromaDB
- [x] SQL Server interview schedule database
- [x] Prompt engineering (Roles, Few-Shot, System Prompts)
- [x] Conversation memory
- [x] Multi-agent system (Main + Exit + Info + Scheduling)
- [x] Prompt Engineering for Exit Advisor (Chain-of-Thought + Few-Shot)
- [x] Streamlit chat interface
- [x] Evaluation notebook — system-level (End/Continue/Schedule) + Exit Advisor prompt comparison, with Accuracy + Confusion Matrix
- [ ] Cloud deployment on Streamlit Community Cloud


---
<br></br>


## Contributing

Contributions are **welcome**! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---
<br></br>


## License

Distributed under the MIT License. See `LICENSE` for more information.

---
<br></br>


## Contact

**Lior Tomer** - [lior.tomer@yazamtech.com](mailto:lior.tomer@yazamtech.com)
Project Link: [https://github.com/Lior176/recruitment_chatbot](https://github.com/Lior176/recruitment_chatbot)

---
<br></br>


## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)


---
