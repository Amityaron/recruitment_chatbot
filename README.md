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
  <b>Technologies:</b> Python, LangChain, OpenAI API, ChromaDB, SQLite, Streamlit
</div>

### How it works

The system uses a **multi-agent architecture**:

| Agent | Role |
|-------|------|
| **Main Agent** | Orchestrates the conversation and decides the next action |
| **Exit Advisor** | Detects when the conversation should end (fine-tuned) |
| **Info Advisor** | Answers candidate questions using RAG on the job description PDF |
| **Scheduling Advisor** | Finds available interview slots via function calling on SQLite DB |

At every turn the Main Agent chooses one of three actions:
- `continue` - keep the conversation going
- `schedule` - propose interview time slots
- `end` - politely close the conversation

---
<br></br>


## Features

- [x] Multi-agent architecture with LangChain Agents & Tools
- [x] RAG (Retrieval-Augmented Generation) with ChromaDB
- [x] Function calling to query SQLite interview schedule
- [x] Conversation memory with LangChain ConversationBufferMemory
- [x] Prompt engineering — Roles, Few-Shot, System Prompts, API Parameters
- [x] Fine-tuning of Exit Advisor on labeled SMS conversations
- [x] Streamlit chat UI simulating SMS
- [x] Evaluation notebook — Accuracy & Confusion Matrix before/after fine-tuning
- [x] Modern Python project structure with Git

---
<br></br>


## Getting Started

### Prerequisites

- Python >= 3.10
- pip
- OpenAI API key

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
```

After fine-tuning, add:
```
FINETUNED_EXIT_MODEL=ft:gpt-4o-mini:...
```

### One-time Setup

```bash
# 1. Create the SQLite interview schedule database
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

### Run Fine-Tuning (optional)

```bash
python -m app.modules.finetuning.finetuning
```

### Run Evaluation

Open and run `tests/test_evals.ipynb` in Jupyter.

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
│   ├── db_Tech.sql
│   └── tech_schedule.db          # Created at runtime
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
│       │   └── database.py       # SQLite schedule DB + queries
│       └── finetuning/
│           └── finetuning.py     # Exit Advisor fine-tuning pipeline
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
- [x] SQLite interview schedule database
- [x] Prompt engineering (Roles, Few-Shot, System Prompts)
- [x] Conversation memory
- [x] Multi-agent system (Main + Exit + Info + Scheduling)
- [x] Fine-tuning pipeline for Exit Advisor
- [x] Streamlit chat interface
- [x] Evaluation notebook (Accuracy + Confusion Matrix)
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

**Amit Yaron** - [sys332@gmail.com](mailto:sys332@gmail.com)  
Project Link: [https://github.com/Amityaron/recruitment_chatbot](https://github.com/Amityaron/recruitment_chatbot)

---
<br></br>


## Acknowledgments

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/overview)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)


---
