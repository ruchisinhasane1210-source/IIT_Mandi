# TraceLearn

**A multimodal AI study companion that cites its sources.**

TraceLearn ingests your course material (lecture videos, audio, PDFs, slide decks and images) and breaks it into small, citable units. Each unit keeps a link to its exact origin: a page number, a slide number or a video timestamp. It uses those units to answer questions with verifiable citations, generate quizzes, and track a prerequisite-aware model of what you actually know.

Built for the **Multimodal AI Hackathon 2026, Track D (Personalized Tutoring & Adaptive Learning)**.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [API Overview](#api-overview)
- [Running Tests](#running-tests)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

**Multimodal ingestion**
- Accepts PDF, PPTX, plain text/Markdown/CSV, MP4/MOV/WebM/MKV video, MP3/WAV/M4A audio, and PNG/JPG images.
- Runs a staged background pipeline: extract, chunk and validate, caption figures, embed, then tag topics and build the prerequisite graph. Progress can be polled through the API.
- Transcribes audio and video with local Whisper (optional), the Gemini audio API, or a placeholder fallback so ingestion never blocks.
- Captions figures and diagrams with Gemini vision when a key is available.

**Grounded tutor with citations**
- Uses hybrid retrieval: vector similarity plus BM25 keyword search, fused with reciprocal rank fusion.
- Labels every answer `grounded`, `partial` or `not covered`. The label is computed from retrieval scores, not self-reported by the model.
- Refuses off-material questions instead of guessing.
- Links each citation to its source: open slide 12, jump to page 84, or play from 12:42.
- Includes a verification step that removes citations pointing to units that don't exist.
- Supports follow-up questions with session history.

**Knowledge graph and learner model**
- Builds an automatic concept and prerequisite graph with DAG repair, rendered as a layered SVG.
- Tracks mastery per topic with Bayesian Knowledge Tracing (BKT).
- Recommends the weakest prerequisite first ("you are missing X, which Y depends on") and generates a study schedule.

**Adaptive assessment**
- Selects questions that target your weak topics, with difficulty tuned to roughly 70% predicted success.
- Offers multiple-choice, short-answer and numerical question types, plus a 10-question diagnostic.
- Produces cited result reports and logs misconceptions.

**Built-in evaluation**
- Measures retrieval hit@5, MRR, citation accuracy, and refusal precision, recall and F1.
- Compares adaptive and random practice policies through a simulated-student run.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask, Jinja2 |
| Frontend | Server-rendered HTML templates, hand-written CSS design system (light/dark themes), vanilla JavaScript |
| Database | SQLite by default; Supabase Postgres + pgvector when `DATABASE_URL` is set |
| Storage | Local `./storage` folder, or Supabase Storage |
| AI models | Google Gemini (text, vision, audio/video, embeddings); Groq (optional fast text fallback) |
| Embeddings | Gemini `text-embedding-004`, or a built-in hashing vectorizer (no key needed) |
| Ingestion | PyMuPDF, pypdf, python-pptx, rank-bm25, pydantic |
| Transcription (optional) | openai-whisper (local), yt-dlp (for remote media URLs) |
| Testing | pytest |
| Production server (optional) | gunicorn |

---

## Project Structure

```
.
├── run.py                     # Development entry point (serves on http://127.0.0.1:5000)
├── requirements.txt           # Python dependencies
├── schema.sql                 # Canonical Postgres/pgvector schema (auto-converted for SQLite)
├── .env.example               # Template for environment variables
├── start.bat / stop.bat       # Windows helpers to start/stop the dev server
│
├── tracelearn/                # Core application package
│   ├── __init__.py            # App factory (create_app) and blueprint registration
│   ├── config.py              # Environment settings and demo-mode detection
│   ├── db.py                  # Dual SQLite/Postgres adapter and query helpers
│   ├── llm.py                 # Gemini + Groq clients with caching and demo fallback
│   ├── embeddings.py          # Gemini embeddings + local hashing vectorizer
│   ├── ingest.py              # Upload → extract → chunk → embed → citable units
│   ├── whisper_transcribe.py  # Audio/video transcription (Whisper → Gemini → stub)
│   ├── vision.py              # Figure/diagram captioning
│   ├── rag.py                 # Hybrid retrieval, grounding status, cited answers
│   ├── graph.py               # Concept graph, DAG repair, SVG layout
│   ├── learner.py             # BKT learner model, recommendations, schedule
│   ├── assessment.py          # Adaptive quiz generation and grading
│   ├── evaluation.py          # Retrieval, refusal and simulated-student metrics
│   ├── seed.py                # Demo course data (MIT 6.006)
│   └── routes/
│       ├── pages.py           # Server-rendered pages
│       ├── api.py             # Core JSON API (/api/...)
│       └── extended_api.py    # Ingestion, knowledge graph and tutor endpoints
│
├── backend/                   # Extended pipelines
│   ├── ingestion/             # Per-modality ingestors (pdf, pptx, audio, video),
│   │                          # topic tagger, and the staged ingestion orchestrator
│   ├── tutor/                 # Query understanding, retriever, support check,
│   │                          # grounded generator, verifier, pipeline
│   └── learner/               # Learner-model hook interface
│
├── templates/                 # Jinja2 pages (dashboard, upload, chat, graph, etc.)
├── static/
│   ├── css/style.css          # Design system
│   └── js/chat.js             # Tutor chat client
│
├── data/sources.json          # Source manifest with licenses and attribution
├── eval/test_set.json         # Hand-built grounded / partial / unsupported questions
├── docs/FRONTEND_HOOKS.md     # Extended API reference and deep-link scheme
├── scripts/seed_demo_course.py# End-to-end ingestion demo using sample fixtures
├── tests/                     # pytest suite + generated fixtures (PDF, PPTX, WAV)
│
└── Multi-modal-master/        # Archived copy of an earlier version (also as .zip);
                               # not used by the current app
```

---

## Prerequisites

- **Python 3.10 or newer.** The code uses modern type-hint syntax such as `str | None`.
- **pip** and **venv**, which ship with most Python installs.
- **Git**, to clone the repository.
- **API** , for database supabase key , for AI features Groq + Gemini key.
- *(Optional)* **FFmpeg**, for broader audio/video handling and local Whisper transcription.
- *(Optional)* API keys for Gemini, Groq and Supabase. See [Configuration](#configuration).

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ruchisinhasane1210-source/IIT_Mandi.git
   cd IIT_Mandi
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. *(Optional)* **Create your environment file**

   ```bash
   # Windows
   copy .env.example .env

   # macOS / Linux
   cp .env.example .env
   ```

   You can leave every value empty to run in demo mode.

---

## Running the App

**Any platform:**

```bash
python run.py
```

Then open **http://127.0.0.1:5000** in your browser.

On startup the terminal prints a status banner showing the current mode (`DEMO` or `LIVE`), the database, storage, LLM and embedding provider in use.

**Windows shortcut:**

- Double-click `start.bat` to activate `venv` (or `.venv`) and launch the server in a new window.
- Double-click `stop.bat` to free port 5000 when you're done.

**Custom port:**

```bash
# macOS / Linux
PORT=8000 python run.py

# Windows (PowerShell)
$env:PORT=8000; python run.py
```

**Production (optional):**

```bash
gunicorn "tracelearn:create_app()" --bind 0.0.0.0:8000
```

---

## Configuration

All settings are read from environment variables, or from a `.env` file in the project root. **Every setting is optional and independent.** Setting `DATABASE_URL` alone upgrades the database; setting `GEMINI_API_KEY` alone upgrades the models.

| Variable | Purpose | Default |
|---|---|---|
| `SESSION_SECRET_KEY` | Flask session signing (set this in production) | Development key |
| `DATABASE_URL` | Supabase Postgres connection string; enables pgvector | Local SQLite (`tracelearn.db`) |
| `SUPABASE_URL` | Supabase project URL (for Storage) | |
| `SUPABASE_SECRET_KEY` | Supabase service-role key | |
| `SUPABASE_STORAGE_BUCKET` | Storage bucket for uploaded media | `tracelearn` |
| `GEMINI_API_KEY` | LLM answers, vision captions, audio/video transcription, embeddings | |
| `GEMINI_CHAT_MODEL` | Gemini chat model | `gemini-2.0-flash` |
| `GEMINI_EMBED_MODEL` | Gemini embedding model | `text-embedding-004` |
| `EMBED_DIM` | Embedding dimension | `768` |
| `GROQ_API_KEY` | Optional fast text fallback | |
| `GROQ_MODEL` | Groq model | `llama-3.3-70b-versatile` |
| `WHISPER_MODEL` | Enables local Whisper transcription (e.g. `base`); requires `openai-whisper` | |
| `T_HIGH` / `T_LOW` | Grounding thresholds for `grounded` / `not covered` | `0.36` / `0.28` |
| `DEMO_MODE` | Set to `1` to force demo mode even when keys are present | |
| `PORT` | Port for `run.py` | `5000` |

**Where to get keys:**
- Gemini: https://aistudio.google.com/apikey (free tier available)
- Groq: https://console.groq.com/keys
- Supabase: Project settings → Database (connection string) and Project settings → API (URL and keys)

> **Note:** The grounding thresholds are calibrated for the built-in hashing vectorizer. If you switch to real Gemini embeddings, re-tune `T_HIGH` and `T_LOW` using the Evaluation page.

### Demo mode vs. live mode

| Capability | Demo mode (no keys) | With `GEMINI_API_KEY` |
|---|---|---|
| Seeded course, dashboard, graph, mastery | ✅ | ✅ |
| Hybrid retrieval, citations, refusal | ✅ | ✅ (with real embeddings) |
| BKT learner model and recommendations | ✅ | ✅ |
| Assessment grading and evaluation suite | ✅ | ✅ |
| Generating new quiz questions | Seeded question bank only | ✅ |
| Figure/diagram captioning | Placeholder | ✅ |
| Audio/video transcription | Placeholder (or local Whisper) | ✅ |

---

## Usage Guide

1. **Dashboard (`/`).** See overall mastery, topics tracked, citable units, verified questions, and the "Fix the foundation" recommendation.
2. **Sources (`/upload`).** Drag in lectures, textbooks, slides, audio or images. Each file is processed in the background into citable units.
3. **Tutor (`/chat`).** Ask a question about your course. Each answer shows its grounding status, a score, and citation buttons. Clicking a citation opens the exact source location in the preview panel. Try the built-in prompts, including an off-material question to see a refusal.
4. **Concept Map (`/graph`).** Explore the prerequisite graph, colored by your current mastery.
5. **Practice (`/assessment`).** Generate an adaptive quiz or run a 10-question diagnostic. Results appear on `/results`.
6. **Mastery (`/mastery`).** Review per-topic BKT estimates, the recommended next topic, and a study schedule.
7. **Evaluation (`/eval`).** Run the evaluation suite and view retrieval, citation and refusal metrics live.

### Seed a sample course end to end

To exercise the full ingestion pipeline on small generated fixtures (a 30-second WAV, a 3-slide PPTX and a 3-page PDF), run:

```bash
python scripts/seed_demo_course.py
```

---

## API Overview

Core endpoints are served under `/api`. Extended endpoints are available both at the root and under `/api`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health and configuration summary |
| `POST` | `/api/courses/<course_id>/chat` | Ask the tutor a question (original contract) |
| `POST` | `/chat` | Source-grounded tutor pipeline with sessions |
| `GET` | `/chat/session/<session_id>` | Tutor conversation history |
| `POST` | `/ingest/upload` | Upload a file (multipart) or submit a URL (JSON) for ingestion |
| `GET` | `/ingest/status/<job_id>` | Ingestion job progress |
| `GET` / `DELETE` | `/sources`, `/sources/<source_id>` | List or remove sources |
| `GET` | `/source/<unit_id>`, `/citation/<unit_id>` | Exact citation text and location |
| `GET` | `/knowledge/graph` | Topic and prerequisite graph |
| `POST` | `/retrieve/debug` | Inspect raw retrieval results |
| `GET` | `/api/students/<user_id>/mastery` | Per-topic mastery |
| `GET` | `/api/students/<user_id>/recommendations` | Next-topic recommendation |
| `POST` | `/api/courses/<course_id>/assessments/generate` | Generate an adaptive quiz |
| `POST` | `/api/evaluation/run` | Run the evaluation suite |

See [`docs/FRONTEND_HOOKS.md`](docs/FRONTEND_HOOKS.md) for request/response schemas and the deep-link format (`?page=`, `?slide=`, `?t=`).

---

## Running Tests

```bash
pytest -q
```

The suite covers the PDF, PPTX and audio ingestors, tagging coverage, prerequisite-graph acyclicity, ingestion idempotency, retrieval recall and MRR, tutor behavior on in-scope, partial, off-material and figure questions, citation verification, follow-up resolution, and API backward compatibility. Test fixtures are generated automatically by `tests/generate_fixtures.py`.

> At the time of writing, two tutor tests (`test_15_in_scope_questions` and `test_10_off_material_questions`) can fail in demo mode because their expectations depend on the grounding thresholds. See [Troubleshooting](#troubleshooting).

---

## Screenshots

*Screenshots are not yet included in the repository. Suggested additions:*

| Page | Suggested file |
|---|---|
| Dashboard | `docs/screenshots/dashboard.png` |
| Tutor with citations | `docs/screenshots/tutor.png` |
| Concept map | `docs/screenshots/concept-map.png` |
| Adaptive practice | `docs/screenshots/practice.png` |
| Evaluation | `docs/screenshots/evaluation.png` |

```markdown
![Tutor with citations](docs/screenshots/tutor.png)
```

---

## Troubleshooting

**`ModuleNotFoundError` (e.g. `fitz`, `pptx`, `rank_bm25`, `pydantic`)**
Make sure your virtual environment is activated, then reinstall the dependencies with `pip install -r requirements.txt`.

**`psycopg2-binary` fails to install**
This package is only needed for Supabase Postgres. Upgrade pip first (`python -m pip install --upgrade pip`). If it still fails, temporarily comment it out of `requirements.txt`; the app falls back to SQLite.

**Syntax errors such as `unsupported operand type(s) for |`**
You're on Python 3.9 or older. Install Python 3.10+ and recreate the virtual environment.

**Port 5000 is already in use**
Run `stop.bat` on Windows, or start on another port with `PORT=8000 python run.py`. On macOS, port 5000 may be taken by AirPlay Receiver; either disable it in System Settings or use a different port.

**The app always says "DEMO" mode**
Check that `.env` is in the project root, that `GEMINI_API_KEY` is filled in, and that `DEMO_MODE` is empty (not `1`).

**Transcripts or figure captions show placeholder text**
This is expected without `GEMINI_API_KEY`. For local transcription, install `openai-whisper` (plus FFmpeg) and set `WHISPER_MODEL=base`.

**Answers are refused or marked "partial" more often than expected**
The grounding thresholds (`T_HIGH`, `T_LOW`) depend on the embedding provider. Re-tune them after switching providers using the Evaluation page.

**I want to reset the demo data**
Stop the server, delete `tracelearn.db` from the project root, and start again. The demo course is re-seeded on boot.

**`run_preview.ps1` doesn't work**
This file currently contains terminal output rather than a working script. Use `python run.py` or `start.bat` instead.

---

## Contributing

Contributions are welcome. To contribute:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Set up** the project as described in [Installation](#installation).
3. **Make your changes**, keeping to the existing code style (type hints, small focused modules, docstrings).
4. **Add or update tests** in `tests/` and make sure `pytest -q` passes.
5. **Never commit secrets.** `.env` is git-ignored; use `.env.example` to document new variables.
6. **Commit** with a clear message and **open a pull request** describing what changed and why.

For bugs or feature ideas, please open an issue with steps to reproduce, your Python version, and whether you're running in demo or live mode.

---

## License

This project does not currently specify a license.

> The `LICENSE.txt` file in the repository contains the Python Software Foundation license text for the Python interpreter, which appears to have been included with a bundled Python runtime. It does not license this project's source code. If you intend to open-source TraceLearn, consider adding a project license such as MIT or Apache-2.0.

---

## Acknowledgements

The demo course uses short derived excerpts from openly licensed materials (see `data/sources.json`):

- **MIT OpenCourseWare, 6.006 Introduction to Algorithms** ([CC BY-NC-SA 4.0](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/))
- **Open Data Structures** by Pat Morin ([CC BY 2.5 CA](https://opendatastructures.org/))

No copyrighted textbook or video content is redistributed in this repository.
