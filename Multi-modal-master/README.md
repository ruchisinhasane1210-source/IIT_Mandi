# TraceLearn

A multimodal AI study companion: it ingests your lecture videos, textbooks and slide
decks, organises them into a prerequisite graph where **every content unit links back to
its exact origin** (page, slide or video timestamp), and uses that to answer questions
with citations, generate verified quizzes, and track a prerequisite-aware model of what
you actually know.

Built for the Multimodal AI Hackathon 2026, Track D (Personalized Tutoring & Adaptive
Learning).

---

## It runs right now, with zero keys

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:5000>. On Windows you can also just double-click `start.bat`
(and `stop.bat` to shut it down).

**No API keys are required to see the whole product.** With none configured, TraceLearn
boots in **demo mode**: it seeds a complete MIT 6.006 (DSA) course — 8 topics, a
prerequisite graph, 12 content units, 12 verified questions, mastery records and a study
plan — and serves it from a local SQLite file. Retrieval, refusal, the BKT learner model,
assessment grading and the evaluation suite are all **fully functional**, because none of
them depend on an LLM.

The only thing demo mode can't do is *generate new* questions or caption figures from
scratch. Add a key and those switch on.

---

## Keys and services — what to give me, and when

| Variable | Needed for | Needed when | Where to get it |
|---|---|---|---|
| `GEMINI_API_KEY` | LLM answers, **vision** (figures/diagrams), **video + audio transcription**, embeddings | Figure captioning, live question generation, real RAGAS | <https://aistudio.google.com/apikey> — free tier |
| `DATABASE_URL` | Supabase Postgres + pgvector instead of local SQLite | Deploying, or sharing one dataset across machines | Supabase → Project settings → Database → connection string |
| `SUPABASE_URL` + `SUPABASE_SECRET_KEY` | Supabase Storage for uploaded media | Same as above | Supabase → Project settings → API |
| `GROQ_API_KEY` | Optional very fast text fallback | Only if you want lower latency without Gemini | <https://console.groq.com/keys> |
| `SESSION_SECRET_KEY` | Flask session signing | Production only | Any random string |

Copy `.env.example` to `.env` and fill in only what you have — **everything is optional
and independent.** Setting `DATABASE_URL` alone upgrades the database; setting
`GEMINI_API_KEY` alone upgrades the models. `DEMO_MODE=1` forces demo mode even with keys.

---

## Stack

Mirrors the pattern of a polished Flask app I was asked to follow:

| Layer | Choice | Why |
|---|---|---|
| Web | **Flask + Jinja2** | One service, server-rendered, trivial to run and deploy |
| Styling | **Hand-written CSS design system** (`static/css/style.css`) | Zero build step, light/dark themes, no CDN or Tailwind build to break |
| Database | **SQLite by default, Supabase Postgres + pgvector when `DATABASE_URL` is set** | Zero-config locally, production-ready on deploy |
| Storage | Local `./storage`, or **Supabase Storage** | Same code path either way |
| Models | **Gemini** primary (text + vision + video/audio + embeddings), **Groq** optional | One key covers every modality |
| Embeddings | Gemini `text-embedding-004`, or a built-in **hashing vectorizer** | Retrieval genuinely works with no key |

### The two-backend trick

`tracelearn/db.py` exposes one query API. Write `?` placeholders once; they are
translated to `%s` for Postgres automatically. `schema.sql` is the single source of
truth and is **down-converted to SQLite at boot** (`VECTOR(768) → TEXT`, pgvector-only
indexes stripped). So there is no migration fork and no dialect branching in route code.

---

## Architecture

```
run.py                     entry point + startup banner
schema.sql                 canonical Postgres/pgvector DDL (auto-downgraded for SQLite)

tracelearn/
  config.py                all env vars, sane defaults, demo-mode detection
  db.py                    dual SQLite/Postgres adapter, query helpers, boot
  llm.py                   Gemini + Groq clients, content-hash caching, demo fallback
  embeddings.py            Gemini embeddings + local hashing vectorizer, cosine
  ingest.py                upload -> extract -> chunk -> embed -> citable units
  rag.py                   hybrid retrieval, RRF fusion, refusal, grounded answering
  graph.py                 concept graph build, DAG repair, layered SVG layout
  learner.py               Bayesian Knowledge Tracing, prereq policy, schedule
  assessment.py            adaptive generation, grading, cited reports
  evaluation.py            retrieval / refusal / simulated-student metrics
  seed.py                  the demo course
  routes/pages.py          server-rendered pages
  routes/api.py            JSON API

templates/                 Jinja pages + a shared inline-SVG icon macro
static/css/style.css       the design system
static/js/chat.js          tutor chat client
eval/test_set.json         hand-built grounded / partial / unsupported questions
data/sources.json          source manifest with license + attribution
```

### Retrieval and refusal (§9)

1. Embed the question. 2. Vector search over content units. 3. BM25 keyword search for
exact terms and equations. 4. Fuse both ranked lists with reciprocal rank fusion.
5. Score each hit by saturating similarity + keyword strength, then map the top score to
`grounded` / `partial` / `not_covered`. 6. Answer only from kept evidence, with citation
validation that drops any citation whose unit isn't real.

**Grounding status is computed from retrieval scores, never self-reported by the model.**

The thresholds `T_HIGH`/`T_LOW` are **tuned on `eval/test_set.json`** and stored in
config, exactly as the plan requires.

### Learner model (§12)

Bayesian Knowledge Tracing per topic — a wrong answer lowers mastery, a correct one
raises it, and the posterior always stays in [0, 1]. Failures look up the prerequisite
graph and recommend the **weakest prerequisite first** ("you are missing X, which Y
depends on"), rather than grinding on the topic you're already stuck on.

---

## Measured results

Run `python -c "from tracelearn import create_app; app=create_app(); app.test_client().post('/api/evaluation/run')"`,
or just open the **Evaluation** tab. Current numbers on the seeded corpus, with **no
model key**:

| Metric | Value |
|---|---|
| Retrieval hit@5 | **1.000** |
| Retrieval MRR | **0.917** |
| Citation accuracy | **0.917** |
| Refusal precision / recall / F1 | **1.000 / 1.000 / 1.000** |
| Mean mastery gain, adaptive vs random | **0.090 vs 0.083** |
| Weak-topic gain, adaptive vs random | **0.120 vs 0.113** |

The simulated-student comparison is a genuine multi-seed simulation (5 profiles ×
2 policies × 5 seeds × 8 sessions), and the prerequisite gate applies **identically to
both policies** so the comparison is fair. Adaptive wins on average and on the weakest
topics; the random baseline still wins for some individual profiles, and the UI says so.

---

## Honest limitations

- Evaluated on one course and a small hand-built test set.
- The retrieval/refusal thresholds are calibrated for the built-in hashing vectorizer;
  switching to real embeddings changes score magnitudes, so re-tune (the eval tab shows
  the live numbers and there is a sweep you can rerun).
- RAGAS-style faithfulness needs an LLM judge and is reported as unavailable in demo mode.
- Simulated students approximate, but do not replace, real learners.
- BKT parameters are defaults, not fitted on real student data.
- Auto-extracted concept graphs can contain wrong prerequisite edges.
- Vision captions and video transcription require `GEMINI_API_KEY`.

---

## Data sources and attribution

See `data/sources.json` for the machine-readable manifest. The demo corpus is:

- **MIT OpenCourseWare, 6.006 Introduction to Algorithms** — CC BY-NC-SA 4.0.
  <https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/>
- **Open Data Structures**, Pat Morin — CC BY 2.5 CA. <https://opendatastructures.org/>

No copyrighted textbook or video content is redistributed in this repository; only
short derived excerpts are used as citable anchors for the demo.
