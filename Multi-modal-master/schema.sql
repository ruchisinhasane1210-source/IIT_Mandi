-- ============================================================
--  TraceLearn — Canonical schema (PostgreSQL + pgvector)
--
--  This single file is the source of truth. At boot, db.py
--  down-converts it to SQLite so the app runs with zero setup
--  (VECTOR -> TEXT, UUID -> TEXT, JSONB -> TEXT, drops extensions).
--  Keep the dialects compatible: TEXT ids, TEXT json columns.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
  id                 TEXT PRIMARY KEY,
  name               TEXT NOT NULL,
  preferred_language TEXT DEFAULT 'English',
  created_at         TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS courses (
  id                 TEXT PRIMARY KEY,
  owner_id           TEXT REFERENCES users(id),
  title              TEXT NOT NULL,
  description        TEXT,
  exam_date          TEXT,
  daily_study_minutes INTEGER DEFAULT 45,
  created_at         TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS sources (
  id                TEXT PRIMARY KEY,
  course_id         TEXT REFERENCES courses(id),
  source_id         TEXT,                    -- manifest key, e.g. ocw_6006_lec08
  file_name         TEXT NOT NULL,
  source_type       TEXT NOT NULL,           -- pdf | slides | video | image
  storage_path      TEXT,
  storage_url       TEXT,
  mime_type         TEXT,
  num_pages         INTEGER,
  duration_seconds  REAL,
  license           TEXT,
  attribution       TEXT,
  source_url        TEXT,
  processing_status TEXT DEFAULT 'pending',  -- pending|processing|done|failed
  progress          INTEGER DEFAULT 0,
  error             TEXT,
  created_at        TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS content_units (
  id               TEXT PRIMARY KEY,
  source_id        TEXT REFERENCES sources(id),
  course_id        TEXT REFERENCES courses(id),
  content          TEXT NOT NULL,
  unit_type        TEXT,                     -- text|slide|transcript|figure
  page_number      INTEGER,
  slide_number     INTEGER,
  timestamp_start  REAL,                     -- float seconds, formatted in UI only
  timestamp_end    REAL,
  figure_reference TEXT,
  topic_ids        TEXT,                     -- JSON array of concept ids
  concept_ids      TEXT,                     -- JSON array of concept ids
  metadata         TEXT,                     -- JSON
  embedding        VECTOR(768),              -- Gemini text-embedding-004
  tsv              TSVECTOR,
  created_at       TEXT DEFAULT (NOW())::text
);

CREATE INDEX IF NOT EXISTS idx_units_embedding ON content_units
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_units_tsv ON content_units USING GIN (tsv);

CREATE TABLE IF NOT EXISTS concepts (
  id          TEXT PRIMARY KEY,
  course_id   TEXT REFERENCES courses(id),
  name        TEXT NOT NULL,
  level       TEXT,                          -- topic|subtopic|concept
  description TEXT,
  parent_id   TEXT,
  order_index INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS prerequisites (
  prerequisite_id TEXT NOT NULL,
  dependent_id    TEXT NOT NULL,
  confidence      REAL DEFAULT 0.5,
  PRIMARY KEY (prerequisite_id, dependent_id)
);

CREATE TABLE IF NOT EXISTS questions (
  id                  TEXT PRIMARY KEY,
  course_id           TEXT REFERENCES courses(id),
  question_text       TEXT NOT NULL,
  question_type       TEXT NOT NULL,         -- mcq|short_answer|numerical
  options             TEXT,                  -- JSON, each option carries its misconception
  correct_answer      TEXT,
  explanation         TEXT,
  difficulty          INTEGER DEFAULT 3 CHECK (difficulty BETWEEN 1 AND 5),
  topic_id            TEXT,
  source_unit_ids     TEXT,                  -- JSON array
  source_citation     TEXT,                  -- JSON
  verification_status TEXT DEFAULT 'pending',
  verification_log    TEXT,                  -- JSON, per-gate results
  question_hash       TEXT,
  embedding           VECTOR(768),
  created_at          TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS assessments (
  id            TEXT PRIMARY KEY,
  user_id       TEXT REFERENCES users(id),
  course_id     TEXT REFERENCES courses(id),
  kind          TEXT,                        -- diagnostic|quiz|mock_exam
  topic_ids     TEXT,                        -- JSON array
  status        TEXT DEFAULT 'active',
  score         REAL,
  created_at    TEXT DEFAULT (NOW())::text,
  completed_at  TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
  id                   TEXT PRIMARY KEY,
  assessment_id        TEXT REFERENCES assessments(id),
  user_id              TEXT REFERENCES users(id),
  question_id          TEXT REFERENCES questions(id),
  submitted_answer     TEXT,
  is_correct           INTEGER,
  confidence           INTEGER,
  response_time_seconds INTEGER,
  misconception        TEXT,
  feedback             TEXT,
  created_at           TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS mastery_records (
  id                  TEXT PRIMARY KEY,
  user_id             TEXT REFERENCES users(id),
  course_id           TEXT REFERENCES courses(id),
  topic_id            TEXT REFERENCES concepts(id),
  p_mastery           REAL DEFAULT 0.30,     -- BKT posterior
  attempts            INTEGER DEFAULT 0,
  correct             INTEGER DEFAULT 0,
  misconception_count INTEGER DEFAULT 0,
  last_practiced_at   TEXT,
  updated_at          TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS misconceptions (
  id           TEXT PRIMARY KEY,
  user_id      TEXT REFERENCES users(id),
  course_id    TEXT REFERENCES courses(id),
  topic_id     TEXT,
  name         TEXT NOT NULL,
  count        INTEGER DEFAULT 1,
  last_seen_at TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS chat_messages (
  id             TEXT PRIMARY KEY,
  course_id      TEXT REFERENCES courses(id),
  user_id        TEXT REFERENCES users(id),
  role           TEXT NOT NULL,              -- user|assistant
  content        TEXT NOT NULL,
  grounding_status TEXT,                     -- grounded|partial|not_covered
  grounding_score  REAL,
  citations      TEXT,                       -- JSON array
  outside_knowledge TEXT,
  topic_ids      TEXT,                       -- JSON array
  created_at     TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS jobs (
  id          TEXT PRIMARY KEY,
  course_id   TEXT REFERENCES courses(id),
  source_id   TEXT REFERENCES sources(id),
  kind        TEXT,
  status      TEXT DEFAULT 'pending',        -- pending|processing|done|failed
  step        TEXT,
  progress    INTEGER DEFAULT 0,
  message     TEXT,
  result      TEXT,                          -- JSON
  created_at  TEXT DEFAULT (NOW())::text,
  updated_at  TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS study_plan_items (
  id          TEXT PRIMARY KEY,
  user_id     TEXT REFERENCES users(id),
  course_id   TEXT REFERENCES courses(id),
  plan_date   TEXT,
  topic_id    TEXT,
  minutes     INTEGER DEFAULT 20,
  kind        TEXT,                          -- review|practice|flashcards
  reason      TEXT,
  done        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS eval_runs (
  id          TEXT PRIMARY KEY,
  course_id   TEXT REFERENCES courses(id),
  kind        TEXT,                          -- ragas|refusal|simulation|all
  status      TEXT DEFAULT 'pending',
  metrics     TEXT,                          -- JSON
  created_at  TEXT DEFAULT (NOW())::text
);

CREATE TABLE IF NOT EXISTS llm_cache (
  key        TEXT PRIMARY KEY,
  response   TEXT,
  model      TEXT,
  created_at TEXT DEFAULT (NOW())::text
);
