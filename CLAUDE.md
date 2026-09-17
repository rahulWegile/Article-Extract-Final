# CLAUDE.md

Production system, deployment, and operations guide for the
Newspaper Archive Studio codebase.

> **Living-document note:** this repo's language routing and
> extraction tuning changes frequently (confirmed by comments in the
> code itself that no longer match the values they describe — see
> the callouts in Section 2). Everything below was verified directly
> against the code on **2026-09-17**. Where this file and the code
> disagree, **the code wins** — every claim below cites the exact
> file to re-check.

---

## 1. System & Architecture Overview

### What it does

Extracts individual newspaper articles — headlines, bodies, images,
authors, dates — from scanned multi-column broadsheet/tabloid PDFs
across English and 12 Indian languages, then makes them searchable
and manually editable.

### End-to-end pipeline flow

```
PDF upload (backend/routes/upload.py)
    │
    ▼
PyMuPDF page rendering (pipeline/render_pdf.py)
    │  -> output/documents/doc_XXXXXX/pages/page_NNN.png
    ▼
Layout & boundary detection
    │  - DocLayout-YOLO (pipeline/layout_detector.py) for every
    │    language except Urdu
    │  - Urdu instead: yolov8m_UrduDoc.pt detects raw text regions
    │    directly (backend/services/urdu_pipeline_service.py) —
    │    bypasses DocLayout-YOLO, PageCleaner, and the shared
    │    article_splitter/sidebox_absorption passes entirely
    ▼
Multi-engine OCR (per language -- see Section 2's matrix)
    │  RapidOCR / Tesseract / UTRNet
    ▼
LLM article grouping + extraction
    │  - Boundary/grouping call: OpenAI or Gemini, per language
    │    (pipeline/gemini/gemini_boundary_pipeline.py via
    │    pipeline/page_processor_gemini.py)
    │  - Article-level extraction: OpenAIArticleExtractor or
    │    GeminiArticleExtractor (pipeline/intelligence/)
    │  - Continuation resolution across pages
    │    (pipeline/intelligence/local/continuation_matching.py)
    ▼
Logical article merging
    │  pipeline/finalization/logical_article_builder.py
    │  -> output/documents/doc_XXXXXX/final_articles/logical_NNNN/
    ▼
PostgreSQL import
    │  pipeline/database/import_final_articles.py
    │  -> documents / articles / article_segments / article_images
    ▼
React/Vite viewer + Fabric.js canvas boundary editor
    (frontend/src/pages/Viewer.jsx, components/viewer/*)
```

### Stack summary

| Layer | Technology |
|---|---|
| Backend | FastAPI (`backend/main.py`), Gunicorn + Uvicorn workers |
| Frontend | React 19 + Vite, Fabric.js-based canvas boundary editor |
| Database | PostgreSQL 16 (`docker/init_db.sql`) |
| Pipeline | PyMuPDF, DocLayout-YOLO / YOLOv8m (Urdu), RapidOCR / Tesseract / UTRNet, OpenAI + Gemini |
| Deployment | Docker Compose (db + backend + frontend), Nginx reverse proxy, single public URL on port 80 |

### Directory architecture

```
├── backend/                  # FastAPI application
│   ├── core/                 # Settings (pydantic-settings), logging, config
│   ├── models/                # Pydantic schemas
│   ├── routes/                # upload, documents, articles, search
│   └── services/               # pipeline_service (orchestrator), urdu_pipeline_service,
│                                # boundary_editor, document_manager, workspace_manager
├── frontend/                  # React + Vite UI (see frontend/Dockerfile, nginx.conf)
├── pipeline/                   # CORE EXTRACTION ENGINE -- see Section 6, Rule 1
│   ├── article/                # ArticleGrouper, splitter, sidebox_absorption, boundary_decomposer
│   ├── database/                # db.py (psycopg3 pool), import_final_articles.py
│   ├── finalization/             # logical_article_builder, final_article_cropper's caller chain
│   ├── gemini/ , openai/          # Gemini/OpenAI API clients
│   ├── intelligence/               # article extractors (openai/gemini/local), masthead parser
│   ├── languages/                   # one module per language -- see Section 2
│   ├── ocr/                          # RapidOCR, Tesseract, UTRNet, Urdu line detector
│   ├── preprocess/                    # PageCleaner (masthead/noise removal)
│   ├── layout_detector.py              # DocLayout-YOLO
│   └── page_processor_gemini.py         # shared per-page orchestrator (all languages, all providers)
├── models/                     # ML weights -- GITIGNORED, see Section 3 Step 2
├── docker/init_db.sql           # Postgres schema (documents/articles/article_segments/article_images)
├── Dockerfile, frontend/Dockerfile, frontend/nginx.conf, docker-compose.yml, docker-compose.cpu.yml
├── requirements.txt
├── run_backend.ps1               # Windows local-dev launcher
├── README.md                       # 3-step Docker quickstart
├── DEPLOYMENT.md                    # cloud GPU VM runbook (RunPod/EC2/Vast.ai)
└── ENGLISH_PIPELINE.md               # English pipeline deep-dive
```

---

## 2. Complete Database Architecture & Schema (`docker/init_db.sql`)

### Logical articles vs. physical segments

A single editorial story ("logical article") can be printed as more
than one physical chunk — a front-page lead that jumps to "Continued
on Page 8", for example. This schema keeps that distinction explicit
rather than flattening it:

- **`articles`** — one row per **logical article**: the complete
  journalistic story, with its final merged title/author/date/text
  and all LLM-derived metadata (category, sentiment, entities,
  topics, keywords). This is what search and the reader-facing views
  query.
- **`article_segments`** — one row per **physical segment**: an
  individual printed crop on one page. A continued/jump story has
  more than one segment row sharing the same `article_id`, ordered by
  `segment_order` and each carrying its own `page_number` and
  `crop_path`. `pipeline/finalization/logical_article_builder.py` is
  what stitches physical segments together into one logical article
  in the first place (cross-segment text-overlap matching); the DB
  schema just mirrors that same one-to-many relationship afterward.

### Table 1 — `documents`

Issue/upload-level metadata — one row per uploaded PDF.

| Column | Type | Notes |
|---|---|---|
| `id` | VARCHAR (PK) | Deterministic UUID5 (`pipeline/database/import_final_articles.py:deterministic_uuid("document", ...)`), not the filesystem `doc_XXXXXX` id |
| `filename` | VARCHAR | Original PDF filename |
| `source_path` | VARCHAR | Path to the source PDF on disk |
| `newspaper_name` | VARCHAR | Masthead-extracted or LLM-identified |
| `publish_date` | DATE | |
| `created_at` | TIMESTAMPTZ | Row insert time |

Indexes: `documents_newspaper_idx`, `documents_publish_date_idx` (both plain B-tree, back `ArticleSearchService.get_newspapers`/`get_publish_dates`).

### Table 2 — `articles` (logical articles)

| Column | Type | Notes |
|---|---|---|
| `id` | VARCHAR (PK) | UUID5 of `document_id:logical_article_id` |
| `document_id` | VARCHAR (FK → `documents.id`, `ON DELETE CASCADE`) | |
| `logical_article_id` | VARCHAR | The pipeline's own logical article id |
| `title`, `author` | VARCHAR | |
| `article_date` | DATE | |
| `category`, `sentiment` | VARCHAR | LLM-classified |
| `summary`, `article_text` | TEXT | |
| `composed_image_path` | VARCHAR | |
| `has_images` | BOOLEAN | |
| `image_count` | INTEGER | |
| `entities`, `topics`, `keywords` | JSONB | LLM-extracted structured metadata |
| `search_vector` | TSVECTOR, **generated** | see below |

Unique constraint: `(document_id, logical_article_id)` — this is what
the importer's `ON CONFLICT (document_id, logical_article_id) DO
UPDATE` upsert targets.

**`search_vector` is a generated column, not application-written:**

```sql
search_vector TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(summary, '')), 'B') ||
    setweight(to_tsvector('english', coalesce(article_text, '')), 'C')
) STORED
```

Postgres recomputes it automatically on every INSERT/UPDATE — the
Python importer never sets it directly, and never needs to.
`articles_search_idx` is a **GIN** index over this column, which is
what makes `backend/services/article_search.py`'s
`search_vector @@ websearch_to_tsquery('english', ...)` and
`ts_rank_cd(search_vector, ...)` calls fast. `articles_doc_idx`
(B-tree on `document_id`) backs the `JOIN documents d ON d.id =
a.document_id` in the same queries.

### Table 3 — `article_segments` (physical segments)

| Column | Type | Notes |
|---|---|---|
| `id` | VARCHAR (PK) | |
| `article_id` | VARCHAR (FK → `articles.id`, `ON DELETE CASCADE`) | |
| `source_article_id` | VARCHAR | The physical crop's own id before logical merging |
| `page_number` | INTEGER | |
| `segment_order` | INTEGER | Ordering within the logical article — segment 1 is the lead, segment 2+ are continuations |
| `article_date`, `title`, `author` | DATE / VARCHAR | Per-segment overrides — `ArticleSearchService._apply_display_metadata` prefers the **last** segment with a non-empty value, so a jump story that starts as a placeholder headline on page 1 and gets its real headline on the page-8 continuation displays correctly |
| `crop_path` | VARCHAR | Path to that segment's cropped image |
| `bbox` | JSONB | |

Unique constraint: `(article_id, segment_order)` — the importer's
second upsert target.

### Table 4 — `article_images`

One row per image the LLM detected inside a logical article, across any of its segments.

| Column | Type | Notes |
|---|---|---|
| `id` | VARCHAR (PK) | |
| `article_id` | VARCHAR (FK → `articles.id`, `ON DELETE CASCADE`) | |
| `segment_id` | VARCHAR (FK → `article_segments.id`, `ON DELETE SET NULL`) | Nullable — an image doesn't always resolve to one specific segment |
| `page_number`, `source_article_id` | INTEGER / VARCHAR | |
| `image_path` | VARCHAR | |
| `description`, `caption` | TEXT | |
| `confidence` | REAL | |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | REAL | |
| `width`, `height` | INTEGER | |
| `created_at` | TIMESTAMPTZ | `ArticleSearchService._get_images` orders by `page_number ASC, created_at ASC` — this column is load-bearing for that, not just an audit trail |

Index: `images_article_idx` (B-tree on `article_id`).

> This 4-table schema (plus `documents`) was reverse-engineered
> directly from every INSERT/SELECT that touches the database —
> there is no ORM or migration file anywhere else in this repo (see
> `pipeline/database/delete_document.py`'s own docstring, which
> explicitly deletes child rows in order rather than relying on FK
> cascade for exactly this reason). `docker/init_db.sql` is the only
> place this schema is defined; treat it as the single source of
> truth and edit it directly if the schema needs to change.

### Connection pool

`pipeline/database/db.py` opens one shared `psycopg3` pool for the
whole process:

```python
_pool = ConnectionPool(
    conninfo=f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}",
    min_size=DB_POOL_MIN_SIZE,   # env DB_POOL_MIN_SIZE, default 1
    max_size=DB_POOL_MAX_SIZE,   # env DB_POOL_MAX_SIZE, default 10
    kwargs={"row_factory": dict_row},
    open=False,
)
```

`get_connection()` lazily opens the pool on first use and returns a
context-manager connection (auto-commit/rollback on exit, returned to
the pool rather than closed); `close_pool()` is called on FastAPI
shutdown (`backend/main.py`'s `shutdown_db_pool`). All queries get
dict-like rows (`row_factory=dict_row`), not tuples.

### Database initialization

- **Docker: fully automatic.** `docker-compose.yml` mounts
  `./docker/init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro` on
  the `db` service. Postgres's own entrypoint runs every `*.sql` file
  in that directory **once**, only on a completely fresh (empty)
  `postgres_data` volume — it will *not* re-run on an existing
  volume, even after a schema edit. To force it to re-apply, you must
  wipe the volume: `docker compose down -v` then `docker compose up
  -d --build` again (this also deletes all existing article data).
- **Local (no Docker):**
  ```bash
  psql -U postgres -c "CREATE DATABASE newspaper_archive;"
  psql -U postgres -d newspaper_archive -f docker/init_db.sql
  ```

---

## 3. Language Routing & Engine Matrix

Each language declares its own OCR engine, LLM provider, and article
extractor in `pipeline/languages/<code>/__init__.py`, resolved by
`pipeline/languages/registry.py`. **This is the single source of
truth** — the table below is a snapshot; re-check the actual file
before relying on it for anything provider-cost- or accuracy-
sensitive.

| Language | OCR engine | Boundary/grouping LLM | Article extraction | Batch: pages / max articles |
|---|---|---|---|---|
| English | RapidOCR | OpenAI | OpenAI | 1 / — |
| Hindi | Tesseract (`hin`) | OpenAI | OpenAI | 2 / 25 |
| Urdu | *dedicated pipeline* (see below) | OpenAI (Gemini fallback) | OpenAI | 1 / 8 |
| Gujarati | Tesseract (`guj`) | OpenAI | OpenAI | 1 / 8 |
| Marathi | RapidOCR | OpenAI | OpenAI | 1 / 8 |
| Punjabi | Tesseract (`pan`) | OpenAI | OpenAI | 1 / 8 |
| Bengali | Tesseract (`ben`) | OpenAI | OpenAI | 1 / 8 |
| Assamese | Tesseract (`asm`) | OpenAI | OpenAI | 2 / 25 |
| Odia | Tesseract (`ori`) | OpenAI | OpenAI | 2 / 8 |
| Kannada | Tesseract (`kan`) | OpenAI | OpenAI | 1 / 8 |
| Tamil | Tesseract (`tam`) | OpenAI | OpenAI | 2 / 25 |
| Telugu | Tesseract (`tel`) | OpenAI | **Gemini** | 1 / — |
| Malayalam | Tesseract (`mal`) | OpenAI | OpenAI | 1 / — |

**⚠️ This differs from what you may have been told, and from stale
comments still sitting in the code itself — verify before trusting
either:**

- **Every language currently has `llm_provider="openai"`** in its
  `__init__.py`. There is no live "Hindi → Gemini" routing today,
  despite `.env.example`'s own comments and a stale note in
  `backend/services/pipeline_service.py` (~line 2494: *"Hindi groups
  boundaries via OpenAI but extracts articles via
  GeminiArticleExtractor"*) both describing an older configuration.
  `backend/services/pipeline_service.py` line ~1444
  (`document_llm_provider = lang_pipeline.llm_provider`) applies the
  per-language field directly with no per-document override anymore.
- **Telugu, not Hindi, is the one language on Gemini** — for
  *article-level extraction only* (`extractor_class=GeminiArticleExtractor`
  in `pipeline/languages/telugu/__init__.py`), because
  `OPENAI_ARTICLE_MODEL=gpt-5.6-luna` was confirmed failing on
  Telugu conjunct clusters. Its own comment flags that Marathi,
  Punjabi, Gujarati, Assamese, Bengali, and Kannada share the same
  Tesseract+OpenAI architecture and *could* need the same fix if the
  same failure shows up there — they haven't been switched as of
  this writing.
- **Urdu has its own dedicated pipeline**
  (`backend/services/urdu_pipeline_service.py`), not the shared
  `LanguagePipeline`/OCR-engine path: `models/urdu_text_detection/
  yolov8m_UrduDoc.pt` detects text regions directly, UTRNet
  (`models/utrnet/UTRNet-Large.pth`) reads them, and a single
  grouping call (OpenAI by default; falls back to Gemini only if no
  `OPENAI_API_KEY` is configured or `LLM_PROVIDER=gemini`) assigns
  OCR region IDs to articles. `pipeline/languages/urdu/__init__.py`
  still exists and is matched by the registry, but
  `backend/services/pipeline_service.py`'s
  `if lang_pipeline.code == "urdu":` branch intercepts before that
  definition's OCR engine is ever used.
- **Unmatched/unrecognized languages fall back to Hindi**, not
  English (`pipeline/languages/registry.py`,
  `resolve_language_pipeline`'s final `return HINDI`).

### Batching constraints

`extraction_max_articles_per_batch` (articles per LLM call) and
`extraction_pages_per_batch` are **not** a single fixed rule across
all languages — each is set per language in that language's own
`__init__.py`:

- **8 articles / 1 page** — Gujarati, Marathi, Punjabi, Bengali,
  Kannada, Urdu (the majority).
- **25 articles / 2 pages** — Hindi, Tamil, Assamese.
- **8 articles / 2 pages** — Odia.
- **No cap (page-count only)** — English, Telugu, Malayalam.

The reasoning (from the code's own comments): `gpt-5.6-luna` starves
its own completion-token budget (~470 tokens/article) past ~8-10
articles in one call, dropping whole columns; a page that would push
a batch over its language's cap is held back to start the next batch
rather than being force-fit in.

### OpenAI reasoning-model temperature rule

`pipeline/intelligence/openai_article_extractor.py` (~line 195):

```python
is_reasoning_model = any(
    x in (self.model or "").lower() for x in ("luna", "o1", "o3", "o4", "gpt-5")
)
self._temperature_supported = not is_reasoning_model
```

Any model name containing `luna`, `o1`, `o3`, `o4`, or `gpt-5`
(substring match, not an exact list — this also catches `gpt-5-mini`,
`gpt-5.6-luna`, etc.) never gets a `temperature` kwarg — these
models reject any value other than their default (1) with a 400.
`kwargs["temperature"] = 0` is only added when
`self._temperature_supported` is true (~line 1406).

---

## 4. Step-by-Step Deployment Runbook (Production / Cloud VM)

Full detail and copy-paste commands live in **`DEPLOYMENT.md`** —
this is the condensed 6-step version.

### Prerequisites

Ubuntu 22.04 VM, NVIDIA GPU (T4/G4dn minimum; A10G/L4+ for closer to
the ~5-10s/page figure vs. ~2.5min/page on CPU), NVIDIA driver,
Docker Engine + Compose plugin, NVIDIA Container Toolkit.

### Step 1 — System prep & NVIDIA Container Toolkit

```bash
sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall && sudo reboot   # then: nvidia-smi

curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# Sanity check
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

### Step 2 — Clone the repo and unpack model weights

```bash
git clone <repo-url> newspaper-archive && cd newspaper-archive
```

**Model weights are gitignored and required on the host** — they are
volume-mounted at runtime (`./models:/app/models`), never baked into
the image. Unzip `models.zip` (shared out-of-band — too large for
git) into `models/` so it matches:

```
models/
  doclayout_yolo.pt
  urdu_line_detector/yolov8m_UrduDoc.pt
  urdu_text_detection/yolov8m_UrduDoc.pt
  utrnet/UTRNet-Large.pth
  tessdata/            (also fetched automatically at image build time)
```

### Step 3 — Configure `.env`

```bash
cp .env.example .env
nano .env   # OPENAI_API_KEY, GEMINI_API_KEY at minimum
```

PostgreSQL credentials are **not** required here for a Docker run —
`docker-compose.yml` defaults `POSTGRES_USER`/`POSTGRES_PASSWORD`/
`POSTGRES_DB` via `${VAR:-default}` substitution (`postgres` /
`postgres` / `newspaper_archive`) and forces `DB_HOST=db` for the
backend container regardless of what's in `.env`. Only set `DB_*` in
`.env` if you want non-default values.

### Step 4 — Start services

```bash
docker compose up -d --build
```

First boot takes a few minutes (CUDA base image pull, pip install,
tessdata download, Postgres schema init via `docker/init_db.sql`).
Watch it with `docker compose logs -f`.

### Step 5 — CPU-only fallback (no GPU / no NVIDIA Container Toolkit)

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

`docker-compose.cpu.yml` sets `OCR_DEVICE=cpu` and clears the GPU
device reservation (via the compose `!reset` tag — a plain empty
`deploy: {}` would **not** remove it, since Compose merges list/map
fields across `-f` files rather than replacing them by default).

### Step 6 — Post-deployment verification

```bash
curl http://localhost/health
# -> {"status":"ok","db":"ok"}

docker exec -it $(docker compose ps -q backend) python3.12 -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True Tesla T4   (or whatever GPU is attached; False on the CPU override)
```

Open `http://<server-ip>` — a single URL serves the frontend and, via
Nginx's reverse proxy, the backend API too (see `frontend/nginx.conf`)
— nothing else needs opening in the security group except 80 and 22.

---

## 5. Local Development Setup (Windows & Non-Docker)

### Backend

```powershell
powershell -ExecutionPolicy Bypass -File run_backend.ps1
```

`run_backend.ps1` does two things that matter:

1. Prepends the venv's own `Scripts/` to `PATH`
   (`$env:PATH = "$scriptDir\venv\Scripts;$env:PATH"`) — CUDA torch's
   driver-safety probe subprocess resolves a bare `python`/`python.exe`
   by PATH lookup internally on import, independent of anything this
   project or uvicorn controls. Without this, it silently resolves to
   the global Python install and loses CUDA torch.
2. Launches with `& "$scriptDir\venv\Scripts\python.exe" -m uvicorn
   backend.main:app --port 8000` — **deliberately not** `--reload`.

**Never use `uvicorn --reload` on Windows for this project.** Reload
mode spawns its worker via `multiprocessing`, which on Windows
re-execs using the *global* Python install rather than the venv —
silently losing CUDA torch and falling back to a CPU-only (or
broken) install. Restart the script by hand after code changes
instead.

Runs on `http://127.0.0.1:8000` (docs at `/docs`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:5173`. `frontend/src/api/api.js` points at
`VITE_API_URL` or falls back to `http://127.0.0.1:8000`.

### Database (no Docker)

Install PostgreSQL 16 locally, create the `newspaper_archive`
database, then run the same schema Docker would auto-apply:

```bash
psql -U postgres -c "CREATE DATABASE newspaper_archive;"
psql -U postgres -d newspaper_archive -f docker/init_db.sql
```

Set `DB_HOST=localhost` (the default in `backend/core/config.py`) in
your local `.env` — no container networking involved.

---

## 6. Day-to-Day Operations & Debugging

### Logs

```bash
docker compose logs -f backend     # tail backend/pipeline logs
docker compose logs -f             # everything
```

### Restarting after config changes

```bash
docker compose restart backend                 # .env changed, no code/deps changed
docker compose up -d --build backend            # code or requirements.txt changed
docker compose up -d --build                    # frontend or Dockerfile changed too
```

`docker compose down` stops everything but **keeps** volumes
(`postgres_data`, plus the bind-mounted `output/`, `uploads/`,
`cache/`, `models/`). Only `docker compose down -v` or manually
deleting those host directories loses data.

### Output directory structure

Verified against a real processed document
(`output/documents/doc_000029/`):

```
output/documents/doc_XXXXXX/
├── document.json                    # top-level metadata, status
├── pages/                            # page_NNN.png (PyMuPDF render)
├── final/                             # page_NNN_final_boundaries.png + boundaries.pdf
├── gemini_article_batches/             # raw per-batch LLM extraction JSON
│                                        # (named "gemini_..." regardless of the
│                                        #  actual provider used -- see
│                                        #  pipeline/intelligence/local/
│                                        #  local_article_extractor.py's own note)
├── final_articles_crops/
│   └── page_NNN/article_NNN/            # crop.json + page_NNN.png per physical crop
├── final_articles/
│   ├── final_articles.json               # merged logical articles
│   └── logical_NNNN/                      # one dir per logical (cross-page-merged) article
└── article_images/
    ├── page_NNN/                           # per-page detected images
    └── logical_NNNN/                        # images attached to each logical article
```

### Re-running / inspecting a single document

- All artifacts for one document live entirely under
  `output/documents/doc_XXXXXX/` — safe to `rm -rf` that one
  directory and re-upload the same PDF to reprocess from scratch.
- To remove it from the database too (not just the filesystem), use
  `pipeline/database/delete_document.py`'s
  `delete_document_from_db(document_id)` (also exposed via the
  `DELETE /documents/{document_id}` route,
  `backend/routes/documents.py`) — it deletes
  `article_images` → `article_segments` → `articles` → `documents`
  explicitly, not via `ON DELETE CASCADE` (there's no committed
  migration file, so FK cascade behavior can't be assumed — see that
  file's own docstring).
- `gemini_article_batches/*.json` holds the raw per-batch LLM output
  — check here first when an article's extracted text looks wrong,
  before assuming a boundary/OCR problem.

---

## 7. Critical Rules & Guardrails for Claude (MUST NEVER VIOLATE)

1. **Never delete or truncate anything in `pipeline/`.** It is the
   project's core IP — article grouping, OCR engines, layout
   detectors, per-language prompts, and boundary repair algorithms
   all live here. Only touch it when explicitly asked.

2. **Unbounded universe constraint.** Never hardcode newspaper names,
   editions, or layouts into core logic — the pipeline must handle
   any newspaper via geometry/OCR/LLM reasoning, not lookup tables.
   *Exception:* `pipeline/intelligence/local/masthead/registry_data.json`
   is a fast-path cache for known templates; the pipeline must always
   gracefully fall back to the LLM path for an unknown paper.

3. **No unnecessary heavy dependencies.** Don't introduce a new OCR
   engine or deep-learning framework. The existing stack —
   `rapidocr`, `pytesseract`, `torch`/`torchvision`, `ultralytics`,
   `doclayout-yolo`, `pymupdf`, `opencv-python`, `paddlepaddle`/
   `paddlex` — is already the full toolset.

4. **Clean repository hygiene.** No scratch debug scripts
   (`check_db*.py`), test images, or dump files (`scratch*.txt`,
   `titles*.txt`) in the repo root. Runtime artifacts belong in
   `output/`, `uploads/`, or `cache/` (all gitignored). Follow
   `.gitignore`/`.dockerignore` as already configured.

5. **YAGNI.** Write only what's needed for the current task — no
   speculative features, no unnecessary wrapper abstractions.

---

## 8. Troubleshooting Guide

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: ... weights not found: <path>` (or a generic file-not-found from `ultralytics`/`doclayout_yolo`) | A model file is missing under `models/` | Confirm `models.zip` was actually unzipped into `models/` at the repo root (Step 2) — check the exact path named in the error against the layout in that step |
| `could not select device driver "" with capabilities: [[gpu]]` | No NVIDIA GPU / NVIDIA Container Toolkit on this host, but `docker-compose.yml`'s GPU reservation is active | Run with the CPU override: `docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build` |
| OpenAI 400: `temperature` not supported for this model | A reasoning-family model (`gpt-5.6-luna`, `o1`/`o3`/`o4`, or anything with `gpt-5` in the name) was sent a custom `temperature` | Should already be prevented by `pipeline/intelligence/openai_article_extractor.py`'s `is_reasoning_model` check (~line 195) — if it still fires, the model name doesn't match any of `luna`/`o1`/`o3`/`o4`/`gpt-5`; extend that substring list |
| Backend can't reach the database / `psycopg` connection errors | Postgres container unhealthy, or credential mismatch between `.env` and what the `db` service actually started with | `docker compose ps` (is `db` healthy?); remember `docker-compose.yml`'s `environment:` block **overrides** `.env`'s `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD` for the backend container — check the compose file's actual values, not just `.env` |
| `ImportError: no pq wrapper available` | `psycopg` couldn't find a libpq implementation | Already fixed in this repo two ways — `requirements.txt` pins `psycopg[binary]` (bundles its own) and the `Dockerfile` installs system `libpq5` as belt-and-suspenders; if this resurfaces, one of those two got reverted |
| `ModuleNotFoundError: No module named 'doclayout_yolo'` | Image was built from an older `requirements.txt` that predates the `doclayout-yolo==0.0.4` pin | Rebuild: `docker compose up -d --build backend` (Docker's layer cache invalidates automatically once `requirements.txt`'s content hash changes) |

---

## Coding Standards & Guidelines

- **Modularity:** keep functions single-purpose; isolate heuristics
  into their own helpers.
- **Resource awareness:** mind CPU/RAM when working with high-DPI
  page images (PyMuPDF / OpenCV / PIL).
- **Preserve existing fixes:** don't rewrite a working module in
  passing — most non-obvious logic here exists because of a
  confirmed real-page failure (see the extensive inline comments
  throughout `pipeline/`), not by accident.
