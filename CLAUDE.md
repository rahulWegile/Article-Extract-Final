# CLAUDE.md

## Project Overview
This repository contains an end-to-end **Newspaper Article Boundary Detection & Extraction System**. It extracts individual articles, headlines, bodies, images, and metadata from scanned multi-column newspaper PDFs across multiple Indian languages and English.

The system consists of:
- **Backend:** FastAPI REST API (`backend/`) orchestrating document uploads, background processing, search, and boundary editing.
- **Frontend:** React + Vite Single Page Application (`frontend/`) featuring an interactive Fabric.js canvas for viewing and manually editing article boundaries.
- **Pipeline:** High-performance computer vision, OCR, and AI pipeline (`pipeline/`) combining DocLayout-YOLO, RapidOCR / Tesseract / UTRNet, geometric heuristics, and LLMs (OpenAI / Gemini).
- **Database:** PostgreSQL storing documents, articles, segments, and images (`docker/init_db.sql`).

---

## CRITICAL RULES FOR CLAUDE (MUST NEVER VIOLATE)

1. **DO NOT DELETE OR REMOVE ANYTHING FROM `pipeline/`**:
   - The `pipeline/` directory is the core intellectual property of the project.
   - NEVER delete, prune, or truncate any file inside `pipeline/` unless specifically and explicitly requested by the user.
   - All article grouping, OCR engines, layout detectors, language prompts, and boundary repair algorithms live here.

2. **Unbounded Universe Constraint**:
   - The pipeline MUST support an unbounded universe of newspapers.
   - Never hardcode lists of newspaper names, editions, or languages in core logic.
   - Dynamic extraction, spatial/geometric heuristics, and OCR/LLMs must handle arbitrary newspapers.
   - *Exception:* `pipeline/intelligence/local/masthead/registry_data.json` is a fast-path cache for known templates to avoid LLM calls; the pipeline must always gracefully fall back for unknown papers.

3. **No Unnecessary Dependencies**:
   - Do NOT introduce new heavy libraries (no new OCR engines, no new deep learning frameworks).
   - Strictly adhere to the existing stack: `rapidocr`, `pymupdf` (`fitz`), `opencv-python`, `pillow`, `torch`, `pytesseract`, and `paddlepaddle`/`paddlex`.

4. **YAGNI (You Aren't Gonna Need It)**:
   - Write only what is strictly necessary to solve the current problem.
   - Do not over-engineer, add speculative features, or create unnecessary boilerplate/wrappers.
   - Keep interactions and code edits token-efficient and direct.

5. **File Hygiene & No Scratch Clutter**:
   - Do not create scratch debug scripts (e.g., `check_db*.py`), test images (`*.png`), or dump files (`scratch*.txt`, `titles*.txt`) in the root directory.
   - All runtime outputs belong in `output/`, `uploads/`, or `cache/` (which are gitignored).
   - Follow `.gitignore` rules strictly.

---

## Directory Architecture

```
├── backend/                  # FastAPI Application
│   ├── core/                 # Settings, logging, config (pydantic-settings)
│   ├── models/               # Pydantic schemas (document.py, etc.)
│   ├── routes/               # API endpoints (upload, documents, articles, search)
│   ├── services/             # Pipeline runner, boundary editor, search, document manager
│   └── main.py               # FastAPI entry point, CORS, static mounts (/documents, /output)
├── frontend/                 # React + Vite UI
│   ├── src/
│   │   ├── api/api.js        # Axios instance configured for backend API
│   │   ├── components/       # DocumentCard, UploadCard, ProcessingStatus, ToastStack
│   │   │   └── viewer/       # CanvasToolbar, BoundaryLayer, ArticleInspector, PageSidebar
│   │   ├── pages/            # Home, Viewer, ArticleDetail, Search
│   │   └── styles/           # Global styles and archive CSS
│   ├── package.json
│   └── vite.config.js
├── pipeline/                 # CORE EXTRACTION ENGINE (DO NOT DELETE FROM HERE)
│   ├── article/              # ArticleGrouper, candidate generators, merge logic
│   ├── database/             # PostgreSQL connection pool (db.py)
│   ├── export/               # PDF & JSON article export utilities
│   ├── finalization/         # High-resolution article image cropper
│   ├── gemini/               # Gemini API client & structured extractors
│   ├── intelligence/         # Local masthead parser, knowledge builder
│   ├── languages/            # Per-language prompt templates (Hindi, Marathi, Urdu, etc.)
│   ├── ocr/                  # RapidOCR, Tesseract, UTRNet Urdu OCR engines
│   ├── openai/               # OpenAI API client fallback
│   ├── preprocess/           # Image deskew, clean, rule line detection
│   ├── layout_detector.py    # YOLOv8 DocLayout boundary detector
│   ├── page_processor_gemini.py # Per-page extraction pipeline orchestrator
│   └── render_pdf.py         # PyMuPDF PDF page renderer
├── models/                   # Local ML weights (DO NOT DELETE)
│   ├── doclayout_yolo.pt     # DocLayout-YOLO model
│   ├── urdu_line_detector/   # YOLOv8m Urdu line detector
│   ├── utrnet/               # UTRNet model weights
│   └── tessdata/             # Tesseract language data
├── docker/                   # Deployment files
│   └── init_db.sql           # PostgreSQL database schema & indexes
├── docker-compose.yml        # Multi-container setup (db, backend, frontend)
├── requirements.txt          # Python dependencies
└── run_backend.ps1           # Windows launch script for backend
```

---

## Running the Application

### Backend (Windows Local Development)
Run using the provided PowerShell script:
```powershell
powershell -ExecutionPolicy Bypass -File run_backend.ps1
```
*Important Windows Quirks:*
- `run_backend.ps1` sets `$env:PATH = "$scriptDir\venv\Scripts;$env:PATH"` so CUDA torch subprocess probes resolve to the venv rather than global Python.
- Do NOT use `uvicorn --reload` on Windows: reload mode spawns worker processes via multiprocessing that resolve to the global Python install, breaking CUDA torch.
- Direct invocation:
  ```powershell
  & ".\venv\Scripts\python.exe" -m uvicorn backend.main:app --port 8000
  ```
- Backend runs on `http://127.0.0.1:8000`. API docs available at `http://127.0.0.1:8000/docs`.

### Frontend Local Development
```bash
cd frontend
npm install
npm run dev
```
- Runs on `http://localhost:5173`.
- Axios is configured in `frontend/src/api/api.js` pointing to `VITE_API_URL` or `http://127.0.0.1:8000`.

### Full Stack via Docker
```bash
docker compose up -d --build
```
- PostgreSQL (`5432`), Backend (internal `8000`), Frontend + Nginx (`80`).
- Models, outputs, uploads, and cache are bind-mounted.

---

## Database Schema Highlights (`docker/init_db.sql`)
- **`documents`**: Tracks uploaded PDFs, total pages, status (`pending`, `processing`, `completed`, `failed`), and document metadata.
- **`articles`**: Extracted articles with newspaper name, edition, date, page number, language, category, headline, full text, and confidence.
- **`article_segments`**: Individual polygon/bounding-box segments making up each article (headline, body paragraphs, images).
- **`article_images`**: Cropped images associated with articles.

---

## Coding Standards & Guidelines
- **Modularity:** Keep functions single-purpose. Isolate heuristics in their own helper functions.
- **Resource Awareness:** Mind CPU and RAM limitations when working with high-DPI page images (PyMuPDF / OpenCV / PIL).
- **Preserve Existing Fixes:** Always preserve existing bug fixes and comments. Do not rewrite working modules.
