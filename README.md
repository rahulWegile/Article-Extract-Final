# Newspaper Archive Studio

Multilingual newspaper boundary detection, article extraction, and
archive search — FastAPI backend, React/Vite frontend, PostgreSQL,
and a computer-vision + OCR + LLM pipeline (see [CLAUDE.md](CLAUDE.md)
for the full architecture).

## Run via Docker

Three steps:

1. **Unzip `models.zip` into `models/`** at the repo root (contains
   the DocLayout-YOLO, Urdu (UTRNet + YOLOv8 detector), and Tesseract
   language weights — too large for git, shared separately).

2. **Copy `.env.example` to `.env`** and fill in your API keys
   (`OPENAI_API_KEY`, `GEMINI_API_KEY`). PostgreSQL credentials are
   already handled automatically inside Docker — no need to set
   `DB_*` unless you want non-default values.

3. **Run it**:

   - With an NVIDIA GPU + NVIDIA Container Toolkit installed:
     ```bash
     docker compose up --build
     ```
   - Without a GPU (CPU fallback):
     ```bash
     docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
     ```

The app is served at **http://localhost** (port 80, via Nginx), and
the backend's health endpoint is at **http://localhost/health**.

For deploying to a cloud GPU VM (RunPod, AWS EC2 G4dn/G5, Vast.ai,
etc.), see [DEPLOYMENT.md](DEPLOYMENT.md).
