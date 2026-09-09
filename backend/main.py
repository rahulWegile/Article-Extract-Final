import sys
from pathlib import Path
import logging

# The pipeline prints unicode symbols (checkmarks, arrows) for progress
# logging. On Windows, stdout/stderr default to the cp1252 console
# codepage, which raises UnicodeEncodeError and crashes the pipeline
# thread the first time one of those symbols is printed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from backend.core.logging_config import configure_logging

configure_logging()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.upload import (
    router as upload_router,
)

from backend.routes.documents import (
    router as documents_router,
)

from backend.routes.articles import (
    router as articles_router,
)

from backend.core.settings import (
    DOCUMENTS_DIR,
)

from backend.core.config import settings

from pipeline.database.db import close_pool, get_connection


logger = logging.getLogger(__name__)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Newspaper Boundary Detector API",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# Local development defaults.
#
# For deployment, set:
#
# CORS_ORIGINS=https://your-frontend-domain.com
#
# Multiple origins can be separated by commas:
#
# CORS_ORIGINS=https://site1.com,https://site2.com
#

cors_origins = settings.CORS_ORIGINS.split(",")


cors_origins = [
    origin.strip()
    for origin in cors_origins
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,

    allow_origins=cors_origins,

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# API ROUTES
# ============================================================

app.include_router(
    upload_router
)


app.include_router(
    documents_router
)


app.include_router(
    articles_router
)


# ============================================================
# EXISTING DOCUMENT FILES
#
# Used by the newspaper page/boundary viewer.
#
# DO NOT REMOVE.
# ============================================================

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/documents",

    StaticFiles(
        directory=str(
            DOCUMENTS_DIR
        )
    ),

    name="documents",
)


# ============================================================
# ARTICLE IMAGES / CROPS
#
# URLs:
#
#     /output/...
#
# Example:
#
#     /output/documents/doc_000007/...
#
# ============================================================

OUTPUT_DIR = Path(
    "output"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


app.mount(
    "/output",

    StaticFiles(
        directory=str(
            OUTPUT_DIR
        )
    ),

    name="output",
)


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message":
            "Newspaper Boundary Detector API Running"
    }


@app.get("/health")
def health():

    db_status = "ok"

    try:

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

    except Exception:

        logger.exception("Health check DB probe failed")

        db_status = "error"

    return {
        "status": "ok",
        "db": db_status,
    }


# ============================================================
# UNHANDLED EXCEPTIONS
#
# Logs the full traceback server-side; the response shape
# matches FastAPI's own default 500 body, so this is purely
# additive observability, not a behavior change.
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):

    logger.exception(
        "Unhandled exception while processing %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


@app.on_event("shutdown")
def shutdown_db_pool():
    close_pool()
