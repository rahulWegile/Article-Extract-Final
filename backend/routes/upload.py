import threading
import traceback
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File
from google.genai.errors import APIError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APIStatusError as OpenAIStatusError

from backend.core.config import settings
from backend.services.pipeline_service import PipelineService
from backend.services import upload_jobs
from backend.services.ai_error_messages import (
    gemini_error_detail as _gemini_error_detail,
    openai_error_detail as _openai_error_detail,
)

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

_UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post("/upload")
def upload_pdf(
    file: UploadFile = File(...)
):

    #
    # Save uploaded PDF
    #

    safe_filename = Path(file.filename).name

    if not safe_filename or safe_filename in (".", ".."):

        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename: {file.filename!r}",
        )

    file_path = UPLOAD_DIR / safe_filename

    bytes_written = 0

    with open(file_path, "wb") as buffer:

        while chunk := file.file.read(_UPLOAD_CHUNK_SIZE):

            bytes_written += len(chunk)

            if bytes_written > MAX_UPLOAD_SIZE_BYTES:

                buffer.close()
                file_path.unlink(missing_ok=True)

                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File exceeds the maximum upload size of "
                        f"{settings.MAX_UPLOAD_SIZE_MB} MB."
                    ),
                )

            buffer.write(chunk)

    print()
    print("=" * 60, flush=True)
    print("PDF UPLOADED", flush=True)
    print("=" * 60, flush=True)
    print(file.filename, flush=True)
    print(flush=True)

    #
    # Run pipeline in the background -- this can take several
    # minutes (layout detection, OCR, Gemini calls per page), so the
    # request returns immediately with a job_id the frontend polls
    # via GET /upload/status/{job_id} instead of blocking the whole
    # upload request until the pipeline finishes.
    #

    job_id = upload_jobs.create_job()

    def _run_pipeline():

        def _on_progress(**fields):
            upload_jobs.update_job(job_id, **fields)

        try:

            # Constructing PipelineService loads the layout/OCR models and
            # the configured LLM client(s) -- any failure here (missing
            # API key, model load error) must be caught the same as a
            # failure during process_pdf, otherwise the job is left
            # "processing" forever with no error ever surfaced to the
            # frontend poller.
            pipeline = PipelineService()

            pipeline.process_pdf(
                pdf_path=str(file_path),
                progress_callback=_on_progress,
            )

        except APIError as exc:

            # 429/503 are retry-worthy conditions on Google's side (quota
            # exhaustion / rate limiting / temporary overload), not a bug
            # in this pipeline -- surface that distinction, with Google's
            # own suggested retry delay when it's provided, instead of an
            # opaque failure.
            if exc.code in (429, 503):
                message = _gemini_error_detail(exc)
            else:
                message = f"The AI service rejected the request: {exc}"

            upload_jobs.update_job(job_id, status="failed", error=message)

        except OpenAIStatusError as exc:

            # 429/5xx are retry-worthy conditions on OpenAI's side (quota
            # exhaustion / rate limiting / temporary overload), not a bug
            # in this pipeline -- surface that distinction instead of an
            # opaque failure.
            if exc.status_code in (429, 500, 502, 503, 504):
                message = _openai_error_detail(exc)
            else:
                message = f"The AI service rejected the request: {exc}"

            upload_jobs.update_job(job_id, status="failed", error=message)

        except (httpx.TransportError, OpenAIConnectionError) as exc:

            # Connection drops/resets/timeouts talking to the AI
            # provider's servers are retried internally already (see
            # GeminiService._call / OpenAIService._call); reaching here
            # means retries were exhausted -- still a network hiccup,
            # not a bug.
            upload_jobs.update_job(
                job_id,
                status="failed",
                error=(
                    "Could not reach the AI service (network error/"
                    f"connection dropped: {exc}). Please try uploading "
                    "again shortly."
                ),
            )

        except Exception as exc:

            print(f"\n[PIPELINE ERROR] Pipeline failed while processing '{file.filename}': {exc}", flush=True)
            traceback.print_exc()

            upload_jobs.update_job(
                job_id,
                status="failed",
                error=(
                    f"Pipeline failed while processing "
                    f"'{file.filename}': {exc}"
                ),
            )

    threading.Thread(target=_run_pipeline, daemon=True).start()

    #
    # Response
    #

    return {

        "success": True,

        "filename": file.filename,

        "job_id": job_id,

    }


@router.get("/upload/status/{job_id}")
def upload_status(job_id: str):

    job = upload_jobs.get_job(job_id)

    if job is None:

        raise HTTPException(
            status_code=404,
            detail=f"Unknown upload job: {job_id}",
        )

    return job