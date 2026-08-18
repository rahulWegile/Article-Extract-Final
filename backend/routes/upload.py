from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File
from google.genai.errors import APIError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APIStatusError as OpenAIStatusError

from backend.core.config import settings
from backend.services.pipeline_service import PipelineService

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

_UPLOAD_CHUNK_SIZE = 1024 * 1024


def _retry_delay_seconds(exc: APIError) -> str | None:
    """Extract Google's own suggested retryDelay (e.g. '54s') from the
    error payload, if present, so the message we show is accurate
    instead of a vague 'a few minutes'."""

    try:
        violations = exc.details.get("error", {}).get("details", [])
        for item in violations:
            if item.get("@type", "").endswith("RetryInfo"):
                return item.get("retryDelay")
    except Exception:
        pass

    return None


def _gemini_error_detail(exc: APIError) -> str:

    retry_delay = _retry_delay_seconds(exc)

    if exc.status == "RESOURCE_EXHAUSTED":

        wait_clause = f" Please retry in {retry_delay}." if retry_delay else ""

        return (
            "The Gemini API quota has been exhausted for this API key/plan "
            f"(free-tier limits are low; e.g. 20 requests/day for some "
            f"models).{wait_clause} Check https://ai.dev/rate-limit or "
            "upgrade your plan if this happens often."
        )

    wait_clause = f" Please try again in {retry_delay}." if retry_delay else (
        " Please try uploading again in a few minutes."
    )

    return f"The AI service is temporarily unavailable (high demand).{wait_clause}"


def _openai_error_detail(exc: OpenAIStatusError) -> str:

    if exc.status_code == 429:

        return (
            "The OpenAI API quota/rate limit has been reached for this "
            "API key/plan. Please wait a moment and try again, or check "
            "https://platform.openai.com/account/limits."
        )

    return (
        "The AI service is temporarily unavailable (high demand). "
        "Please try uploading again in a few minutes."
    )


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
    # Run pipeline
    #

    pipeline = PipelineService()

    try:

        pipeline.process_pdf(
            pdf_path=str(file_path)
        )

    except APIError as exc:

        # 429/503 are retry-worthy conditions on Google's side (quota
        # exhaustion / rate limiting / temporary overload), not a bug
        # in this pipeline -- surface that distinction, with Google's
        # own suggested retry delay when it's provided, instead of an
        # opaque 500.
        if exc.code in (429, 503):

            raise HTTPException(
                status_code=503,
                detail=_gemini_error_detail(exc),
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=f"The AI service rejected the request: {exc}",
        ) from exc

    except OpenAIStatusError as exc:

        # Same idea as the Gemini APIError handling above, but for the
        # OpenAI-backed engine (the default ARTICLE_EXTRACTOR_ENGINE).
        if exc.status_code in (429, 500, 502, 503, 504):

            raise HTTPException(
                status_code=503,
                detail=_openai_error_detail(exc),
            ) from exc

        raise HTTPException(
            status_code=502,
            detail=f"The AI service rejected the request: {exc}",
        ) from exc

    except OpenAIConnectionError as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not reach the AI service (network error/connection "
                f"dropped: {exc}). Please try uploading again shortly."
            ),
        ) from exc

    except httpx.TransportError as exc:

        # Connection drops/resets/timeouts talking to Google's servers
        # (e.g. "Server disconnected without sending a response") are
        # retried internally already (see GeminiService._generate);
        # reaching here means retries were exhausted -- still a
        # network hiccup, not a bug, so treat it like the API-busy case.
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not reach the AI service (network error/connection "
                f"dropped: {exc}). Please try uploading again shortly."
            ),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed while processing '{file.filename}': {exc}",
        ) from exc

    #
    # Response
    #

    return {

        "success": True,

        "filename": file.filename,

        "message": "Pipeline completed successfully."

    }