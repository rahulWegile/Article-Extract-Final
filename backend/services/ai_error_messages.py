"""Human-friendly error text for LLM provider failures (Gemini/OpenAI).

Shared by the upload route (job-status polling) and the pipeline service
(document.json persistence) so a 429/5xx from either provider always
surfaces the same friendly wording instead of a raw SDK error dump --
regardless of which of the two places catches it first.
"""

from google.genai.errors import APIError
from openai import APIStatusError as OpenAIStatusError


def retry_delay_seconds(exc: APIError) -> str | None:
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


def gemini_error_detail(exc: APIError) -> str:

    retry_delay = retry_delay_seconds(exc)

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


def openai_error_detail(exc: OpenAIStatusError) -> str:

    retry_after = None

    try:
        header_value = exc.response.headers.get("retry-after")
        if header_value:
            retry_after = max(0.0, float(header_value))
    except Exception:
        retry_after = None

    if exc.status_code == 429:

        wait_clause = f" Please retry in {retry_after}s." if retry_after else ""

        return (
            "The OpenAI API quota/rate limit has been exhausted for this "
            f"API key/plan.{wait_clause} Check your OpenAI usage dashboard "
            "or upgrade your plan if this happens often."
        )

    wait_clause = f" Please try again in {retry_after}s." if retry_after else (
        " Please try uploading again in a few minutes."
    )

    return f"The AI service is temporarily unavailable (high demand).{wait_clause}"
