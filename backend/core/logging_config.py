import logging
import sys

from backend.core.config import settings


class _SuppressUploadStatusPolling(logging.Filter):
    """
    The frontend polls GET /upload/status/{job_id} every few seconds
    while a document is processing -- that's expected, repeated
    traffic, not worth a log line every time (unlike every other
    request, which is still logged normally).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "/upload/status/" not in record.getMessage()


def configure_logging() -> None:

    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )

    logging.getLogger("uvicorn.access").addFilter(
        _SuppressUploadStatusPolling()
    )
