from pipeline.gemini.gemini_service import GeminiService

from pipeline.gemini.newspaper_metadata_prompt import (
    NEWSPAPER_METADATA_PROMPT,
)


class NewspaperClient:
    """
    Handles newspaper metadata extraction using Gemini.
    """

    def __init__(self):
        self.service = GeminiService()

    # -------------------------------------------------
    # Newspaper Metadata
    # -------------------------------------------------

    def extract_metadata(
        self,
        image_path: str,
    ):
        """
        Extract newspaper metadata from the first page.

        Returns metadata such as:
        - newspaper_name
        - edition
        - publish_date
        - language
        """

        return self.service.analyze_image(
            image_path=image_path,
            prompt=NEWSPAPER_METADATA_PROMPT,
        )