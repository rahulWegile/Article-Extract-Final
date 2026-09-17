"""
Resolves a document's detected language (metadata["language"]) to its
LanguagePipeline. Replaces the old chain of
PipelineService._is_<language>_language()/_get_ocr_engine_for_language
checks with one explicit lookup.

Iteration order matches the old _get_ocr_engine_for_language chain
exactly (backend/services/pipeline_service.py, pre-refactor lines
517-610): hindi, gujarati, marathi, tamil, telugu, kannada, malayalam,
punjabi, bengali, assamese, odia -- plus urdu, added after that
chain was retired. Any language that matches none
of these -- including an empty/unrecognized string, exactly like
before -- falls back to English, same as the old fallthrough
`return self.ocr_engine`.
"""

from pipeline.languages.english import ENGLISH
from pipeline.languages.hindi import HINDI
from pipeline.languages.gujarati import GUJARATI
from pipeline.languages.marathi import MARATHI
from pipeline.languages.tamil import TAMIL
from pipeline.languages.telugu import TELUGU
from pipeline.languages.kannada import KANNADA
from pipeline.languages.malayalam import MALAYALAM
from pipeline.languages.punjabi import PUNJABI
from pipeline.languages.bengali import BENGALI
from pipeline.languages.assamese import ASSAMESE
from pipeline.languages.odia import ODIA
from pipeline.languages.urdu import URDU

_NON_DEFAULT_PIPELINES = (
    ENGLISH,
    HINDI,
    GUJARATI,
    MARATHI,
    TAMIL,
    TELUGU,
    KANNADA,
    MALAYALAM,
    PUNJABI,
    BENGALI,
    ASSAMESE,
    ODIA,
    URDU,
)


def resolve_language_pipeline(language: str):
    for language_pipeline in _NON_DEFAULT_PIPELINES:
        if language_pipeline.matches(language):
            return language_pipeline

    return HINDI
