import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from backend.services.workspace_manager import WorkspaceManager
from backend.services.document_manager import DocumentManager
from backend.services.ai_error_messages import (
    gemini_error_detail,
    openai_error_detail,
)

from pipeline.render_pdf import render_pdf
from pipeline.layout_detector import LayoutDetector

# ============================================================
# RAPIDOCR
# ============================================================

from pipeline.ocr.rapidocr_engine import RapidOCREngine

from pipeline.languages.registry import resolve_language_pipeline

from pipeline.page_processor_gemini import (
    prepare_page,
    run_gemini,
    run_openai,
    run_local,
    finish_page,
)

from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.openai.newspaper_client import (
    NewspaperClient as OpenAINewspaperClient,
)

from pipeline.intelligence.local.masthead.masthead_extractor import (
    LocalMastheadExtractor,
    build_printed_page_map,
)

# ============================================================
# NEW PIPELINE STAGES
# ============================================================

from pipeline.intelligence.local.local_article_extractor import (
    LocalArticleExtractor,
)

from pipeline.finalization.logical_article_builder import (
    LogicalArticleBuilder,
)

from pipeline.database.import_final_articles import (
    import_document_directory,
)

from backend.services.urdu_pipeline_service import UrduPipelineService


class PipelineService:
    """
    Runs the complete newspaper article processing pipeline.

    FINAL ARCHITECTURE:

        PDF
          ↓
        Render pages
          ↓
        Newspaper metadata
          ↓
        Save document metadata
          ↓
        Layout detection
          ↓
        RapidOCR
          ↓
        Page Cleaner
          ↓
        Page-level Gemini
          ↓
        Gemini final boundaries
          ↓
        Exact final article crops
          ↓
        Article-level Gemini
          ↓
        Continuation resolution
          ↓
        Logical article finalization
          ↓
        Image extraction
          ↓
        Image compositor
          ↓
        final_articles.json
          ↓
        PostgreSQL database
          ↓
        Frontend


    IMPORTANT:

    The OLD Gemini Candidate Cleaner and
    NewspaperIntelligence pipeline are NOT used.

    We intentionally do NOT call:

        CandidateCleaner
        NewspaperIntelligence

    after final article crops.

    The page-level Gemini inside
    page_processor_gemini.py remains part of the
    existing boundary-detection pipeline.

    The newer GeminiArticleExtractor is then used
    AFTER final article crops have been created.

    The newer article-level Gemini performs:

        - article text
        - headline
        - author
        - article date
        - summary
        - category
        - entities
        - topics
        - keywords
        - sentiment
        - continuation detection
        - image information

    It uses the existing batch/pending architecture.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(self):

        print()
        print("=" * 60)
        print("LOADING PIPELINE MODELS")
        print("=" * 60)

        # =====================================================
        # Layout Detector (lazy -- see the `detector` property
        # below; skips loading DocLayout-YOLO into VRAM for
        # documents that route to a dedicated pipeline, e.g. Urdu)
        # =====================================================

        self._detector = None

        # =====================================================
        # RapidOCR
        # =====================================================

        self.ocr_engine = RapidOCREngine()

        # Lazily-created, cached per-language OCR engine instances,
        # keyed by LanguagePipeline.code (see
        # pipeline/languages/registry.py). Kept separate per language
        # so switching one language's document to its engine never
        # affects another language processed by this same (long-
        # lived) PipelineService instance. Pre-seeded with "english"
        # so English documents reuse this exact self.ocr_engine
        # instance (also used by the local masthead extractor below)
        # instead of EnglishPipeline.ocr_engine_factory building a
        # second, redundant one.
        self._ocr_engine_cache = {

        }

        # Lazily constructed on the first Urdu document -- avoids
        # loading yolov8m_UrduDoc.pt/UTRNet on every process start for
        # documents that never turn out to be Urdu.
        self._urdu_pipeline_service = None

        # =====================================================
        # Newspaper Metadata Client
        # =====================================================

        self.llm_provider = (
            os.getenv("LLM_PROVIDER", "openai")
            .strip()
            .lower()
        )

        # Local mode must not require an API key at all, so the
        # network client is simply never constructed.
        #
        # Newspaper masthead metadata (name/edition/date/language)
        # always goes through OpenAI, regardless of LLM_PROVIDER or
        # the document's language -- this call is what DETERMINES
        # the document's language in the first place, so there is
        # nothing to route on yet at this point. The per-document
        # provider choice below (see
        # pipeline/languages/registry.py) only applies to boundary
        # detection and article extraction, both of which run after
        # the language is already known.
        if self.llm_provider == "local":
            self.newspaper_client = None
        else:
            self.newspaper_client = OpenAINewspaperClient()

        self.local_masthead_extractor = (
            LocalMastheadExtractor(
                ocr_engine=self.ocr_engine,
            )
        )

        print(
            "✓ Layout Detector Loaded"
        )

        print(
            "✓ RapidOCR Loaded"
        )

        print(
            "✓ Newspaper Client Loaded"
        )

        print(
            "✓ Local Masthead Extractor Loaded "
            f"({len(self.local_masthead_extractor.templates)} templates)"
        )

    @property
    def detector(self):
        if self._detector is None:
            self._detector = LayoutDetector()
        return self._detector

    def _detect_script_locally(self, page_image_path):
        """
        Cheap Devanagari-vs-Latin script probe for local mode.

        There is no LLM here to name the language outright, so both
        the default (Latin/English) and the Devanagari OCR engines
        are run on the same small, text-dense crop of the page, and
        whichever gets meaningfully higher average recognition
        confidence wins. A wrong-script recognizer reliably produces
        low-confidence noise, which is what makes this comparison
        work without understanding the text itself.

        Runs once per document, and only when local mode could not
        otherwise identify the language -- the one-time cost of
        loading the Devanagari model is paid only when it might
        actually be needed.

        Returns "Hindi", or None to leave the language as-is.
        """

        import cv2

        try:
            image = cv2.imread(str(page_image_path))
        except Exception:
            return None

        if image is None:
            return None

        height, width = image.shape[:2]

        # A central, text-dense band: skips the masthead/graphics
        # usually at the very top of the page.
        y1 = int(height * 0.15)
        y2 = int(height * 0.45)

        crop = image[y1:y2, :]

        def average_confidence(engine):

            try:
                result = engine.reader(crop)
            except Exception:
                return 0.0

            scores = getattr(result, "scores", None) if result else None

            if not scores:
                return 0.0

            scores = [s for s in scores if s is not None]

            return sum(scores) / len(scores) if scores else 0.0

        latin_confidence = average_confidence(self.ocr_engine)

        if "hindi" not in self._ocr_engine_cache:
            self._ocr_engine_cache["hindi"] = (
                resolve_language_pipeline(
                    "hindi"
                ).ocr_engine_factory()
            )

        hindi_confidence = average_confidence(
            self._ocr_engine_cache["hindi"]
        )

        print(
            "  Script probe -- latin confidence: "
            f"{latin_confidence:.2f}, devanagari confidence: "
            f"{hindi_confidence:.2f}"
        )

        # A clear margin is required so a genuinely English/mixed
        # page is not switched to Devanagari over ordinary noise.
        if (
            hindi_confidence > latin_confidence + 0.05
            and hindi_confidence > 0.3
        ):
            return "Hindi"

        return None

    @staticmethod
    def _clean_metadata_value(value):
        """
        Convert metadata values to clean strings.

        Empty values become None.
        """

        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value

    # --------------------------------------------------------
    # Filename metadata fallback
    # --------------------------------------------------------

    @staticmethod
    def _metadata_from_filename(pdf_path):
        """
        Fallback metadata extraction from newspaper filename.

        Expected examples:

            Times of India_Chandigarh_20260804.pdf

            The Asian Age_Delhi_20260714.pdf

        This is only a fallback when Gemini metadata extraction
        does not return usable values.
        """

        name = Path(pdf_path).stem

        # Remove repeated "_removed" suffixes.
        name = re.sub(
            r"(?:_removed)+$",
            "",
            name,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace.
        name = re.sub(
            r"\s+",
            " ",
            name,
        ).strip()

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        date_match = re.search(
            r"(20\d{6})",
            name,
        )

        publish_date = None

        if date_match:

            raw_date = (
                date_match.group(1)
            )

            try:

                parsed = datetime.strptime(
                    raw_date,
                    "%Y%m%d",
                )

                publish_date = (
                    parsed.date().isoformat()
                )

            except ValueError:

                publish_date = None

        # ----------------------------------------------------
        # Remove date from filename
        # ----------------------------------------------------

        without_date = re.sub(
            r"_?20\d{6}",
            "",
            name,
        )

        # ----------------------------------------------------
        # Remove trailing separators
        # ----------------------------------------------------

        without_date = (
            without_date
            .strip("_ ")
        )

        # ----------------------------------------------------
        # Split newspaper and edition
        # ----------------------------------------------------

        parts = [
            p.strip()
            for p in without_date.split("_")
            if p.strip()
        ]

        newspaper_name = None
        edition = None

        if len(parts) >= 2:

            newspaper_name = (
                parts[0]
            )

            edition = (
                " ".join(parts[1:])
            )

        elif len(parts) == 1:

            newspaper_name = parts[0]

        # ----------------------------------------------------
        # Normalize known newspaper names
        # ----------------------------------------------------

        if newspaper_name:

            normalized = (
                newspaper_name
                .strip()
                .lower()
            )

            known_names = {
                "times of india":
                    "The Times of India",

                "the times of india":
                    "The Times of India",

                "asian age":
                    "The Asian Age",

                "the asian age":
                    "The Asian Age",
            }

            newspaper_name = (
                known_names.get(
                    normalized,
                    newspaper_name,
                )
            )

        # ----------------------------------------------------
        # Language hint
        #
        # Urdu's Perso-Arabic script is the case most prone to
        # silently defaulting to English (see
        # _build_document_metadata) when masthead-based language
        # detection is unresolved or wrong, so a filename that
        # names Urdu outright or a known Urdu masthead is checked
        # here as a strong, cheap signal. "siasat" covers The
        # Siasat Daily (uploads/siasat-daily-*.pdf), a real Urdu
        # paper whose filename contains neither "urdu" nor
        # "inqilab"/"inquilab".
        # ----------------------------------------------------

        filename_lower = name.lower()

        urdu_filename_hints = (
            "urdu",
            "siasat",
            "inqilab",
            "inquilab",
        )

        language = None

        if any(
            hint in filename_lower
            for hint in urdu_filename_hints
        ):

            language = "Urdu"

        return {
            "newspaper_name":
                newspaper_name,

            "edition":
                edition,

            "publish_date":
                publish_date,

            "language":
                language,
        }

    # --------------------------------------------------------
    # Build reliable document metadata
    # --------------------------------------------------------

    def _build_document_metadata(
        self,
        pdf_path,
        extracted_metadata,
    ):
        """
        Build final document metadata.

        Priority:

            1. Gemini metadata
            2. Filename fallback

        This prevents empty newspaper_name/publish_date
        from silently reaching PostgreSQL.
        """

        if not isinstance(
            extracted_metadata,
            dict,
        ):

            extracted_metadata = {}

        filename_metadata = (
            self._metadata_from_filename(
                pdf_path
            )
        )

        newspaper_name = (
            self._clean_metadata_value(
                extracted_metadata.get(
                    "newspaper_name"
                )
            )
        )

        edition = (
            self._clean_metadata_value(
                extracted_metadata.get(
                    "edition"
                )
            )
        )

        publish_date = (
            self._clean_metadata_value(
                extracted_metadata.get(
                    "publish_date"
                )
            )
        )

        language = (
            self._clean_metadata_value(
                extracted_metadata.get(
                    "language"
                )
            )
        )

        # ----------------------------------------------------
        # Fallbacks
        # ----------------------------------------------------

        if not newspaper_name:

            newspaper_name = (
                filename_metadata.get(
                    "newspaper_name"
                )
            )

        if not edition:

            edition = (
                filename_metadata.get(
                    "edition"
                )
            )

        if not publish_date:

            publish_date = (
                filename_metadata.get(
                    "publish_date"
                )
            )

        if not language:

            language = filename_metadata.get(
                "language"
            )

        if not language:

            language = "Hindi"

        # ----------------------------------------------------
        # Urdu filename override
        #
        # A filename that names Urdu outright or a known Urdu
        # masthead (e.g. "inqilab", "siasat") is a stronger signal
        # than a language value that is still unresolved or was
        # misclassified as English -- Urdu's Perso-Arabic script
        # is the case most prone to that failure. Only overrides
        # when the resolved language isn't already Urdu, so a
        # correct non-Urdu classification is never touched.
        # ----------------------------------------------------

        if (
            filename_metadata.get("language") == "Urdu"
            and resolve_language_pipeline(language).code != "urdu"
        ):

            language = "Urdu"

        return {
            "newspaper_name":
                newspaper_name or "",

            "edition":
                edition or "",

            "publish_date":
                publish_date or "",

            "language":
                language,
        }

    # --------------------------------------------------------
    # Save document metadata
    # --------------------------------------------------------

    @staticmethod
    def _save_document_metadata(
        document_dir,
        metadata,
        page_count,
        article_count=None,
        final_article_count=None,
        status=None,
        gemini_status=None,
        finalization_status=None,
        database_status=None,
        boundaries_ready=None,
    ):
        """
        Write document.json.

        This method is intentionally called BEFORE database
        import so that import_final_articles.py sees the
        correct newspaper metadata.
        """

        document_dir = Path(
            document_dir
        )

        metadata_file = (
            document_dir / "document.json"
        )

        if metadata_file.exists():

            with open(
                metadata_file,
                "r",
                encoding="utf-8",
            ) as f:

                document_metadata = (
                    json.load(f)
                )

        else:

            document_metadata = {}

        # ----------------------------------------------------
        # Basic metadata
        # ----------------------------------------------------

        document_metadata[
            "page_count"
        ] = page_count

        document_metadata[
            "newspaper_name"
        ] = (
            metadata.get(
                "newspaper_name",
                "",
            )
            or ""
        )

        document_metadata[
            "edition"
        ] = (
            metadata.get(
                "edition",
                "",
            )
            or ""
        )

        document_metadata[
            "publish_date"
        ] = (
            metadata.get(
                "publish_date",
                "",
            )
            or ""
        )

        document_metadata[
            "language"
        ] = (
            metadata.get(
                "language",
                "",
            )
            or ""
        )

        if metadata.get("metadata_source") is not None:

            document_metadata[
                "metadata_source"
            ] = metadata.get(
                "metadata_source"
            )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        if article_count is not None:

            document_metadata[
                "article_count"
            ] = article_count

            document_metadata[
                "final_article_crops"
            ] = article_count

        if final_article_count is not None:

            document_metadata[
                "final_article_count"
            ] = final_article_count

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if status is not None:

            document_metadata[
                "status"
            ] = status

        if gemini_status is not None:

            document_metadata[
                "gemini_article_extraction"
            ] = gemini_status

        if finalization_status is not None:

            document_metadata[
                "finalization"
            ] = finalization_status

        if database_status is not None:

            document_metadata[
                "database_import"
            ] = database_status

        if boundaries_ready is not None:

            document_metadata[
                "boundaries_ready"
            ] = boundaries_ready

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        with open(
            metadata_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                document_metadata,
                f,
                indent=4,
                ensure_ascii=False,
            )

        return document_metadata

    # ========================================================
    # MARK DOCUMENT FAILED
    # ========================================================

    @staticmethod
    def _mark_document_failed(document_dir, error: Exception):
        """
        Minimal, targeted document.json update on pipeline failure.

        Deliberately does NOT go through _save_document_metadata: that
        method re-derives every field (newspaper_name, page_count, ...)
        from a freshly-passed `metadata` dict, which we don't have here
        and which would blank out fields already saved earlier in the
        run. This only sets/overwrites the status + error fields.
        """

        metadata_file = Path(document_dir) / "document.json"

        if not metadata_file.exists():
            return

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                document_metadata = json.load(f)
        except Exception:
            document_metadata = {}

        # A 429/503 from the LLM provider's API is a retry-worthy
        # condition on the provider's side (quota exhaustion / rate
        # limiting / temporary overload), not a bug in this pipeline;
        # surface that distinction (plus the provider's own error
        # `status`, e.g. RESOURCE_EXHAUSTED vs UNAVAILABLE) so the
        # frontend/operator knows re-uploading is likely to work
        # without any code change. Covers both Google's genai SDK
        # exception names and the OpenAI SDK's status-code attribute.
        error_type_name = type(error).__name__
        status_code = getattr(error, "code", None)
        is_gemini_api_error = error_type_name in {
            "ServerError", "ClientError", "APIError",
        }
        is_openai_api_error = error_type_name in {
            "APIStatusError", "APIConnectionError", "APITimeoutError",
        }
        openai_status_code = getattr(error, "status_code", None)

        provider_unavailable = (
            (is_gemini_api_error and status_code in (429, 503))
            or (
                is_openai_api_error
                and (
                    openai_status_code in (429, 500, 502, 503, 504)
                    or error_type_name in {"APIConnectionError", "APITimeoutError"}
                )
            )
        )

        # Never surface the raw SDK error (a multi-line dict dump for
        # Gemini's 429s) to the frontend -- run it through the same
        # friendly formatter used for the upload job's status message
        # so both surfaces agree and the operator sees plain wording.
        if is_gemini_api_error and status_code in (429, 503):
            message = gemini_error_detail(error)
        elif is_openai_api_error and openai_status_code in (429, 500, 502, 503, 504):
            message = openai_error_detail(error)
        elif provider_unavailable:
            message = "The AI service is temporarily unavailable. Please try uploading again in a few minutes."
        else:
            message = str(error)

        document_metadata["status"] = "failed"
        document_metadata["error"] = {
            "type": error_type_name,
            "message": message,
            "transient": provider_unavailable,
            "api_status": (
                getattr(error, "status", None) if is_gemini_api_error
                else openai_status_code if is_openai_api_error
                else None
            ),
        }

        crop_dir = document_dir / "final_articles_crops"
        if crop_dir.is_dir() and any(crop_dir.iterdir()):
            document_metadata["boundaries_ready"] = True

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(document_metadata, f, indent=4, ensure_ascii=False)

    # ========================================================
    # PROCESS PDF
    # ========================================================

    def process_pdf(
        self,
        pdf_path,
        progress_callback=None,
    ):

        def _report(stage, percent, **extra):

            if progress_callback is None:
                return

            try:
                progress_callback(stage=stage, progress=percent, **extra)
            except Exception:
                # Progress reporting must never break the pipeline itself.
                pass

        print()
        print("=" * 60)
        print("PROCESSING PDF")
        print("=" * 60)

        _report("Starting", 1)

        # =====================================================
        # Temporary Workspace
        # =====================================================

        # Held in a variable so the cleanup in `finally` removes THIS
        # run's workspace specifically, rather than a fresh instance
        # wiping a concurrently-running upload's workspace too.
        workspace_manager = WorkspaceManager()

        workspace = (
            workspace_manager.create()
        )

        print(
            f"Workspace : {workspace}"
        )

        # Bound up front so the except/finally below can reference it
        # even if the pipeline fails before STEP 3 (Create Document)
        # ever assigns it.
        document_dir = None

        try:

            # =================================================
            # STEP 1
            # Render PDF
            # =================================================

            print()
            print("=" * 60)
            print("PDF RENDERING")
            print("=" * 60)

            _report("Rendering PDF", 2)

            pages = render_pdf(
                pdf_path=pdf_path,
                output_dir=str(
                    workspace / "pages"
                ),
                dpi=300,
            )

            print(
                f"Rendered {len(pages)} pages"
            )

            if not pages:

                raise RuntimeError(
                    "No pages were rendered "
                    "from the PDF."
                )

            # =================================================
            # Printed page folio map
            #
            # A continuation marker such as "Continued from P 1"
            # quotes the PRINTED page number, which can disagree
            # with the physical PDF page index (e.g. an unnumbered
            # front jacket ad pushes the real front page a few PDF
            # pages in). This reads the running folio off every
            # page so continuation matching can translate between
            # the two. Best-effort and never blocking: an empty
            # map here just means continuation matching falls back
            # to treating printed numbers as physical page indices,
            # exactly as before this existed.
            # =================================================

            printed_page_map = {}

            try:

                printed_page_map = build_printed_page_map(
                    self.local_masthead_extractor,
                    pages,
                )

                if printed_page_map:

                    print(
                        f"Printed->PDF page map: {printed_page_map}"
                    )

            except Exception as exc:

                print(
                    f"⚠ Printed-page folio mapping failed: {exc}"
                )

                printed_page_map = {}

            # =================================================
            # STEP 2
            # Newspaper Metadata
            # =================================================

            print()
            print("=" * 60)
            print("NEWSPAPER METADATA")
            print("=" * 60)

            _report("Reading newspaper metadata", 8)

            metadata = None
            metadata_source = None

            local_metadata_enabled = (
                os.getenv(
                    "LOCAL_METADATA_EXTRACTOR_ENABLED",
                    "true",
                )
                .strip()
                .lower()
                != "false"
            )

            if local_metadata_enabled:

                try:

                    local_result = (
                        self.local_masthead_extractor.extract_metadata(
                            image_path=pages[0],
                        )
                    )

                    if local_result is not None:

                        metadata = local_result
                        metadata_source = "local"

                except Exception as exc:

                    print()
                    print(
                        f"⚠ Local masthead extraction errored: {exc}"
                    )

            if metadata is None and self.llm_provider == "local":

                # Fully-local mode makes no network calls at all, so
                # there is no LLM fallback to reach for here. An
                # unrecognised masthead simply stays unidentified
                # rather than silently contacting an API.
                print()
                print(
                    "Local mode: skipping LLM metadata fallback "
                    "(masthead not recognised locally)"
                )

                metadata = {}
                metadata_source = "local_unresolved"

            if metadata is None:

                try:

                    metadata = (
                        self.newspaper_client.extract_metadata(
                            image_path=pages[0],
                        )
                    )

                    # self.newspaper_client is always OpenAI's
                    # client here (see __init__) regardless of
                    # LLM_PROVIDER, so the source label must say so
                    # literally rather than echo self.llm_provider
                    # -- which could read "gemini" for a Hindi
                    # document even though metadata always comes
                    # from OpenAI.
                    metadata_source = "openai"

                    if local_metadata_enabled:

                        try:

                            self.local_masthead_extractor.record_gemini_result(
                                image_path=pages[0],
                                gemini_metadata=metadata,
                            )

                        except Exception as exc:

                            print()
                            print(
                                f"⚠ Local masthead learning errored: {exc}"
                            )

                except Exception as exc:

                    print()
                    print(
                        "⚠ Newspaper metadata extraction failed:"
                    )

                    print(
                        f"  {exc}"
                    )

                    print(
                        "  Using filename fallback."
                    )

                    metadata = {}
                    metadata_source = "filename_fallback"

            if not isinstance(
                metadata,
                dict,
            ):

                metadata = {}

            print(
                f"Metadata source : {metadata_source}"
            )

            # -------------------------------------------------
            # Build reliable metadata
            # -------------------------------------------------

            metadata = (
                self._build_document_metadata(
                    pdf_path=pdf_path,
                    extracted_metadata=metadata,
                )
            )

            metadata["metadata_source"] = metadata_source

            print(
                f"Newspaper : "
                f"{metadata.get('newspaper_name', '')}"
            )

            print(
                f"Edition   : "
                f"{metadata.get('edition', '')}"
            )

            print(
                f"Date      : "
                f"{metadata.get('publish_date', '')}"
            )

            print(
                f"Language  : "
                f"{metadata.get('language', '')}"
            )

            # -------------------------------------------------
            # LOCAL-MODE METADATA FALLBACK
            # -------------------------------------------------
            #
            # In local mode there is no LLM to identify an unknown
            # masthead, and a filename like "2 poage.pdf" carries
            # neither a paper name nor a date. Failing the whole
            # upload over unknown cover metadata would throw away
            # perfectly good article extraction, so the document is
            # accepted with clearly-marked placeholder values
            # instead. The API modes keep the hard failure, where
            # missing metadata really does signal a broken call.
            # -------------------------------------------------

            if self.llm_provider == "local":

                if not metadata.get("newspaper_name"):

                    metadata["newspaper_name"] = "Unknown"

                    print(
                        "⚠ Local mode: newspaper_name unknown "
                        "(masthead not in local templates)"
                    )

                if not metadata.get("publish_date"):

                    fallback_date = datetime.fromtimestamp(
                        Path(pdf_path).stat().st_mtime
                    ).strftime("%Y-%m-%d")

                    metadata["publish_date"] = fallback_date

                    print(
                        "⚠ Local mode: publish_date not found in "
                        "masthead or filename -- using the PDF's "
                        f"file date ({fallback_date}) as a "
                        "placeholder"
                    )

            # -------------------------------------------------
            # IMPORTANT SAFETY CHECK
            # -------------------------------------------------

            if not metadata.get(
                "newspaper_name"
            ):

                raise RuntimeError(
                    "Could not determine newspaper_name "
                    "from LLM metadata or filename."
                )

            if not metadata.get(
                "publish_date"
            ):

                raise RuntimeError(
                    "Could not determine publish_date "
                    "from LLM metadata or filename."
                )

            # -------------------------------------------------
            # LOCAL-MODE SCRIPT DETECTION
            #
            # _build_document_metadata defaults an unresolved
            # language to "English" so publish_date/newspaper_name
            # checks never see a blank value. With no LLM to name
            # the actual language, that default was silently
            # selecting the English/Latin OCR engine for every
            # Hindi document run in local mode -- OCR wasn't merely
            # inaccurate, it was reading Devanagari text with a
            # recognizer that has no Devanagari characters in its
            # vocabulary at all. A quick script probe on the
            # rendered page catches this before OCR runs for real.
            # -------------------------------------------------

            # Urdu is not one of this probe's two candidates (Latin vs
            # Devanagari), so it must never run once Urdu is already
            # resolved -- otherwise a Nastaliq page's Devanagari-probe
            # confidence can false-positive and silently swap the
            # correct UTRNet Urdu engine for the Hindi/Devanagari
            # RapidOCR engine.
            if (
                self.llm_provider == "local"
                and metadata.get("metadata_source")
                == "local_unresolved"
                and resolve_language_pipeline(
                    metadata.get("language", "")
                ).code != "urdu"
            ):

                detected_language = (
                    self._detect_script_locally(
                        pages[0]
                    )
                )

                if detected_language:

                    metadata["language"] = detected_language

                    print(
                        "Local mode: script probe detected "
                        f"{detected_language}"
                    )

            # -------------------------------------------------
            # Resolve this document's language pipeline (see
            # pipeline/languages/registry.py): decides which OCR
            # engine, LLM provider, grouping prompt/policy, and
            # article extractor this document uses, all in one
            # place instead of scattered per-language checks.
            # -------------------------------------------------

            lang_pipeline = resolve_language_pipeline(
                metadata.get("language", "")
            )

            # -------------------------------------------------
            # URDU: dedicated pipeline.
            #
            # Bypasses DocLayout-YOLO/PageCleaner/article_splitter/
            # sidebox_absorption entirely -- yolov8m_UrduDoc.pt
            # detects raw text regions directly, UTRNet reads them,
            # and a single Gemini call groups OCR region ids into
            # articles (Gemini never sees or returns pixel
            # coordinates). See backend/services/
            # urdu_pipeline_service.py.
            # -------------------------------------------------

            if lang_pipeline.code == "urdu":

                document = DocumentManager().create_document(pdf_path)
                document_id = document["document_id"]
                document_dir = Path(document["document_dir"])

                _report("Document created", 10, document_id=document_id)

                self._save_document_metadata(
                    document_dir=document_dir,
                    metadata=metadata,
                    page_count=len(pages),
                    status="processing",
                )

                destination_pages = document_dir / "pages"
                if destination_pages.exists():
                    shutil.rmtree(destination_pages)
                shutil.move(str(workspace / "pages"), str(destination_pages))

                moved_pages = [
                    destination_pages / f"page_{index:03d}.png"
                    for index in range(1, len(pages) + 1)
                ]

                if self._urdu_pipeline_service is None:
                    self._urdu_pipeline_service = UrduPipelineService()

                urdu_result = self._urdu_pipeline_service.process_document(
                    document_dir=document_dir,
                    pages=moved_pages,
                    metadata=metadata,
                    progress_callback=progress_callback,
                )

                final_article_crops = urdu_result.get("final_article_crops", [])

                self._save_document_metadata(
                    document_dir=document_dir,
                    metadata=metadata,
                    page_count=len(pages),
                    article_count=len(final_article_crops),
                    final_article_count=len(final_article_crops),
                    status="completed",
                    gemini_status="completed",
                    finalization_status="completed",
                    database_status="completed",
                    boundaries_ready=True,
                )

                _report("Completed", 100, document_id=document_id, status="completed")

                return {
                    "document_id": document_id,
                    "document_dir": document_dir,
                    "pages": moved_pages,
                    "metadata": metadata,
                    "final_article_crops": final_article_crops,
                    "status": "completed",
                }

            if lang_pipeline.code not in self._ocr_engine_cache:
                self._ocr_engine_cache[lang_pipeline.code] = (
                    lang_pipeline.ocr_engine_factory()
                )

            active_ocr_engine = (
                self._ocr_engine_cache[lang_pipeline.code]
            )

            print(
                f"OCR engine : {lang_pipeline.ocr_engine_label}"
            )

            # -------------------------------------------------
            # LLM ROUTING (boundary detection + article
            # extraction only -- newspaper metadata above always
            # uses OpenAI, see __init__)
            #
            # This is a PER-DOCUMENT decision based on the
            # language pipeline just resolved above, not the
            # global LLM_PROVIDER value -- LLM_PROVIDER=gemini/
            # openai no longer manually pins boundary/extraction
            # to one engine, only LLM_PROVIDER=local still does
            # (the explicit no-API-calls override, orthogonal to
            # language).
            # -------------------------------------------------

            if self.llm_provider == "local":
                document_llm_provider = "local"
            else:
                document_llm_provider = lang_pipeline.llm_provider

            print(
                f"LLM routing : {document_llm_provider} "
                f"(language={metadata.get('language', '')})"
            )

            # =================================================
            # STREAM B: DOCUMENT-ORDER EXTRACTION
            #
            # Fired now, off the raw rendered page images only --
            # no dependency on OCR/layout/boundary detection or
            # Stream A's article crops, so it runs concurrently with
            # the rest of the pipeline instead of waiting on it. Has
            # to wait for `document_llm_provider` specifically (just
            # determined above) so it can route to the SAME provider
            # as Stream A -- gpt-5.6-luna cannot read Devanagari, so
            # a Hindi document must use Gemini here too, exactly like
            # Stream A's boundary/extraction calls already do.
            #
            # Results are only collected and reconciled once Stream
            # A's article extraction (below) has finished -- nothing
            # here blocks anything else. Skipped entirely in local
            # mode (no API calls at all).
            #
            # Off by default (DOCUMENT_ORDER_RECONCILE=1 to enable):
            # this is a new, unvalidated accuracy layer with a real
            # per-document cost, not a bug fix -- see reconcile.py
            # for the merge policy (Stream A stays authoritative).
            #
            # Hindi-only: the reference English pipeline has no
            # Stream B at all, so English never runs it regardless
            # of the env flag -- only a Hindi-routed document can
            # enable it.
            # =================================================

            document_order_enabled = (
                document_llm_provider == "gemini"
                and lang_pipeline.document_order_extractor_factory
                is not None
                and os.getenv(
                    "DOCUMENT_ORDER_RECONCILE",
                    "0",
                )
                == "1"
            )

            document_order_executor = None
            document_order_futures = []

            if document_order_enabled:

                document_order_executor = (
                    ThreadPoolExecutor(
                        max_workers=int(
                            os.getenv(
                                "DOCUMENT_ORDER_CONCURRENCY",
                                "4",
                            )
                        ),
                    )
                )

                # document_order_enabled already requires
                # lang_pipeline.document_order_extractor_factory to
                # be set, which today only Hindi's LanguagePipeline
                # declares (English and the other languages have no
                # Stream B, matching the reference pipeline).
                document_order_futures = (
                    lang_pipeline.document_order_extractor_factory()
                    .submit_batches(
                        document_order_executor,
                        pages,
                    )
                )

                print(
                    f"Document-order extraction (Stream B, "
                    f"{document_llm_provider}): "
                    f"{len(document_order_futures)} batch(es) "
                    "submitted in background"
                )

            # =================================================
            # STEP 3
            # Create Document
            # =================================================

            print()
            print("=" * 60)
            print("CREATING DOCUMENT")
            print("=" * 60)

            document = (
                DocumentManager().create_document(
                    pdf_path
                )
            )

            document_id = (
                document["document_id"]
            )

            document_dir = Path(
                document["document_dir"]
            )

            print()
            print(
                f"Document ID : "
                f"{document_id}"
            )

            print(
                f"Document Dir: "
                f"{document_dir}"
            )

            _report(
                "Document created",
                10,
                document_id=document_id,
            )

            # =================================================
            # SAVE INITIAL DOCUMENT METADATA
            # =================================================

            print()
            print("=" * 60)
            print("SAVING INITIAL DOCUMENT METADATA")
            print("=" * 60)

            self._save_document_metadata(
                document_dir=document_dir,
                metadata=metadata,
                page_count=len(pages),
                status="processing",
            )

            print(
                "✓ Initial document metadata saved"
            )

            # =================================================
            # STEP 4
            # Create Output Directories
            # =================================================

            (
                document_dir / "page_json"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                document_dir / "llm_response"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            (
                document_dir / "final"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            # =================================================
            # STEP 5
            # Process Every Page
            # =================================================

            print()
            print("=" * 60)
            print("PAGE PROCESSING")
            print("=" * 60)

            page_results = {}

            all_final_article_crops = []

            # ---------------------------------------------
            # PAGE-BY-PAGE ARTICLE EXTRACTION SETUP
            #
            # Article-level extraction now runs immediately after
            # EACH page's crops are written below (batch size 1),
            # instead of waiting for every page to finish and then
            # running one extraction pass over the whole document.
            # `pending_continuations` carries forward page-to-page
            # exactly like process_document()'s own per-batch loop
            # already does -- process_batch() is called directly,
            # one page at a time, instead of via process_document().
            #
            # Excluded when:
            #   - document_llm_provider == "local" (LocalArticleExtractor
            #     has its own, unrelated batching -- untouched).
            #   - document_order_enabled (Hindi's Stream B heading-repair
            #     mutates already-written crops and MUST fully finish,
            #     across every page, before any extraction call reads
            #     them -- see STEP 11.5 below. Interleaving would let an
            #     early page's extraction race ahead of its own repair,
            #     so these documents keep the original all-pages-then-
            #     extract-once flow at STEP 12 instead.)
            # ---------------------------------------------

            interleave_extraction = (
                document_llm_provider != "local"
                and not document_order_enabled
            )

            article_extractor = None
            all_extracted_articles = []
            all_continuation_links = []
            pending_continuations = []
            page_extraction_results = []
            gemini_batch_dir = document_dir / "gemini_article_batches"

            # Two-phase flow: Phase 1 (the per-page loop below) runs
            # boundary creation + cropping for every page, continuously,
            # without pausing for extraction -- page_crop_counts just
            # records how many crops each page produced. Phase 2 (after
            # the loop, in STEP 12) then batches extraction across the
            # WHOLE document using the exact same
            # extraction_pages_per_batch / extraction_max_articles_per_batch
            # rules, just applied once at the end instead of per-page.
            page_crop_counts = {}

            # Per-page article IDs (in the same order _discover_articles
            # returns them), recorded alongside page_crop_counts. Phase 2
            # uses this to split a single oversized page (more crops
            # than extraction_max_articles_per_batch) into several
            # sub-batches instead of shipping it as one oversized call.
            page_crop_article_ids = {}

            if interleave_extraction:

                gemini_batch_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                article_extractor = lang_pipeline.extractor_class(
                    pages_per_batch=(
                        lang_pipeline.extraction_pages_per_batch
                    ),
                    prompt_template=(
                        lang_pipeline.extraction_prompt_template
                    ),
                )

            if interleave_extraction:

                # ---------------------------------------------
                # PHASE 1: PAGE-BY-PAGE BOUNDARY + CROP PIPELINE
                #
                # Prepare (layout/OCR), boundary grouping, and finish/
                # crop run for page N -- fully -- before page N+1
                # even starts preparing (1-page lookahead prefetch:
                # page N+1's prepare_page runs in a background thread
                # while page N's boundary-grouping call is in flight,
                # so the CPU/GPU is never idle waiting on the network).
                # No concurrency across pages beyond that lookahead, no
                # document-wide barrier for boundaries themselves.
                #
                # Article-level extraction is NOT run here -- it only
                # records each page's crop count (page_crop_counts) and
                # runs afterward, once, as PHASE 2 in STEP 12 below,
                # once every page in the document has finished this
                # loop.
                #
                # Only used when interleave_extraction is True (see
                # the setup above STEP 5) -- local mode and Stream B
                # (document_order_enabled) documents keep the exact
                # pipelined approach in the `else` branch below.
                # ---------------------------------------------

                if document_llm_provider == "local":
                    run_page_llm = run_local
                elif document_llm_provider == "gemini":
                    run_page_llm = run_gemini
                else:
                    run_page_llm = run_openai

                print()
                print("=" * 60)
                print(
                    "STRICT PAGE-BY-PAGE PIPELINE "
                    f"({document_llm_provider.upper()})"
                )
                print("=" * 60)

                with ThreadPoolExecutor(
                    max_workers=1,
                ) as prepare_executor:

                    # 1-page lookahead: page N+1's prepare_page (YOLO +
                    # OCR, CPU/GPU-bound) runs in this one background
                    # thread while the main thread runs page N's network
                    # calls (run_page_llm, article_extractor.process_batch)
                    # below -- the CPU/GPU is never idle waiting on a
                    # cloud API response. Bounded to exactly one page
                    # ahead (max_workers=1, one future in flight at a
                    # time) so memory usage stays fixed regardless of
                    # document length.

                    next_prep_future = prepare_executor.submit(
                        prepare_page,
                        page_number=1,
                        page_path=pages[0],
                        detector=self.detector,
                        ocr_engine=active_ocr_engine,
                        document_id=document_id,
                        document_dir=document_dir,
                        is_rtl=lang_pipeline.is_rtl,
                        layout_confidence=lang_pipeline.layout_confidence,
                        ocr_engine_label=lang_pipeline.ocr_engine_label,
                    )

                    for page_idx, page_path in enumerate(
                        pages,
                    ):

                        page_number = page_idx + 1

                        print()
                        print("=" * 60)
                        print(
                            f"PAGE {page_number}/{len(pages)}"
                        )
                        print("=" * 60)

                        _report(
                            f"Processing page {page_number}/{len(pages)}",
                            11 + round(54 * (page_number - 1) / len(pages)),
                        )

                        prep = next_prep_future.result()

                        if page_idx + 1 < len(pages):

                            next_page_number = page_number + 1

                            next_page_path = pages[page_idx + 1]

                            next_prep_future = (
                                prepare_executor.submit(
                                    prepare_page,
                                    page_number=next_page_number,
                                    page_path=next_page_path,
                                    detector=self.detector,
                                    ocr_engine=active_ocr_engine,
                                    document_id=document_id,
                                    document_dir=document_dir,
                                    is_rtl=lang_pipeline.is_rtl,
                                    layout_confidence=lang_pipeline.layout_confidence,
                                    ocr_engine_label=lang_pipeline.ocr_engine_label,
                                )
                            )

                        gemini_response, gemini_elapsed, gemini_usage = run_page_llm(
                            page_path=prep["page_path"],
                            json_path=prep["json_path"],
                            prompt=lang_pipeline.grouping_prompt,
                        )

                        result = finish_page(
                            prep,
                            gemini_response,
                            gemini_elapsed,
                            gemini_usage,
                            use_contested_block_arbitration=(
                                document_llm_provider != "local"
                                and lang_pipeline.use_contested_block_arbitration
                            ),
                            use_orphan_block_reassignment=(
                                lang_pipeline
                                .use_orphan_block_reassignment
                            ),
                            use_orphan_title_root_repair=(
                                lang_pipeline
                                .use_orphan_title_root_repair
                            ),
                            use_unclaimed_kicker_recovery=(
                                lang_pipeline
                                .use_unclaimed_kicker_recovery
                            ),
                            use_unclaimed_image_recovery=(
                                lang_pipeline
                                .use_unclaimed_image_recovery
                            ),
                            use_unclaimed_footprint_recovery=(
                                lang_pipeline
                                .use_unclaimed_footprint_recovery
                            ),
                            use_article_splitter=(
                                lang_pipeline.use_article_splitter
                            ),
                            use_wide_top_banner_detachment=(
                                getattr(lang_pipeline, "use_wide_top_banner_detachment", True)
                            ),
                            use_dropped_article_recovery=(
                                lang_pipeline.use_dropped_article_recovery
                            ),
                            use_boundary_decomposition=(
                                lang_pipeline.use_boundary_decomposition
                            ),
                        )

                        page_results[
                            page_number
                        ] = result

                        page_crops = (
                            result.get(
                                "final_article_crops",
                                [],
                            )
                        )

                        all_final_article_crops.extend(
                            page_crops
                        )

                        if page_crops:

                            # Extraction no longer runs here -- see
                            # PHASE 2 below, which fires once every
                            # page in the document has finished this
                            # Phase 1 loop. Only the crop count (and
                            # article IDs, for oversized-page chunking)
                            # is recorded now, for Phase 2 to batch
                            # against without re-scanning disk.
                            page_crop_counts[page_number] = (
                                len(page_crops)
                            )

                            page_crop_article_ids[page_number] = [
                                crop.get("article_id")
                                for crop in page_crops
                            ]

                        print()

                        print(
                            f"Page {page_number}"
                        )

                        print(
                            f"Final Article Crops : "
                            f"{len(page_crops)}"
                        )

            else:

                # ---------------------------------------------
                # PIPELINED: page prepare (layout/OCR/knowledge/
                # cleaning/export) and the page-level LLM call are
                # no longer two fully sequential phases with a
                # document-wide barrier between them.
                #
                # OCR/prepare still happens one page at a time on
                # the main thread (CPU-bound, no benefit from
                # threading it), but as soon as EACH page's prepare
                # finishes, that page's LLM call is submitted to run
                # concurrently in the background while prepare moves
                # on to the next page immediately -- instead of
                # waiting for every page's OCR to finish before any
                # LLM call starts.
                #
                # Previously: total time ~= sum(prepare) + sum(LLM)/concurrency.
                # Now: total time trends toward max(sum(prepare), sum(LLM)/concurrency),
                # since the two stages overlap across pages instead of
                # being separated by a whole-document barrier.
                # ---------------------------------------------

                if document_llm_provider == "local":
                    run_page_llm = run_local
                elif document_llm_provider == "gemini":
                    run_page_llm = run_gemini
                else:
                    run_page_llm = run_openai

                page_concurrency_env = (
                    "GEMINI_PAGE_CONCURRENCY"
                    if document_llm_provider == "gemini"
                    else "OPENAI_PAGE_CONCURRENCY"
                )

                gemini_concurrency = int(
                    os.getenv(
                        page_concurrency_env,
                        "4",
                    )
                )

                # Hard ceiling on page-boundary/article-grouping LLM
                # concurrency, applied on top of the provider-specific
                # setting above -- too many boundary requests in flight
                # at once was overloading the API and degrading
                # page-boundary accuracy (missed boundaries, weak
                # grouping decisions). Additional page-boundary requests
                # beyond this cap simply queue on the executor below and
                # run as soon as a slot frees up -- see
                # PAGE_BOUNDARY_MAX_CONCURRENCY.
                page_boundary_max_concurrency = int(
                    os.getenv(
                        "PAGE_BOUNDARY_MAX_CONCURRENCY",
                        "2",
                    )
                )

                gemini_concurrency = min(
                    gemini_concurrency,
                    page_boundary_max_concurrency,
                )

                # OCR/prepare concurrency: RapidOCR + YOLO both release
                # the GIL during their actual compute (ONNX Runtime /
                # torch inference), so a small thread pool here gives a
                # real speedup instead of the "no benefit from threading
                # it" situation pure-Python CPU work would have. Bounded
                # low (default 2) since this machine has 6 physical
                # cores shared with everything else the user has open --
                # see OCR_PAGE_CONCURRENCY to change it.
                prepare_concurrency = int(
                    os.getenv(
                        "OCR_PAGE_CONCURRENCY",
                        "2",
                    )
                )

                print()
                print("=" * 60)
                print(
                    f"PIPELINED PAGE PREPARE "
                    f"(concurrency={prepare_concurrency}) + "
                    f"{document_llm_provider.upper()} "
                    f"(concurrency={gemini_concurrency})"
                )
                print("=" * 60)

                preps = []
                gemini_results = {}

                with ThreadPoolExecutor(
                    max_workers=max(1, gemini_concurrency),
                ) as executor, ThreadPoolExecutor(
                    max_workers=max(1, prepare_concurrency),
                ) as prepare_executor:

                    futures = {}

                    prepare_futures = {
                        prepare_executor.submit(
                            prepare_page,
                            page_number=page_number,
                            page_path=page_path,
                            detector=self.detector,
                            ocr_engine=active_ocr_engine,
                            document_id=document_id,
                            document_dir=document_dir,
                            is_rtl=lang_pipeline.is_rtl,
                            layout_confidence=lang_pipeline.layout_confidence,
                            ocr_engine_label=lang_pipeline.ocr_engine_label,
                        ): page_number
                        for page_number, page_path in enumerate(
                            pages,
                            start=1,
                        )
                    }

                    print()
                    print("=" * 60)
                    print(
                        f"PREPARING {len(pages)} PAGE(S) "
                        f"(up to {prepare_concurrency} at a time)"
                    )
                    print("=" * 60)

                    _report(
                        f"Preparing {len(pages)} page(s)",
                        11,
                    )

                    for prepare_future in as_completed(
                        prepare_futures,
                    ):

                        page_number = prepare_futures[prepare_future]

                        prep = prepare_future.result()

                        preps.append(prep)

                        print(
                            f"✓ Page {page_number}/{len(pages)} prepared"
                        )

                        _report(
                            f"Prepared page {page_number}/{len(pages)}",
                            11 + round(24 * len(preps) / len(pages)),
                        )

                        # ---------------------------------------------
                        # Submit this page's LLM call NOW -- it runs in
                        # the background while prepare keeps going on
                        # other pages, instead of waiting for every page
                        # to finish preparing first.
                        # ---------------------------------------------

                        print(
                            f"→ Submitting {document_llm_provider} call "
                            f"for page {page_number} (runs in "
                            f"background while other pages are "
                            f"prepared)"
                        )

                        futures[
                            executor.submit(
                                run_page_llm,
                                page_path=prep["page_path"],
                                json_path=prep["json_path"],
                                prompt=lang_pipeline.grouping_prompt,
                            )
                        ] = prep["page_number"]

                    # Prepared out of completion order (up to
                    # prepare_concurrency pages in flight at once) --
                    # restore page-number order so every downstream step
                    # that iterates `preps` in sequence behaves exactly
                    # as it did when prepare was strictly sequential.
                    preps.sort(
                        key=lambda p: p["page_number"],
                    )

                    _report(
                        f"All {len(preps)} page(s) prepared -- "
                        f"waiting on remaining "
                        f"{document_llm_provider} calls",
                        35,
                    )

                    completed = 0

                    for future in as_completed(futures):

                        page_number = futures[future]

                        gemini_results[
                            page_number
                        ] = future.result()

                        completed += 1

                        _report(
                            f"{document_llm_provider} analyzed page "
                            f"{page_number} ({completed}/{len(preps)})",
                            35 + round(30 * completed / len(preps)),
                        )

                # ---------------------------------------------
                # Boundary pipeline + cropping per page, unchanged
                # and sequential (each page's result only depends
                # on its own LLM response, already collected above).
                # ---------------------------------------------

                for prep in preps:

                    page_number = prep["page_number"]

                    gemini_response, gemini_elapsed, gemini_usage = (
                        gemini_results[page_number]
                    )

                    print()
                    print("=" * 60)

                    print(
                        f"FINISHING PAGE "
                        f"{page_number}/{len(pages)}"
                    )

                    print("=" * 60)

                    _report(
                        f"Finishing page {page_number}/{len(pages)}",
                        65 + round(3 * page_number / len(pages)),
                    )

                    result = finish_page(
                        prep,
                        gemini_response,
                        gemini_elapsed,
                        gemini_usage,
                        # Matches the previous
                        # `is_hindi=(document_llm_provider == "gemini")`
                        # exactly: arbitration is a per-language policy,
                        # but "local" mode (no LLM calls at all) must
                        # still force it off regardless of the detected
                        # language, same as before.
                        use_contested_block_arbitration=(
                            document_llm_provider != "local"
                            and lang_pipeline.use_contested_block_arbitration
                        ),
                        use_orphan_block_reassignment=(
                            lang_pipeline
                            .use_orphan_block_reassignment
                        ),
                        use_orphan_title_root_repair=(
                            lang_pipeline
                            .use_orphan_title_root_repair
                        ),
                        use_unclaimed_kicker_recovery=(
                            lang_pipeline
                            .use_unclaimed_kicker_recovery
                        ),
                        use_unclaimed_image_recovery=(
                            lang_pipeline
                            .use_unclaimed_image_recovery
                        ),
                        use_unclaimed_footprint_recovery=(
                            lang_pipeline
                            .use_unclaimed_footprint_recovery
                        ),
                        use_article_splitter=(
                            lang_pipeline.use_article_splitter
                        ),
                        use_wide_top_banner_detachment=(
                            getattr(lang_pipeline, "use_wide_top_banner_detachment", True)
                        ),
                        use_dropped_article_recovery=(
                            lang_pipeline.use_dropped_article_recovery
                        ),
                        use_boundary_decomposition=(
                            lang_pipeline.use_boundary_decomposition
                        ),
                    )

                    # ---------------------------------------------
                    # Save complete page result
                    # ---------------------------------------------

                    page_results[
                        page_number
                    ] = result

                    # ---------------------------------------------
                    # NEW FINAL ARTICLE CROPS
                    # ---------------------------------------------

                    page_crops = (
                        result.get(
                            "final_article_crops",
                            [],
                        )
                    )

                    all_final_article_crops.extend(
                        page_crops
                    )

                    # ---------------------------------------------
                    # PAGE-BY-PAGE ARTICLE EXTRACTION
                    #
                    # Runs immediately, as soon as THIS page's crops
                    # exist -- see the interleave_extraction setup above
                    # STEP 5's loop for why this is skipped for local
                    # mode / Stream B documents.
                    # ---------------------------------------------

                    if interleave_extraction and page_crops:

                        print()
                        print("=" * 60)
                        print(
                            f"ARTICLE-LEVEL EXTRACTION -- PAGE "
                            f"{page_number}"
                        )
                        print("=" * 60)

                        batch_result = (
                            article_extractor.process_batch(
                                document_dir=document_dir,
                                page_numbers=[page_number],
                                batch_index=page_number,
                                output_root=gemini_batch_dir,
                                pending_continuations=(
                                    pending_continuations
                                ),
                                known_pages=list(
                                    range(1, len(pages) + 1)
                                ),
                            )
                        )

                        page_extraction_results.append(
                            batch_result
                        )

                        all_extracted_articles.extend(
                            batch_result.get("articles", [])
                        )

                        all_continuation_links.extend(
                            batch_result.get(
                                "continuation_links", [],
                            )
                        )

                        pending_continuations = (
                            batch_result.get(
                                "pending_continuations", [],
                            )
                        )

                        pending_path = (
                            gemini_batch_dir
                            / "pending_continuations.json"
                        )

                        with open(
                            pending_path,
                            "w",
                            encoding="utf-8",
                        ) as f:

                            json.dump(
                                {
                                    "pending_count":
                                        len(
                                            pending_continuations
                                        ),
                                    "pending":
                                        pending_continuations,
                                },
                                f,
                                indent=4,
                                ensure_ascii=False,
                            )

                    # ---------------------------------------------
                    # Page summary
                    # ---------------------------------------------

                    print()

                    print(
                        f"Page {page_number}"
                    )

                    print(
                        f"Final Article Crops : "
                        f"{len(page_crops)}"
                    )

            # =================================================
            # STEP 6
            # Final Crop Summary
            # =================================================

            print()
            print("=" * 60)
            print("FINAL ARTICLE CROP SUMMARY")
            print("=" * 60)

            _report("Finalizing article crops", 69)

            print(
                f"Pages Processed : "
                f"{len(pages)}"
            )

            print(
                f"Final Article Crops : "
                f"{len(all_final_article_crops)}"
            )

            if not all_final_article_crops:

                raise RuntimeError(
                    "No final article crops were created."
                )

            # -------------------------------------------------
            # Article boundaries are now final for every page --
            # reported as its own distinct event (rather than folded
            # into the generic "Finalizing article crops" stage
            # above) so the frontend can reliably show a one-time
            # notification exactly at this point, before batching
            # for article-level extraction begins below (STEP 12).
            # -------------------------------------------------

            _report(
                "Article boundaries created",
                70,
                event="boundaries_created",
                article_count=len(all_final_article_crops),
            )

            # =================================================
            # STEP 10
            # Move Rendered Pages
            # =================================================

            destination_pages = (
                document_dir / "pages"
            )

            if destination_pages.exists():

                shutil.rmtree(
                    destination_pages
                )

            shutil.move(
                str(
                    workspace / "pages"
                ),
                str(
                    destination_pages
                ),
            )

            print(
                "✓ Page images moved to "
                "document storage"
            )

            # -------------------------------------------------
            # Everything the Viewer needs to open this document
            # (plain page images, final boundary visualizations,
            # and per-page article crops) is now on disk, even
            # though article-level extraction/batching (STEP 12)
            # and finalization haven't run yet -- persist that so
            # the frontend can let the document be opened and
            # boundary-edited immediately, instead of waiting for
            # `status` to reach "completed".
            # -------------------------------------------------

            self._save_document_metadata(
                document_dir=document_dir,
                metadata=metadata,
                page_count=len(pages),
                status="processing",
                boundaries_ready=True,
            )

            print(
                "✓ document.json marked boundaries_ready"
            )

            # =================================================
            # STEP 11
            # OLD GEMINI INTELLIGENCE
            # =================================================

            print()
            print("=" * 60)
            print("OLD GEMINI INTELLIGENCE DISABLED")
            print("=" * 60)

            print(
                "✓ Candidate Cleaner: SKIPPED"
            )

            print(
                "✓ Newspaper Intelligence: SKIPPED"
            )

            # =================================================
            # STEP 11.5
            # BOUNDARY REPAIR FROM STREAM B HEADING INDEX
            #
            # Stream B (document_order_extractor.py) has been
            # reading raw pages independently since right after PDF
            # rendering, and by now (Stream A's own per-page loop is
            # completely finished -- this does NOT make Stream A
            # wait on anything) most or all of its batches are
            # already done or close to it.
            #
            # Each batch is collected via as_completed(), NOT in
            # submission order, so a heading is repaired as soon as
            # ITS batch returns rather than waiting for every batch
            # to finish first. For every heading in a batch: if it
            # matches a real page_json title block that isn't yet
            # covered by any article's block_ids, AND that heading
            # has real matching body text belonging to exactly one
            # existing article (see heading_repair.py), that
            # article's block_ids/boundary/crop is corrected in
            # place -- surgically, one article at a time. Nothing is
            # ever fabricated: with no clear match, Stream A's
            # existing boundaries are left exactly as they are.
            #
            # This has to happen BEFORE article-level extraction
            # below, since that stage reads final_articles_crops
            # from disk -- a heading repaired only afterward would
            # already have been sent to the vision model without it.
            #
            # The sections collected here are reused by the
            # (unchanged) text-level Stream B reconciliation further
            # down instead of re-collecting the same futures twice.
            # =================================================

            document_order_sections = []

            if document_order_enabled:

                print()
                print("=" * 60)
                print("BOUNDARY REPAIR FROM STREAM B")
                print("=" * 60)

                for future in as_completed(document_order_futures):

                    try:
                        batch_sections = future.result()
                    except Exception as exc:
                        print(
                            "⚠ Document-order batch failed, "
                            f"skipping: {exc}"
                        )
                        continue

                    document_order_sections.extend(batch_sections)

                    lang_pipeline.document_order_repair_headings(
                        document_dir,
                        batch_sections,
                    )

                document_order_executor.shutdown(
                    wait=True
                )

                print("=" * 60)

            # =================================================
            # STEP 12
            # ARTICLE-LEVEL EXTRACTION
            # =================================================

            # "local" is the one global override (LLM_PROVIDER=local)
            # shared with boundary detection's document_llm_provider
            # above. Otherwise this reports whichever extractor_class
            # this language pipeline actually instantiates below, NOT
            # document_llm_provider -- that value is boundary/grouping
            # routing only, and can now differ from article extraction
            # (e.g. Hindi groups boundaries via OpenAI but extracts
            # articles via GeminiArticleExtractor).
            if document_llm_provider == "local":

                article_extractor_engine = "local"

            else:

                extractor_class_name = (
                    lang_pipeline.extractor_class.__name__
                )

                if "Gemini" in extractor_class_name:
                    article_extractor_engine = "gemini"
                elif "OpenAI" in extractor_class_name:
                    article_extractor_engine = "openai"
                else:
                    article_extractor_engine = document_llm_provider

            print()
            print("=" * 60)
            print(
                f"ARTICLE-LEVEL EXTRACTION "
                f"({article_extractor_engine.upper()})"
            )
            print("=" * 60)

            print(
                "Input:"
            )

            print(
                f"  {document_dir / 'final_articles_crops'}"
            )

            if interleave_extraction:

                # PHASE 2: boundary creation + cropping already
                # finished for every page in Phase 1 (STEP 5) above,
                # without pausing for extraction. Now batch extraction
                # across the WHOLE document, using the same
                # extraction_pages_per_batch / extraction_max_articles_per_batch
                # rules as before -- unchanged batching, only the
                # timing moved from per-page to after-all-pages.
                print(
                    f"Batch size: {lang_pipeline.extraction_pages_per_batch} "
                    f"page(s), max "
                    f"{lang_pipeline.extraction_max_articles_per_batch} "
                    f"articles/batch (Phase 2, after all boundaries)"
                )

                _report(
                    f"Extracting articles ({article_extractor_engine})",
                    72,
                )

                # ---------------------------------------------
                # Pre-compute batch specs (pure, no I/O) using the
                # exact same extraction_pages_per_batch /
                # extraction_max_articles_per_batch rules the
                # sequential flush loop used to apply one flush at a
                # time. Each spec is
                # {"page_numbers": [...], "article_id_subset": [...] | None}
                # -- article_id_subset is None for a normal multi-page
                # group (process_batch sends every crop on those
                # pages) and a specific list of article IDs when a
                # SINGLE page alone produced more crops than
                # max_articles (see the chunking branch below).
                # ---------------------------------------------

                extraction_batches = []
                current_group_pages = []
                current_group_article_count = 0

                def _flush_current_group():
                    if current_group_pages:
                        extraction_batches.append(
                            {
                                "page_numbers": list(
                                    current_group_pages
                                ),
                                "article_id_subset": None,
                            }
                        )

                for page_number in sorted(page_crop_counts):

                    crop_count = page_crop_counts[page_number]

                    max_articles = (
                        lang_pipeline
                        .extraction_max_articles_per_batch
                    )

                    if (
                        max_articles is not None
                        and crop_count > max_articles
                    ):

                        # This single page alone exceeds the cap --
                        # flush whatever multi-page group is pending,
                        # then split THIS page's own article IDs into
                        # sub-batches of at most max_articles each,
                        # instead of shipping every crop on the page
                        # in one oversized call.
                        _flush_current_group()
                        current_group_pages = []
                        current_group_article_count = 0

                        article_ids = (
                            page_crop_article_ids.get(
                                page_number, [],
                            )
                        )

                        for chunk_start in range(
                            0,
                            len(article_ids),
                            max_articles,
                        ):

                            chunk = article_ids[
                                chunk_start
                                : chunk_start + max_articles
                            ]

                            extraction_batches.append(
                                {
                                    "page_numbers": [page_number],
                                    "article_id_subset": chunk,
                                }
                            )

                        continue

                    projected_count = (
                        current_group_article_count
                        + crop_count
                    )

                    if (
                        current_group_pages
                        and max_articles is not None
                        and projected_count > max_articles
                    ):

                        # This page would push the running
                        # article-crop total past the cap -- start a
                        # new group with this page instead (mirrors
                        # OpenAIArticleExtractor._make_page_batches'
                        # own byte-size cap: a single oversized page
                        # still ships alone).
                        _flush_current_group()
                        current_group_pages = []
                        current_group_article_count = 0

                    current_group_pages.append(
                        page_number
                    )

                    current_group_article_count += (
                        crop_count
                    )

                    if (
                        len(current_group_pages)
                        >= lang_pipeline.extraction_pages_per_batch
                    ):
                        _flush_current_group()
                        current_group_pages = []
                        current_group_article_count = 0

                # Whatever remains from the last, possibly partial, group.
                _flush_current_group()

                # ---------------------------------------------
                # Run batches concurrently (max_workers=2).
                #
                # Cross-batch continuations (e.g. Page 6 -> Page 12)
                # cannot be resolved from inside a single batch's own
                # extraction call once batches no longer run strictly
                # in page order -- they are resolved globally, after
                # every batch finishes, by finalize_batches()'s
                # text-based continuation repair
                # (_repair_titleless_continuations) and then by
                # LogicalArticleBuilder (STEP 13). Chaining
                # pending_continuations batch-to-batch therefore only
                # created a false sequential dependency between
                # batches that don't actually depend on each other.
                # Each batch now runs independently with
                # pending_continuations=[]; max_workers=2 keeps at
                # most 2 OpenAI requests in flight at once (same
                # safety margin as PAGE_BOUNDARY_MAX_CONCURRENCY
                # above).
                # ---------------------------------------------

                print(
                    f"Extraction batches: "
                    f"{len(extraction_batches)} "
                    f"(max_workers=2, concurrent)"
                )

                batch_results_by_index = {}

                with ThreadPoolExecutor(
                    max_workers=2,
                ) as extraction_executor:

                    extraction_futures = {
                        extraction_executor.submit(
                            article_extractor.process_batch,
                            document_dir=document_dir,
                            page_numbers=batch_spec["page_numbers"],
                            batch_index=batch_index,
                            output_root=gemini_batch_dir,
                            pending_continuations=[],
                            known_pages=list(
                                range(1, len(pages) + 1)
                            ),
                            article_id_subset=(
                                batch_spec["article_id_subset"]
                            ),
                        ): batch_index
                        for batch_index, batch_spec in enumerate(
                            extraction_batches,
                            start=1,
                        )
                    }

                    for extraction_future in as_completed(
                        extraction_futures,
                    ):

                        batch_index = (
                            extraction_futures[
                                extraction_future
                            ]
                        )

                        batch_results_by_index[
                            batch_index
                        ] = extraction_future.result()

                # Fold results back in ascending batch-index (page)
                # order, not completion order, so downstream ordering
                # stays identical to the previous sequential run.
                for batch_index in sorted(
                    batch_results_by_index
                ):

                    batch_result = (
                        batch_results_by_index[batch_index]
                    )

                    page_extraction_results.append(
                        batch_result
                    )

                    all_extracted_articles.extend(
                        batch_result.get("articles", [])
                    )

                    all_continuation_links.extend(
                        batch_result.get(
                            "continuation_links", [],
                        )
                    )

                    pending_continuations.extend(
                        batch_result.get(
                            "pending_continuations", [],
                        )
                    )

                pending_path = (
                    gemini_batch_dir
                    / "pending_continuations.json"
                )

                with open(
                    pending_path,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        {
                            "pending_count":
                                len(pending_continuations),
                            "pending":
                                pending_continuations,
                        },
                        f,
                        indent=4,
                        ensure_ascii=False,
                    )

                type(article_extractor).finalize_batches(
                    document_dir=document_dir,
                    model=article_extractor.model,
                    pages_per_batch=article_extractor.pages_per_batch,
                    all_articles=all_extracted_articles,
                    all_links=all_continuation_links,
                    pending_continuations=pending_continuations,
                    output_root=gemini_batch_dir,
                    results=page_extraction_results,
                    printed_page_map=printed_page_map,
                )

                gemini_articles = page_extraction_results

            else:

                print(
                    "Batch size: 3 pages"
                )

                _report(
                    f"Extracting articles ({article_extractor_engine})",
                    72,
                )

                if article_extractor_engine == "local":

                    article_extractor = (
                        LocalArticleExtractor(
                            pages_per_batch=3,
                            ocr_engine=active_ocr_engine,
                        )
                    )

                else:

                    article_extractor = (
                        lang_pipeline.extractor_class(
                            pages_per_batch=(
                                lang_pipeline
                                .extraction_pages_per_batch
                            ),
                            prompt_template=(
                                lang_pipeline
                                .extraction_prompt_template
                            ),
                        )
                    )

                gemini_articles = (
                    article_extractor.process_document(
                        document_dir,
                        printed_page_map=printed_page_map,
                    )
                )

            # =================================================
            # STREAM B TEXT RECONCILIATION
            #
            # Reuses the sections already collected in STEP 11.5
            # above (every batch was awaited there, before article-
            # level extraction ran) -- fold in genuine TEXT gaps
            # only; Stream A's article_text stays authoritative; see
            # reconcile.py. Boundary/crop-level gaps (a heading
            # missing from its article) were already handled in STEP
            # 11.5, separately from this text-only pass.
            # =================================================

            if document_order_enabled:

                gemini_articles, _unresolved_sections = (
                    lang_pipeline.document_order_reconcile_articles(
                        gemini_articles,
                        document_order_sections,
                    )
                )

            _report("Article-level extraction completed", 90)

            print()
            print(
                "✓ Article-level extraction completed"
            )

            print(
                f"Gemini result items: "
                f"{len(gemini_articles)}"
            )

            gemini_batch_dir = (
                document_dir
                / "gemini_article_batches"
            )

            if not gemini_batch_dir.exists():

                raise RuntimeError(
                    "Gemini article batch directory "
                    "was not created:\n"
                    f"{gemini_batch_dir}"
                )

            # =================================================
            # STEP 13
            # LOGICAL ARTICLE FINALIZATION
            # =================================================

            print()
            print("=" * 60)
            print("LOGICAL ARTICLE FINALIZATION")
            print("=" * 60)

            _report("Building logical articles", 92)

            print(
                "Building logical articles..."
            )

            final_builder = (
                LogicalArticleBuilder(
                    document_dir
                )
            )

            final_builder.build_all()

            print()
            print(
                "✓ Logical article finalization completed"
            )

            final_articles_file = (
                document_dir
                / "final_articles"
                / "final_articles.json"
            )

            if not final_articles_file.exists():

                raise RuntimeError(
                    "Final articles JSON was not created:\n"
                    f"{final_articles_file}"
                )

            # -------------------------------------------------
            # Read final article count
            # -------------------------------------------------

            try:

                with open(
                    final_articles_file,
                    "r",
                    encoding="utf-8",
                ) as f:

                    final_article_data = (
                        json.load(f)
                    )

                if isinstance(
                    final_article_data,
                    list,
                ):

                    final_article_count = (
                        len(
                            final_article_data
                        )
                    )

                elif isinstance(
                    final_article_data,
                    dict,
                ):

                    articles_value = (
                        final_article_data.get(
                            "articles",
                            [],
                        )
                    )

                    if isinstance(
                        articles_value,
                        list,
                    ):

                        final_article_count = (
                            len(
                                articles_value
                            )
                        )

                    else:

                        final_article_count = 0

                else:

                    final_article_count = 0

            except Exception:

                final_article_count = 0

            print(
                "✓ Final articles JSON:"
            )

            print(
                f"  {final_articles_file}"
            )

            print(
                f"✓ Final logical articles: "
                f"{final_article_count}"
            )

            # =================================================
            # STEP 14
            # UPDATE DOCUMENT METADATA
            # =================================================

            print()
            print("=" * 60)
            print("UPDATING DOCUMENT METADATA")
            print("=" * 60)

            _report("Updating document metadata", 95)

            document_metadata = (
                self._save_document_metadata(
                    document_dir=document_dir,
                    metadata=metadata,
                    page_count=len(pages),
                    article_count=len(
                        all_final_article_crops
                    ),
                    final_article_count=(
                        final_article_count
                    ),
                    status="processing",
                    gemini_status="completed",
                    finalization_status="completed",
                    database_status="pending",
                )
            )

            print(
                "✓ document.json updated BEFORE database import"
            )

            print(
                f"  Newspaper : "
                f"{document_metadata.get('newspaper_name')}"
            )

            print(
                f"  Edition   : "
                f"{document_metadata.get('edition')}"
            )

            print(
                f"  Date      : "
                f"{document_metadata.get('publish_date')}"
            )

            # =================================================
            # SAFETY CHECK BEFORE DATABASE IMPORT
            # =================================================

            if not document_metadata.get(
                "newspaper_name"
            ):

                raise RuntimeError(
                    "Database import blocked: "
                    "newspaper_name is empty."
                )

            if not document_metadata.get(
                "publish_date"
            ):

                raise RuntimeError(
                    "Database import blocked: "
                    "publish_date is empty."
                )

            # =================================================
            # STEP 15
            # DATABASE IMPORT
            # =================================================

            print()
            print("=" * 60)
            print("DATABASE IMPORT")
            print("=" * 60)

            _report("Importing into database", 97)

            print(
                f"Document ID : "
                f"{document_id}"
            )

            print(
                f"Document Dir: "
                f"{document_dir}"
            )

            import_document_directory(
                document_dir
            )

            print()
            print(
                "✓ Database import completed"
            )

            # =================================================
            # STEP 16
            # FINAL DOCUMENT STATUS
            # =================================================

            document_metadata = (
                self._save_document_metadata(
                    document_dir=document_dir,
                    metadata=metadata,
                    page_count=len(pages),
                    article_count=len(
                        all_final_article_crops
                    ),
                    final_article_count=(
                        final_article_count
                    ),
                    status="completed",
                    gemini_status="completed",
                    finalization_status="completed",
                    database_status="completed",
                )
            )

            print(
                "✓ document.json final status updated"
            )

            # =================================================
            # STEP 17
            # FINAL PIPELINE STATUS
            # =================================================

            print()
            print("=" * 70)
            print("FINAL ARTICLE PIPELINE STATUS")
            print("=" * 70)

            print(
                "✓ Final article boundaries created"
            )

            print(
                "✓ Exact article crops created"
            )

            print(
                "✓ Article-level Gemini extraction completed"
            )

            print(
                "✓ Continuation resolution completed"
            )

            print(
                "✓ Logical articles finalized"
            )

            print(
                "✓ Article images extracted"
            )

            print(
                "✓ Article images composed"
            )

            print(
                "✓ final_articles.json created"
            )

            print(
                "✓ Document metadata saved before import"
            )

            print(
                "✓ PostgreSQL database import completed"
            )

            # =================================================
            # COMPLETE
            # =================================================

            print()
            print("=" * 70)
            print("NEWSPAPER PIPELINE COMPLETED")
            print("=" * 70)

            print(
                f"Document ID          : "
                f"{document_id}"
            )

            print(
                f"Pages                : "
                f"{len(pages)}"
            )

            print(
                f"Final Article Crops  : "
                f"{len(all_final_article_crops)}"
            )

            print(
                f"Final Logical Articles: "
                f"{final_article_count}"
            )

            print(
                f"Newspaper            : "
                f"{metadata.get('newspaper_name')}"
            )

            print(
                f"Edition              : "
                f"{metadata.get('edition')}"
            )

            print(
                f"Publish Date         : "
                f"{metadata.get('publish_date')}"
            )

            print(
                f"Final Articles JSON  : "
                f"{final_articles_file}"
            )

            print(
                f"Database             : "
                f"newspaper_archive"
            )

            print(
                "Status               : COMPLETED"
            )

            print(
                f"Output Directory     : "
                f"{document_dir}"
            )

            print("=" * 70)

            _report(
                "Completed",
                100,
                document_id=document_id,
                status="completed",
            )

            # =================================================
            # RETURN
            # =================================================

            return {

                "document_id":
                    document_id,

                "document_dir":
                    document_dir,

                "pages":
                    pages,

                "metadata":
                    metadata,

                "final_articles":
                    gemini_articles,

                "final_article_count":
                    final_article_count,

                "final_articles_file":
                    final_articles_file,

                "final_article_crops":
                    all_final_article_crops,

                "page_results":
                    page_results,

                "status":
                    "completed",
            }

        except Exception as exc:

            import traceback
            print()
            print("=" * 60)
            print("PIPELINE FAILED")
            print("=" * 60)
            traceback.print_exc()
            print("=" * 60)

            if document_dir is not None:

                self._mark_document_failed(
                    document_dir=document_dir,
                    error=exc,
                )

            _report(
                "Failed",
                100,
                status="failed",
                error=str(exc),
            )

            raise

        finally:

            # =================================================
            # Always clean temporary workspace
            # =================================================

            workspace_manager.cleanup()
