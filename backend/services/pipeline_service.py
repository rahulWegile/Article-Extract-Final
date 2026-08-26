import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from backend.services.workspace_manager import WorkspaceManager
from backend.services.document_manager import DocumentManager

from pipeline.render_pdf import render_pdf
from pipeline.layout_detector import LayoutDetector

# ============================================================
# RAPIDOCR
# ============================================================

from pipeline.ocr.rapidocr_engine import RapidOCREngine

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
)

# ============================================================
# NEW PIPELINE STAGES
# ============================================================

from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
)

from pipeline.intelligence.openai_article_extractor import (
    OpenAIArticleExtractor,
)

from pipeline.intelligence.local.local_article_extractor import (
    LocalArticleExtractor,
)

from pipeline.finalization.logical_article_builder import (
    LogicalArticleBuilder,
)

from pipeline.database.import_final_articles import (
    import_document_directory,
)


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
        # Layout Detector
        # =====================================================

        self.detector = LayoutDetector()

        # =====================================================
        # RapidOCR
        # =====================================================

        self.ocr_engine = RapidOCREngine()

        # Lazily-created, cached Devanagari-configured OCR engine.
        # Kept separate from self.ocr_engine so switching a Hindi
        # document to it never affects English documents processed
        # by this same (long-lived) PipelineService instance.
        self._hindi_ocr_engine = None

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
        # Hindi-vs-English choice below (see _is_hindi_language)
        # only applies to boundary detection and article extraction,
        # both of which run after the language is already known.
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

    # ========================================================
    # LANGUAGE DETECTION (shared by OCR engine AND LLM routing)
    # ========================================================

    @staticmethod
    def _is_hindi_language(language):
        """
        Shared Hindi/Devanagari check for both OCR engine selection
        (_get_ocr_engine_for_language) and per-document LLM routing
        (see the LLM ROUTING note in process_pdf): "Hindi", "hi",
        "hin", and anything containing "hindi" (e.g. "Hindi
        (Devanagari)") all count. Kept as one place so the two
        decisions can never quietly disagree with each other.
        """

        language = (language or "").strip().lower()

        return (
            language in ("hi", "hin", "hindi")
            or "hindi" in language
        )

    # ========================================================
    # OCR ENGINE SELECTION
    # ========================================================

    def _get_ocr_engine_for_language(self, language):
        """
        Pick the RapidOCR engine to use for this document.

        Hindi documents get a Devanagari-configured engine;
        everything else keeps using the default (English/Latin)
        engine that was already validated. This is a per-document
        choice, not a global config change, so English documents
        are never affected by the Hindi model.
        """

        if not self._is_hindi_language(language):
            return self.ocr_engine

        if self._hindi_ocr_engine is None:

            self._hindi_ocr_engine = RapidOCREngine(
                lang="hi"
            )

        return self._hindi_ocr_engine

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

        if self._hindi_ocr_engine is None:
            self._hindi_ocr_engine = RapidOCREngine(lang="hi")

        hindi_confidence = average_confidence(
            self._hindi_ocr_engine
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

        return {
            "newspaper_name":
                newspaper_name,

            "edition":
                edition,

            "publish_date":
                publish_date,
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

            language = "English"

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

        document_metadata["status"] = "failed"
        document_metadata["error"] = {
            "type": error_type_name,
            "message": str(error),
            "transient": provider_unavailable,
            "api_status": (
                getattr(error, "status", None) if is_gemini_api_error
                else openai_status_code if is_openai_api_error
                else None
            ),
        }

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

            if (
                self.llm_provider == "local"
                and metadata.get("metadata_source")
                == "local_unresolved"
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
            # Pick OCR engine for this document's language
            # -------------------------------------------------

            active_ocr_engine = (
                self._get_ocr_engine_for_language(
                    metadata.get("language", "")
                )
            )

            print(
                f"OCR engine : "
                f"{'devanagari' if active_ocr_engine is self._hindi_ocr_engine else 'default'}"
            )

            # -------------------------------------------------
            # LLM ROUTING (boundary detection + article
            # extraction only -- newspaper metadata above always
            # uses OpenAI, see __init__)
            #
            # Hindi/Devanagari documents route to Gemini
            # (gemini-3.6-flash): confirmed this session, on the
            # same real headline crop, to read Devanagari
            # correctly where gpt-5.6-luna fabricates unrelated
            # text instead. Every other language stays on OpenAI.
            #
            # This is a PER-DOCUMENT decision based on the
            # language just determined above, not the global
            # LLM_PROVIDER value -- LLM_PROVIDER=gemini/openai no
            # longer manually pins boundary/extraction to one
            # engine, only LLM_PROVIDER=local still does (the
            # explicit no-API-calls override, orthogonal to
            # language).
            # -------------------------------------------------

            if self.llm_provider == "local":
                document_llm_provider = "local"
            elif self._is_hindi_language(
                metadata.get("language", "")
            ):
                document_llm_provider = "gemini"
            else:
                document_llm_provider = "openai"

            print(
                f"LLM routing : {document_llm_provider} "
                f"(language={metadata.get('language', '')})"
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

            print()
            print("=" * 60)
            print(
                f"PIPELINED PAGE PREPARE + "
                f"{document_llm_provider.upper()} "
                f"(concurrency={gemini_concurrency})"
            )
            print("=" * 60)

            preps = []
            gemini_results = {}

            with ThreadPoolExecutor(
                max_workers=max(1, gemini_concurrency),
            ) as executor:

                futures = {}

                for page_number, page_path in enumerate(
                    pages,
                    start=1,
                ):

                    print()
                    print("=" * 60)

                    print(
                        f"PREPARING PAGE "
                        f"{page_number}/{len(pages)}"
                    )

                    print("=" * 60)

                    _report(
                        f"Preparing page {page_number}/{len(pages)}",
                        11 + round(24 * (page_number - 1) / len(pages)),
                    )

                    prep = prepare_page(
                        page_number=page_number,
                        page_path=page_path,
                        detector=self.detector,
                        ocr_engine=active_ocr_engine,
                        document_id=document_id,
                        document_dir=document_dir,
                    )

                    preps.append(prep)

                    # ---------------------------------------------
                    # Submit this page's LLM call NOW -- it runs in
                    # the background while the loop immediately
                    # continues preparing the next page, instead of
                    # waiting for every page to finish preparing
                    # first.
                    # ---------------------------------------------

                    print(
                        f"→ Submitting {document_llm_provider} call "
                        f"for page {page_number} (runs in "
                        f"background while later pages are "
                        f"prepared)"
                    )

                    futures[
                        executor.submit(
                            run_page_llm,
                            page_path=prep["page_path"],
                            json_path=prep["json_path"],
                        )
                    ] = prep["page_number"]

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

                gemini_response, gemini_elapsed = (
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
            # STEP 12
            # ARTICLE-LEVEL EXTRACTION
            # =================================================

            # Same per-document choice as boundary detection above
            # (document_llm_provider), not a separate global
            # ARTICLE_EXTRACTOR_ENGINE setting -- Hindi/Devanagari
            # documents extract via Gemini, everything else via
            # OpenAI, "local" only when LLM_PROVIDER=local.
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

            elif article_extractor_engine == "gemini":

                article_extractor = (
                    GeminiArticleExtractor(
                        pages_per_batch=3,
                    )
                )

            else:

                article_extractor = (
                    OpenAIArticleExtractor(
                        pages_per_batch=3,
                    )
                )

            gemini_articles = (
                article_extractor.process_document(
                    document_dir
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

            print()
            print("=" * 60)
            print("PIPELINE FAILED")
            print("=" * 60)
            print(f"{type(exc).__name__}: {exc}")
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