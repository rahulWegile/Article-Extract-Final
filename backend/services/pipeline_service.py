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

from pipeline.page_processor_gemini import process_page

from pipeline.gemini.newspaper_client import (
    NewspaperClient,
)

# ============================================================
# NEW PIPELINE STAGES
# ============================================================

from pipeline.intelligence.gemini_article_extractor import (
    GeminiArticleExtractor,
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

        # =====================================================
        # Newspaper Metadata Client
        # =====================================================

        self.newspaper_client = (
            NewspaperClient()
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

    # ========================================================
    # METADATA HELPERS
    # ========================================================

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

        # A 429/503 from the Gemini API is a retry-worthy condition on
        # Google's side (quota exhaustion / rate limiting / temporary
        # overload), not a bug in this pipeline; surface that
        # distinction (plus Google's own error `status`, e.g.
        # RESOURCE_EXHAUSTED vs UNAVAILABLE) so the frontend/operator
        # knows re-uploading is likely to work without any code change.
        status_code = getattr(error, "code", None)
        is_api_error = type(error).__name__ in {
            "ServerError", "ClientError", "APIError",
        }
        gemini_unavailable = is_api_error and status_code in (429, 503)

        document_metadata["status"] = "failed"
        document_metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "transient": gemini_unavailable,
            "api_status": getattr(error, "status", None) if is_api_error else None,
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(document_metadata, f, indent=4, ensure_ascii=False)

    # ========================================================
    # PROCESS PDF
    # ========================================================

    def process_pdf(
        self,
        pdf_path,
    ):

        print()
        print("=" * 60)
        print("PROCESSING PDF")
        print("=" * 60)

        # =====================================================
        # Temporary Workspace
        # =====================================================

        workspace = (
            WorkspaceManager().create()
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

            try:

                metadata = (
                    self.newspaper_client.extract_metadata(
                        image_path=pages[0],
                    )
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

            if not isinstance(
                metadata,
                dict,
            ):

                metadata = {}

            # -------------------------------------------------
            # Build reliable metadata
            # -------------------------------------------------

            metadata = (
                self._build_document_metadata(
                    pdf_path=pdf_path,
                    extracted_metadata=metadata,
                )
            )

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
            # IMPORTANT SAFETY CHECK
            # -------------------------------------------------

            if not metadata.get(
                "newspaper_name"
            ):

                raise RuntimeError(
                    "Could not determine newspaper_name "
                    "from Gemini metadata or filename."
                )

            if not metadata.get(
                "publish_date"
            ):

                raise RuntimeError(
                    "Could not determine publish_date "
                    "from Gemini metadata or filename."
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
                document_dir / "gemini"
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

            for page_number, page_path in enumerate(
                pages,
                start=1,
            ):

                print()
                print("=" * 60)

                print(
                    f"PROCESSING PAGE "
                    f"{page_number}/{len(pages)}"
                )

                print("=" * 60)

                # ---------------------------------------------
                # Existing page pipeline
                # ---------------------------------------------

                result = process_page(
                    page_number=page_number,
                    page_path=page_path,
                    detector=self.detector,
                    ocr_engine=self.ocr_engine,
                    document_id=document_id,
                    document_dir=document_dir,
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

            article_extractor_engine = (
                os.getenv(
                    "ARTICLE_EXTRACTOR_ENGINE",
                    "gemini",
                )
                .strip()
                .lower()
            )

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

            if article_extractor_engine == "local":

                article_extractor = (
                    LocalArticleExtractor(
                        pages_per_batch=3,
                        ocr_engine=self.ocr_engine,
                    )
                )

            else:

                article_extractor = (
                    GeminiArticleExtractor(
                        pages_per_batch=3,
                    )
                )

            gemini_articles = (
                article_extractor.process_document(
                    document_dir
                )
            )

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

            raise

        finally:

            # =================================================
            # Always clean temporary workspace
            # =================================================

            WorkspaceManager().cleanup()