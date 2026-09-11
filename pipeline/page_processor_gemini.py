import json
import os
import time

import cv2

from pipeline.block_parser import parse_results
from pipeline.sort_blocks import sort_blocks
from pipeline.article.visual_separator import recover_missing_titles

from pipeline.knowledge.block_knowledge_builder import (
    BlockKnowledgeBuilder,
)

from pipeline.preprocess.page_cleaner import PageCleaner

from pipeline.export.gemini_page_exporter import (
    GeminiPageExporter,
)

from pipeline.gemini.gemini_service import GeminiService
from pipeline.languages.hindi.grouping_prompt import (
    HINDI_GROUPING_PROMPT,
)
from pipeline.languages.english.grouping_prompt import (
    ENGLISH_GROUPING_PROMPT,
)
from pipeline.openai.openai_service import OpenAIService

from pipeline.gemini.gemini_boundary_pipeline import (
    GeminiBoundaryPipeline,
)

from pipeline.intelligence.final_article_cropper import (
    FinalArticleCropper,
)


# =========================================================
# TIMING HELPER
# =========================================================

def _record_timing(timings, label, start_time):
    """
    Record elapsed time for a pipeline stage.
    Timing is printed only at the end of the page.
    """

    elapsed = time.perf_counter() - start_time
    timings[label] = elapsed

    return time.perf_counter()


# =========================================================
# PREPARE PAGE (everything before the Gemini network call)
# =========================================================

def prepare_page(
    page_number,
    page_path,
    detector,
    ocr_engine,
    document_id,
    document_dir,
    is_rtl=False,
    layout_confidence=None,
):

    # =====================================================
    # TOTAL PAGE TIMER
    # =====================================================

    page_start = time.perf_counter()

    # All timings are collected here and printed only
    # after the complete page pipeline has finished.
    timings = {}

    print()
    print("=" * 60)
    print(f"Processing Page {page_number}")
    print("=" * 60)

    # =====================================================
    # Stage 1 : Layout Detection
    # =====================================================

    print()
    print("Detecting layout...")

    stage_start = time.perf_counter()

    detect_kwargs = (
        {"conf": layout_confidence}
        if layout_confidence is not None
        else {}
    )

    results = detector.detect(
        page_path,
        **detect_kwargs,
    )

    blocks = parse_results(
        results
    )

    print(
        f"Detected {len(blocks)} layout blocks"
    )

    if is_rtl:

        recovered_titles = recover_missing_titles(
            cv2.imread(str(page_path)),
            blocks,
        )

        if recovered_titles:

            print(
                f"Recovered {len(recovered_titles)} missing "
                "title block(s) from page image geometry"
            )

            blocks.extend(recovered_titles)

    blocks = sort_blocks(
        blocks,
        is_rtl=is_rtl,
    )

    stage_start = _record_timing(timings, 
        "Layout detection",
        stage_start,
    )

    # =====================================================
    # Stage 2 : Page-Level RapidOCR
    # =====================================================

    print()
    print("Running RapidOCR...")

    stage_start = time.perf_counter()

    ocr_results = ocr_engine.process_blocks(
        page_path,
        blocks,
        page_number=page_number,
    )

    for block, ocr in zip(
        blocks,
        ocr_results,
    ):

        block.text = ocr.text

        block.ocr_confidence = (
            ocr.confidence
        )

        block.ocr_lines = (
            ocr.lines
        )

    print(
        "✓ RapidOCR completed"
    )

    stage_start = _record_timing(timings, 
        "RapidOCR",
        stage_start,
    )

    # =====================================================
    # Stage 2.5 : Build Block Knowledge
    # =====================================================

    print()
    print(
        "Building Block Knowledge..."
    )

    stage_start = time.perf_counter()

    knowledge_builder = (
        BlockKnowledgeBuilder()
    )

    knowledge_map = {}

    for block in blocks:

        knowledge = (
            knowledge_builder.build(
                block,
                page_number=page_number,
            )
        )

        block.knowledge = (
            knowledge
        )

        knowledge_map[
            block.id
        ] = knowledge

    print(
        f"Generated Block Knowledge : "
        f"{len(knowledge_map)}"
    )

    stage_start = _record_timing(timings, 
        "Block knowledge",
        stage_start,
    )

    # =====================================================
    # Stage 2.6 : Page Cleaner
    # =====================================================

    print()
    print(
        "Cleaning page blocks..."
    )

    stage_start = time.perf_counter()

    page_height, page_width = (
        results[0].orig_shape
    )

    clean_blocks = (
        PageCleaner().clean(
            blocks=blocks,
            page_width=page_width,
            page_height=page_height,
            page_number=page_number,
            is_rtl=is_rtl,
        )
    )

    print(
        f"✓ Clean blocks : "
        f"{len(clean_blocks)}"
    )

    stage_start = _record_timing(timings, 
        "Page cleaning",
        stage_start,
    )

    # =====================================================
    # Stage 2.7 : Export Clean Page JSON
    # =====================================================

    print()
    print(
        "Exporting Page JSON..."
    )

    stage_start = time.perf_counter()

    json_path = (
        GeminiPageExporter().export(
            page_number=page_number,
            page_width=page_width,
            page_height=page_height,
            blocks=clean_blocks,
            output_dir=(
                document_dir
                / "page_json"
            ),
        )
    )

    print(
        f"✓ Page JSON saved -> "
        f"{json_path}"
    )

    stage_start = _record_timing(timings,
        "Page JSON export",
        stage_start,
    )

    return {
        "page_number": page_number,
        "page_path": page_path,
        "document_dir": document_dir,
        "json_path": json_path,
        "clean_blocks": clean_blocks,
        "knowledge_map": knowledge_map,
        "timings": timings,
        "page_start": page_start,
    }


# =========================================================
# RUN GEMINI (the network call, independent per page --
# safe to run concurrently across a document's pages)
# =========================================================

def run_gemini(page_path, json_path, prompt=HINDI_GROUPING_PROMPT):

    print()
    print(
        "Running Gemini..."
    )

    service = GeminiService()

    gemini_start = time.perf_counter()

    gemini_response = (
        service.analyze_page(
            image_path=page_path,
            json_path=json_path,
            prompt=prompt,
        )
    )

    gemini_elapsed = (
        time.perf_counter()
        - gemini_start
    )

    return gemini_response, gemini_elapsed


# =========================================================
# RUN OPENAI (the network call, independent per page --
# safe to run concurrently across a document's pages)
# =========================================================

def run_openai(page_path, json_path, prompt=ENGLISH_GROUPING_PROMPT):

    print()
    print(
        "Running OpenAI..."
    )

    service = OpenAIService(
        model=os.getenv(
            "OPENAI_BOUNDARY_MODEL",
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        ),
    )

    openai_start = time.perf_counter()

    openai_response = (
        service.analyze_page(
            image_path=page_path,
            json_path=json_path,
            prompt=prompt,
        )
    )

    openai_elapsed = (
        time.perf_counter()
        - openai_start
    )

    return openai_response, openai_elapsed


# =========================================================
# LOCAL GROUPING (NO API CALL)
# =========================================================

def run_local(page_path, json_path, prompt=None):
    """
    Drop-in replacement for run_openai / run_gemini that groups the
    page with pipeline.article.local_grouper instead of a model.

    Returns the same (response, elapsed) pair and the same response
    shape, so finish_page and every stage after it are unchanged.

    Accepts (and ignores) `prompt` purely so callers can invoke
    run_local/run_gemini/run_openai interchangeably with the same
    keyword arguments -- local mode never calls an LLM, so there is
    no prompt to send.
    """

    print()
    print(
        "Grouping locally (no API call)..."
    )

    from pipeline.article.local_grouper import (
        build_local_response,
    )

    local_start = time.perf_counter()

    local_response = build_local_response(
        json_path,
        page_image_path=page_path,
    )

    local_elapsed = (
        time.perf_counter()
        - local_start
    )

    print(
        f"Local grouping: "
        f"{len(local_response.get('articles', []))} article(s) "
        f"in {local_elapsed:.2f}s"
    )

    return local_response, local_elapsed


# =========================================================
# FINISH PAGE (everything after the network call)
# =========================================================

def finish_page(
    prep,
    gemini_response,
    gemini_elapsed,
    use_contested_block_arbitration: bool = True,
    use_orphan_block_reassignment: bool = True,
    use_orphan_title_root_repair: bool = True,
    use_unclaimed_kicker_recovery: bool = True,
    use_unclaimed_image_recovery: bool = True,
    use_unclaimed_footprint_recovery: bool = True,
    use_article_splitter: bool = True,
):

    page_number = prep["page_number"]
    page_path = prep["page_path"]
    document_dir = prep["document_dir"]
    json_path = prep["json_path"]
    clean_blocks = prep["clean_blocks"]
    knowledge_map = prep["knowledge_map"]
    timings = prep["timings"]
    page_start = prep["page_start"]

    timings["LLM API"] = gemini_elapsed

    # -----------------------------------------------------
    # Save LLM response
    # -----------------------------------------------------

    stage_start = time.perf_counter()

    response_dir = (
        document_dir
        / "llm_response"
    )

    response_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    response_path = (
        response_dir
        / f"page_{page_number:03d}_response.json"
    )

    with open(
        response_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            gemini_response,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"✓ LLM response saved -> "
        f"{response_path}"
    )

    stage_start = _record_timing(timings,
        "LLM response save",
        stage_start,
    )

    # =====================================================
    # Stage 3 : LLM Boundary Pipeline
    # =====================================================

    print()
    print("=" * 60)
    print(
        "LLM BOUNDARY PIPELINE"
    )
    print("=" * 60)

    final_dir = (
        document_dir
        / "final"
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        final_dir
        / f"page_{page_number:03d}"
          "_final_boundaries.png"
    )

    # -----------------------------------------------------
    # Boundary timer
    # -----------------------------------------------------

    boundary_start = time.perf_counter()

    boundaries = (
        GeminiBoundaryPipeline().run(
            image_path=page_path,
            page_json_path=str(
                json_path
            ),
            gemini_response_path=str(
                response_path
            ),
            output_path=str(
                output_path
            ),
            use_contested_block_arbitration=(
                use_contested_block_arbitration
            ),
            use_orphan_block_reassignment=(
                use_orphan_block_reassignment
            ),
            use_orphan_title_root_repair=(
                use_orphan_title_root_repair
            ),
            use_unclaimed_kicker_recovery=(
                use_unclaimed_kicker_recovery
            ),
            use_unclaimed_image_recovery=(
                use_unclaimed_image_recovery
            ),
            use_unclaimed_footprint_recovery=(
                use_unclaimed_footprint_recovery
            ),
            use_article_splitter=use_article_splitter,
        )
    )

    boundary_elapsed = (
        time.perf_counter()
        - boundary_start
    )

    timings["Boundary pipeline"] = boundary_elapsed

    print()
    print("=" * 60)
    print(
        "LLM PIPELINE COMPLETED"
    )
    print("=" * 60)

    # =====================================================
    # Print Final Boundary Objects
    # =====================================================

    print()
    print("=" * 60)
    print(
        "FINAL BOUNDARY OBJECTS"
    )
    print("=" * 60)

    for i, boundary in enumerate(
        boundaries,
        start=1,
    ):

        print()

        print(
            f"Boundary {i}"
        )

        print(
            f"Type : {type(boundary)}"
        )

        print(
            boundary.__dict__
        )

    # =====================================================
    # Stage 3.5 : FINAL ARTICLE CROPS
    # =====================================================

    print()
    print("=" * 60)
    print(
        "FINAL ARTICLE CROPS"
    )
    print("=" * 60)

    final_cropper = (
        FinalArticleCropper()
    )

    # -----------------------------------------------------
    # Crop timer
    # -----------------------------------------------------

    crop_start = time.perf_counter()

    final_article_crops = (
        final_cropper.crop_articles(
            document_dir=document_dir,
            page_number=page_number,
            boundaries=boundaries,
            page_image_path=page_path,
            page_json_path=str(
                json_path
            ),
        )
    )

    crop_elapsed = (
        time.perf_counter()
        - crop_start
    )

    timings["Article cropping"] = crop_elapsed

    print()

    print(
        f"✓ Created "
        f"{len(final_article_crops)} "
        f"final article crops"
    )

    # =====================================================
    # Print Crop Information
    # =====================================================

    print()
    print("=" * 60)
    print(
        "ARTICLE CROPS CREATED"
    )
    print("=" * 60)

    for crop in final_article_crops:

        print()

        print(
            f"Article ID : "
            f"{crop.get('article_id')}"
        )

        print(
            f"Page       : "
            f"{crop.get('page')}"
        )

        print(
            f"BBox       : "
            f"{crop.get('bbox')}"
        )

        print(
            f"Size       : "
            f"{crop.get('width')} "
            f"x "
            f"{crop.get('height')}"
        )

        print(
            f"Crop       : "
            f"{crop.get('crop_path')}"
        )

    # =====================================================
    # Stage 3.6 : FINAL ARTICLE CROPS
    # =====================================================

    print()
    print("=" * 60)
    print(
        "FINAL ARTICLE CROPS"
    )
    print("=" * 60)

    print(
        "✓ Final article crop OCR: SKIPPED"
    )

    print(
        "✓ The LLM will read final crop "
        "images directly"
    )

    # =====================================================
    # TOTAL TIME + FINAL TIMING SUMMARY
    # =====================================================

    total_elapsed = (
        time.perf_counter()
        - page_start
    )

    timings["TOTAL PAGE TIME"] = total_elapsed

    print()
    print("=" * 60)
    print("PIPELINE TIMING SUMMARY")
    print("=" * 60)

    timing_order = [
        "Layout detection",
        "RapidOCR",
        "Block knowledge",
        "Page cleaning",
        "Page JSON export",
        "LLM API",
        "LLM response save",
        "Boundary pipeline",
        "Article cropping",
    ]

    for label in timing_order:
        if label in timings:
            print(
                f"{label:<25} : "
                f"{timings[label]:8.2f} sec"
            )

    print("-" * 60)

    print(
        f"TOTAL PAGE TIME           : "
        f"{total_elapsed:8.2f} sec"
    )

    print("=" * 60)

    # =====================================================
    # RETURN
    # =====================================================

    return {
        "blocks": clean_blocks,

        "knowledge": knowledge_map,

        "page_json": json_path,

        "gemini_response": gemini_response,

        "gemini_response_path": response_path,

        "boundaries": boundaries,

        "final_article_crops": (
            final_article_crops
        ),
    }


# =========================================================
# PROCESS PAGE (single-page convenience wrapper --
# prepare + run the configured LLM synchronously + finish)
# =========================================================

def process_page(
    page_number,
    page_path,
    detector,
    ocr_engine,
    document_id,
    document_dir,
):

    prep = prepare_page(
        page_number=page_number,
        page_path=page_path,
        detector=detector,
        ocr_engine=ocr_engine,
        document_id=document_id,
        document_dir=document_dir,
    )

    is_hindi = (
        os.getenv("LLM_PROVIDER", "openai").strip().lower() == "gemini"
    )

    run_page_llm = run_gemini if is_hindi else run_openai

    llm_response, llm_elapsed = run_page_llm(
        page_path=page_path,
        json_path=prep["json_path"],
    )

    return finish_page(
        prep,
        llm_response,
        llm_elapsed,
        use_contested_block_arbitration=is_hindi,
    )