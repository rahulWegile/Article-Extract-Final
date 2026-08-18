import json
import os
import time

from pipeline.block_parser import parse_results
from pipeline.sort_blocks import sort_blocks

from pipeline.knowledge.block_knowledge_builder import (
    BlockKnowledgeBuilder,
)

from pipeline.preprocess.page_cleaner import PageCleaner

from pipeline.export.gemini_page_exporter import (
    GeminiPageExporter,
)

from pipeline.openai.openai_service import OpenAIService
from pipeline.gemini.gemini_prompt import ARTICLE_GROUP_PROMPT

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
# PROCESS PAGE
# =========================================================

def process_page(
    page_number,
    page_path,
    detector,
    ocr_engine,
    document_id,
    document_dir,
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

    results = detector.detect(
        page_path
    )

    blocks = parse_results(
        results
    )

    print(
        f"Detected {len(blocks)} layout blocks"
    )

    blocks = sort_blocks(
        blocks
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

    # =====================================================
    # Stage 2.8 : OpenAI Article Analysis
    # =====================================================

    print()
    print(
        "Running OpenAI..."
    )

    # gpt-4o-mini is not reliable at keeping block-to-article
    # ownership exclusive on dense newspaper pages (verified: it
    # collapsed a ~8-story front page into 3 overlapping articles).
    # This stage gets its own, stronger default model.
    service = OpenAIService(
        model=os.getenv("OPENAI_BOUNDARY_MODEL", "gpt-4o"),
    )

    # -----------------------------------------------------
    # OpenAI API timer
    # -----------------------------------------------------

    gemini_start = time.perf_counter()

    gemini_response = (
        service.analyze_page(
            image_path=page_path,
            json_path=json_path,
            prompt=ARTICLE_GROUP_PROMPT,
        )
    )

    gemini_elapsed = (
        time.perf_counter()
        - gemini_start
    )

    timings["OpenAI API"] = gemini_elapsed

    # -----------------------------------------------------
    # Save OpenAI response
    # -----------------------------------------------------

    stage_start = time.perf_counter()

    response_dir = (
        document_dir
        / "gemini"
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
        f"✓ OpenAI response saved -> "
        f"{response_path}"
    )

    stage_start = _record_timing(timings, 
        "OpenAI response save",
        stage_start,
    )

    # =====================================================
    # Stage 3 : Gemini Boundary Pipeline
    # =====================================================

    print()
    print("=" * 60)
    print(
        "GEMINI BOUNDARY PIPELINE"
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
        "GEMINI PIPELINE COMPLETED"
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
        "✓ OpenAI will read final crop "
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
        "OpenAI API",
        "OpenAI response save",
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