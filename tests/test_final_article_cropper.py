from pathlib import Path

from pipeline.intelligence.final_article_cropper import (
    FinalArticleCropper,
)

from pipeline.gemini.gemini_boundary_pipeline import (
    GeminiBoundaryPipeline,
)


DOCUMENT_DIR = Path(
    "output/documents/doc_000001"
)

PAGE_NUMBER = 3

PAGE_IMAGE = (
    DOCUMENT_DIR
    / "pages"
    / f"page_{PAGE_NUMBER:03d}.png"
)

PAGE_JSON = (
    DOCUMENT_DIR
    / "page_json"
    / f"page_{PAGE_NUMBER:03d}.json"
)

GEMINI_RESPONSE = (
    DOCUMENT_DIR
    / "gemini"
    / f"page_{PAGE_NUMBER:03d}_response.json"
)

OUTPUT_IMAGE = (
    DOCUMENT_DIR
    / "final"
    / f"page_{PAGE_NUMBER:03d}_final_boundaries.png"
)


def main():

    print()
    print("=" * 70)
    print("EXTRACT FINAL ARTICLE CROPS")
    print("=" * 70)

    pipeline = GeminiBoundaryPipeline()

    boundaries = pipeline.run(
        image_path=str(PAGE_IMAGE),
        page_json_path=str(PAGE_JSON),
        gemini_response_path=str(GEMINI_RESPONSE),
        output_path=str(OUTPUT_IMAGE),
    )

    print()
    print(
        f"Final boundaries found: "
        f"{len(boundaries)}"
    )

    cropper = FinalArticleCropper()

    results = cropper.crop_articles(
        document_dir=DOCUMENT_DIR,
        page_number=PAGE_NUMBER,
        boundaries=boundaries,
    )

    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)

    print(
        f"Articles extracted: "
        f"{len(results)}"
    )

    print(
        "Output directory:"
    )

    print(
        DOCUMENT_DIR
        / "final_articles_crops"
    )

    print("=" * 70)

    # =====================================================
    # block_ids REGRESSION CHECK
    #
    # crop.json must persist the source block_ids so the
    # local article extractor can map figure/figure_caption
    # blocks onto each article crop.
    # =====================================================

    for result in results:

        assert "block_ids" in result, (
            f"crop.json missing 'block_ids' for "
            f"{result.get('article_id')}"
        )

        print(
            f"{result['article_id']} "
            f"block_ids={result['block_ids']}"
        )


if __name__ == "__main__":
    main()