import json

import cv2

from pipeline.block_parser import LayoutBlock

from pipeline.gemini.gemini_parser import GeminiParser

from pipeline.article.article_grouper import (
    ArticleGrouper,
)

from pipeline.article.article_splitter import (
    split_oversized_articles,
)

from pipeline.article.boundary_builder import (
    BoundaryBuilder,
)

from pipeline.visualization.draw_final_boundaries import (
    FinalBoundaryVisualizer,
)


class GeminiBoundaryPipeline:
    """
    Build final article boundaries from
    Gemini grouped articles.
    """

    # -----------------------------------------------------

    def _load_blocks(
        self,
        page_json_path,
    ):

        with open(
            page_json_path,
            "r",
            encoding="utf-8",
        ) as f:

            page = json.load(f)

        blocks = []

        for item in page["blocks"]:

            block = LayoutBlock(

                id=item["id"],

                cls=item["class"],

                x1=item["bbox"]["x1"],
                y1=item["bbox"]["y1"],
                x2=item["bbox"]["x2"],
                y2=item["bbox"]["y2"],

                confidence=1.0,

            )

            block.text = item.get(
                "text",
                "",
            )

            block.reading_order = item.get(
                "reading_order",
                0,
            )

            blocks.append(
                block
            )

        return blocks

    # -----------------------------------------------------

    def run(
        self,
        image_path,
        page_json_path,
        gemini_response_path,
        output_path,
        is_hindi: bool = True,
    ):

        print()
        print("=" * 60)
        print("GEMINI BOUNDARY PIPELINE")
        print("=" * 60)

        #
        # Load page blocks
        #

        blocks = self._load_blocks(
            page_json_path,
        )

        #
        # Load Gemini response
        #

        with open(
            gemini_response_path,
            "r",
            encoding="utf-8",
        ) as f:

            response = json.load(f)

        #
        # Parse Gemini response
        #

        parser = GeminiParser()

        parsed = parser.parse(

            response,

            blocks,

        )

        #
        # Build articles
        #

        grouper = ArticleGrouper()

        articles = grouper.build(

            parsed,

            blocks,

            is_hindi=is_hindi,

        )

        #
        # Split any article the model over-merged: re-checks any
        # article containing more than one title block against
        # local_grouper's own geometric root-detection, splitting it
        # only when that algorithm actually finds separate stories
        # (never touches ArticleGrouper's or the LLM's own decision
        # otherwise -- see article_splitter.py).
        #
        # Hindi-only: the reference English pipeline has no
        # equivalent step, so English keeps the model's own grouping
        # as-is here.
        #

        if is_hindi:

            try:
                page_image = cv2.imread(str(image_path))
            except Exception:
                page_image = None

            articles = split_oversized_articles(
                articles,
                page_image,
            )

        #
        # Build boundaries
        #

        builder = BoundaryBuilder()

        boundaries = builder.build(

            articles,

        )

        #
        # Render boundaries
        #

        FinalBoundaryVisualizer().save(

            image_path=image_path,

            boundaries=boundaries,

            output_path=output_path,

        )

        print()
        print("=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Articles Built   : {len(articles)}")
        print(f"Boundaries Built : {len(boundaries)}")
        print(f"Saved            : {output_path}")
        print("=" * 60)
        print()

        return boundaries