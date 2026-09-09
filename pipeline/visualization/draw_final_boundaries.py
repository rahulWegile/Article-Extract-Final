import os
import cv2


class FinalBoundaryVisualizer:
    """
    Draw the final article boundaries.

    Every article -- however many layout blocks/columns/headline
    pieces/images it took to compose it -- gets exactly ONE
    rectangle: its plain outer union boundary (see
    article_region_reconstructor.py, STEP 8). Internal whitespace,
    and even an incidental overlap with a neighbouring article's
    rectangle, is drawn as-is; nothing here fragments one article
    into several displayed boxes.
    """

    def save(
        self,
        image_path,
        boundaries,
        output_path,
    ):

        image = cv2.imread(str(image_path))

        if image is None:

            raise ValueError(
                f"Cannot load image: {image_path}"
            )

        os.makedirs(

            os.path.dirname(output_path),

            exist_ok=True,

        )

        for boundary in boundaries:

            x1 = int(boundary.x1)
            y1 = int(boundary.y1)
            x2 = int(boundary.x2)
            y2 = int(boundary.y2)

            cv2.rectangle(

                image,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),      # Green

                3,

            )

            #
            # Label
            #

            label = (

                f"A{boundary.article_id}"

                f" ({len(boundary.block_ids)} blocks)"

            )

            cv2.putText(

                image,

                label,

                (x1, max(20, y1 - 10)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 0, 0),

                2,

                cv2.LINE_AA,

            )

        cv2.imwrite(

            str(output_path),

            image,

        )

        print()

        print("=" * 60)

        print("FINAL BOUNDARY VISUALIZATION")

        print("=" * 60)

        print(f"Saved -> {output_path}")

        print("=" * 60)

        print()

    # Cycling palette (BGR, OpenCV order) used to give each article a
    # visually distinct member-block color in debug view. Plain,
    # high-contrast colors -- there is no meaning to which article
    # gets which color, only that neighbouring articles are easy to
    # tell apart by eye.
    _DEBUG_PALETTE = [
        (255, 0, 0),      # blue
        (0, 128, 255),    # orange
        (0, 0, 255),      # red
        (255, 0, 255),    # magenta
        (0, 200, 200),    # yellow-ish
        (200, 0, 200),    # purple
        (0, 165, 255),    # amber
        (128, 255, 0),    # spring green
    ]

    def save_debug_view(
        self,
        image_path,
        blocks,
        articles,
        boundaries,
        output_path,
    ):
        """
        Debug rendering (STEP 12): draws, in one image --

        1. every ORIGINAL detected block, thin gray, regardless of
           whether it ended up in any article (so a dropped/unclaimed
           block is visible too);
        2. every article's own MEMBER blocks, colored per article_id
           (a color cycling through _DEBUG_PALETTE identifies which
           blocks a given article claims);
        3. each article's final OUTER boundary, drawn thick in that
           same color;
        4. any article whose outer boundary falls ENTIRELY inside
           another article's outer boundary -- the visual signature
           of a nested/independent sub-article sitting inside a
           parent's region -- gets a dashed outline instead of solid,
           so the two are easy to tell apart by eye while confirming
           neither absorbed the other's block_ids.

        `blocks` is the full page's detected block list (used for
        layer 1); `articles` supplies each boundary's member blocks
        for layers 2 and 4.
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Layer 1: every original detected block, thin gray.
        for block in blocks:
            cv2.rectangle(
                image,
                (int(block.x1), int(block.y1)),
                (int(block.x2), int(block.y2)),
                (160, 160, 160),
                1,
            )

        articles_by_id = {a.article_id: a for a in articles}

        def is_nested(boundary, others):
            for other in others:
                if other.article_id == boundary.article_id:
                    continue
                if (
                    boundary.x1 >= other.x1
                    and boundary.y1 >= other.y1
                    and boundary.x2 <= other.x2
                    and boundary.y2 <= other.y2
                ):
                    return True
            return False

        def dashed_rectangle(img, pt1, pt2, color, thickness, dash=18):

            x1, y1 = pt1
            x2, y2 = pt2

            for x in range(x1, x2, dash * 2):
                cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
                cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)

            for y in range(y1, y2, dash * 2):
                cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
                cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)

        for index, boundary in enumerate(boundaries):

            color = self._DEBUG_PALETTE[index % len(self._DEBUG_PALETTE)]

            article = articles_by_id.get(boundary.article_id)

            # Layer 2: this article's own member blocks.
            if article is not None:
                for block in article.blocks:
                    cv2.rectangle(
                        image,
                        (int(block.x1), int(block.y1)),
                        (int(block.x2), int(block.y2)),
                        color,
                        2,
                    )

            # Layer 3/4: the outer boundary -- dashed when it sits
            # entirely inside another article's boundary (nested/
            # independent sub-article), solid otherwise.
            x1, y1, x2, y2 = (
                int(boundary.x1),
                int(boundary.y1),
                int(boundary.x2),
                int(boundary.y2),
            )

            if is_nested(boundary, boundaries):
                dashed_rectangle(image, (x1, y1), (x2, y2), color, 3)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

            label = f"A{boundary.article_id} ({len(boundary.block_ids)} blocks)"

            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(output_path), image)

        print()
        print("=" * 60)
        print("DEBUG REGION VISUALIZATION")
        print("=" * 60)
        print(f"Saved -> {output_path}")
        print("=" * 60)
        print()