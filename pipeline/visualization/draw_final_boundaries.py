import os
import cv2


class FinalBoundaryVisualizer:
    """
    Draw the final Gemini article boundaries.
    """

    def save(
        self,
        image_path,
        boundaries,
        output_path,
    ):

        image = cv2.imread(image_path)

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

            #
            # Draw boundary. An article boundary_decomposer.py gave a
            # multi-rectangle shape (see Article.sub_rects) is drawn
            # as its precise pieces instead of the single overall
            # rectangle, which would still visually cross whatever
            # neighbouring article it was reshaped to avoid.
            #

            sub_rects = getattr(boundary, "sub_rects", None) or []

            rects_to_draw = sub_rects if sub_rects else [(x1, y1, x2, y2)]

            for rx1, ry1, rx2, ry2 in rects_to_draw:

                cv2.rectangle(

                    image,

                    (int(rx1), int(ry1)),

                    (int(rx2), int(ry2)),

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

            output_path,

            image,

        )

        print()

        print("=" * 60)

        print("FINAL BOUNDARY VISUALIZATION")

        print("=" * 60)

        print(f"Saved -> {output_path}")

        print("=" * 60)

        print()