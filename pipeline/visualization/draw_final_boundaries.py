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
            # Draw boundary
            #

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