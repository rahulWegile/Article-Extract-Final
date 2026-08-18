from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, ImageDraw


class ImageCompositor:
    """
    Compose actual article images into one final image.

    Example:

        Image A              Image B
        ┌──────────┐        ┌──────────┐
        │          │        │          │
        │  PHOTO   │        │  PHOTO   │
        │          │        │          │
        └──────────┘        └──────────┘

                    ↓

        ┌────────────────────────────────┐
        │       PHOTO A   |   PHOTO B    │
        └────────────────────────────────┘

    Images are placed horizontally by default.
    """

    def __init__(
        self,
        background="white",
        gap=20,
        padding=20,
        max_height=1000,
        add_labels=False,
    ):
        self.background = background
        self.gap = gap
        self.padding = padding
        self.max_height = max_height
        self.add_labels = add_labels

    # =========================================================
    # LOAD
    # =========================================================

    @staticmethod
    def load_image(
        path: str | Path,
    ) -> Image.Image:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        return Image.open(path).convert("RGB")

    # =========================================================
    # RESIZE
    # =========================================================

    @staticmethod
    def resize_to_height(
        image: Image.Image,
        target_height: int,
    ) -> Image.Image:

        if image.height == target_height:
            return image

        scale = (
            target_height
            / image.height
        )

        width = max(
            1,
            int(
                image.width * scale
            ),
        )

        return image.resize(
            (width, target_height),
            Image.Resampling.LANCZOS,
        )

    # =========================================================
    # BORDER
    # =========================================================

    @staticmethod
    def add_border(
        image: Image.Image,
    ) -> Image.Image:

        return ImageOps.expand(
            image,
            border=1,
            fill="black",
        )

    # =========================================================
    # HORIZONTAL COMPOSITION
    # =========================================================

    def compose_horizontal(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
    ) -> str:

        if not image_paths:
            raise ValueError(
                "No images supplied."
            )

        images = []

        for path in image_paths:

            try:
                image = self.load_image(
                    path
                )

                if image.width < 2:
                    continue

                if image.height < 2:
                    continue

                image = self.add_border(
                    image
                )

                images.append(image)

            except Exception as exc:

                print(
                    f"WARNING: Failed to load "
                    f"{path}: {exc}"
                )

        if not images:
            raise RuntimeError(
                "No valid images available."
            )

        # -----------------------------------------------------
        # Normalize height
        # -----------------------------------------------------

        natural_max_height = max(
            image.height
            for image in images
        )

        target_height = min(
            natural_max_height,
            self.max_height,
        )

        normalized = []

        for image in images:

            normalized.append(
                self.resize_to_height(
                    image,
                    target_height,
                )
            )

        # -----------------------------------------------------
        # Canvas size
        # -----------------------------------------------------

        content_width = sum(
            image.width
            for image in normalized
        )

        content_width += (
            self.gap
            * (
                len(normalized) - 1
            )
        )

        canvas_width = (
            content_width
            + self.padding * 2
        )

        canvas_height = (
            target_height
            + self.padding * 2
        )

        canvas = Image.new(
            "RGB",
            (
                canvas_width,
                canvas_height,
            ),
            self.background,
        )

        # -----------------------------------------------------
        # Paste
        # -----------------------------------------------------

        x = self.padding

        for index, image in enumerate(
            normalized
        ):

            y = (
                self.padding
                + (
                    target_height
                    - image.height
                )
                // 2
            )

            canvas.paste(
                image,
                (x, y),
            )

            x += image.width

            if index < len(
                normalized
            ) - 1:

                x += self.gap

        # -----------------------------------------------------
        # Save
        # -----------------------------------------------------

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        canvas.save(
            output_path,
            format="JPEG",
            quality=95,
            optimize=True,
        )

        return str(
            output_path
        )

    # =========================================================
    # ARTICLE IMAGE COMPOSITION
    # =========================================================

    def compose_article_images(
        self,
        image_paths: list[str | Path],
        output_path: str | Path,
    ) -> str | None:

        if not image_paths:
            return None

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # One image.
        if len(image_paths) == 1:

            image = self.load_image(
                image_paths[0]
            )

            image.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

            return str(
                output_path
            )

        return self.compose_horizontal(
            image_paths=image_paths,
            output_path=output_path,
        )
