import io
import re
from pathlib import Path

import fitz
from PIL import Image

MAX_DIMENSION = 2000
JPEG_QUALITY = 80


def export_document_pdf(document_dir):
    document_dir = Path(document_dir)
    final_dir = document_dir / "final"

    pages = sorted(
        final_dir.glob("page_*_final_boundaries.png"),
        key=lambda p: int(re.search(r"page_(\d+)_final_boundaries", p.name).group(1)),
    )

    if not pages:
        raise FileNotFoundError(
            f"No boundary pages found in {final_dir}"
        )

    output_path = final_dir / "boundaries.pdf"

    doc = fitz.open()

    for image_path in pages:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            scale = min(1.0, MAX_DIMENSION / max(img.width, img.height))
            if scale < 1.0:
                img = img.resize(
                    (round(img.width * scale), round(img.height * scale)),
                    Image.LANCZOS,
                )

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY)

            page = doc.new_page(width=img.width, height=img.height)
            page.insert_image(page.rect, stream=buffer.getvalue())

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    return output_path
