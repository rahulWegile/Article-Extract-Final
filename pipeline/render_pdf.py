import fitz
from pathlib import Path


def render_pdf(pdf_path, output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    saved = []

    for i, page in enumerate(doc):

        pix = page.get_pixmap(matrix=matrix)

        filename = output_dir / f"page_{i+1:03}.png"

        pix.save(filename)

        saved.append(filename)

    print(f"Saved {len(saved)} pages")

    return saved