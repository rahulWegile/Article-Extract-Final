import fitz
from pathlib import Path

# Was 4000*4000 (16M px), which forced a broadsheet page down to an
# effective ~236 DPI instead of the requested 300 (a 977x1528pt page
# needs 25.9M px at true 300 DPI). Devanagari matras and conjuncts
# are exactly the detail that downscale loses first, and it fed both
# the OCR step and the per-block vision transcription a softer image
# than necessary for no real reason -- a 6000x6000 pixmap is well
# within what MuPDF and available memory handle (measured ~80MB per
# page at that size). Raised with headroom above a large broadsheet
# at true 300 DPI; the halving fallback below still protects against
# a page that's unusually large even by that standard.
MAX_PIXELS = 6000 * 6000


def render_pdf(pdf_path, output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    saved = []

    for i, page in enumerate(doc):

        page_zoom = dpi / 72
        width, height = page.rect.width, page.rect.height

        # Scale down the zoom for oversized pages so the rendered
        # pixmap stays under MuPDF's internal pixel-size limit.
        pixels = (width * page_zoom) * (height * page_zoom)
        if pixels > MAX_PIXELS:
            page_zoom *= (MAX_PIXELS / pixels) ** 0.5

        matrix = fitz.Matrix(page_zoom, page_zoom)

        try:
            pix = page.get_pixmap(matrix=matrix)
        except RuntimeError:
            # Fallback: MuPDF still rejected it, halve the zoom and retry once.
            matrix = fitz.Matrix(page_zoom / 2, page_zoom / 2)
            pix = page.get_pixmap(matrix=matrix)

        filename = output_dir / f"page_{i+1:03}.png"

        pix.save(filename)

        saved.append(filename)

    print(f"Saved {len(saved)} pages")

    return saved