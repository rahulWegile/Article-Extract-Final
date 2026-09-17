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

# Some source PDFs (confirmed on real Punjabi Jagran e-paper PDFs)
# declare a MediaBox far smaller than an actual physical newspaper
# page -- e.g. 401x655pt (~5.6x9.1in) -- while embedding their full
# scanned page image inside it at exactly that "300 DPI" pixel count
# (1671x2730px). Rendering "300 DPI" against that undersized MediaBox
# is internally consistent but leaves a single newspaper column only
# a few hundred pixels wide -- and fine 8pt Gurmukhi body text
# blurs first -- too little for the vision model to resolve small
# matras/conjuncts, independent of anything OCR- or
# extraction-prompt-side. MuPDF rasterizes past an image's native
# resolution on request (confirmed: the same page at 600/1200 "dpi"
# renders to 2x/4x the pixels via resampling), so a floor on the
# long edge recovers a true-300-DPI-equivalent render (~2800x4500px
# on this newspaper's page proportions) for these pages while
# leaving already-large broadsheet pages -- which already clear this
# floor at true 300 DPI -- untouched.
MIN_LONG_EDGE_PIXELS = 4500


def render_pdf(pdf_path, output_dir, dpi=300):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    saved = []

    for i, page in enumerate(doc):

        page_zoom = dpi / 72
        width, height = page.rect.width, page.rect.height

        # Boost undersized pages so the long edge reaches the
        # legibility floor before the oversized-page cap below.
        long_edge = max(width, height) * page_zoom
        if long_edge < MIN_LONG_EDGE_PIXELS:
            page_zoom *= MIN_LONG_EDGE_PIXELS / long_edge

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