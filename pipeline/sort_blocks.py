def sort_blocks(blocks, is_rtl=False):
    """
    Sort blocks from top to bottom, then left to right (or,
    for a right-to-left script, top to bottom then RIGHT to left).

    This order becomes each block's `reading_order` when the page is
    exported (pipeline/export/gemini_page_exporter.py enumerates
    blocks in this exact order), and the grouping prompt is told to
    use reading_order as a tie-breaker. Leaving it left-to-right for
    an RTL document -- e.g. Urdu -- would hand the model a reading
    order that runs backwards through every row.
    """

    return sorted(
        blocks,
        key=lambda b: (
            b.y1,
            -b.x1 if is_rtl else b.x1,
        ),
    )
