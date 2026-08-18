def sort_blocks(blocks):
    """
    Sort blocks from top to bottom, then left to right.
    """

    return sorted(
        blocks,
        key=lambda b: (
            b.y1,
            b.x1,
        ),
    )