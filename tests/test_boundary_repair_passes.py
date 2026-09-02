"""
Invariants for the two boundary-repair passes added to stop a page
losing a whole story, and to stop one article's rectangle being drawn
over another's.

These are the guarantees the passes were measured against on the 116
stored pages in output/documents; they are asserted here on synthetic
pages so a later change cannot quietly give them up.

Run with:  python -m pytest tests/test_boundary_repair_passes.py -q
"""

from pipeline.block_parser import LayoutBlock
from pipeline.article.article_grouper import Article
from pipeline.article.dropped_article_recovery import (
    recover_dropped_articles,
)
from pipeline.article.boundary_decomposer import (
    decompose_overlapping_articles,
)


PAGE_W = 4000
PAGE_H = 6000


def block(bid, cls, x1, y1, x2, y2, text="", role=None):
    b = LayoutBlock(
        id=bid,
        cls=cls,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=1.0,
    )
    b.text = text
    b.reading_order = bid
    if role is not None:
        b.role = role
    return b


def prose(n):
    return "x" * n


def article_of(blocks, article_id=1):
    return Article(
        article_id=article_id,
        blocks=list(blocks),
        block_ids=[b.id for b in blocks],
        confidence=1.0,
    )


def story_blocks(base_id, x1, y1, role="advertisement"):
    """A story-shaped cluster: one headline plus four fat paragraphs."""
    out = [
        block(
            base_id,
            "title",
            x1,
            y1,
            x1 + 900,
            y1 + 200,
            "A real headline that was dropped",
            role,
        )
    ]
    for i in range(4):
        top = y1 + 220 + i * 420
        out.append(
            block(
                base_id + 1 + i,
                "plain text",
                x1,
                top,
                x1 + 900,
                top + 400,
                prose(400),
                role,
            )
        )
    return out


def existing_article(base_id=100, x1=200, y1=200):
    blocks = [
        block(base_id, "title", x1, y1, x1 + 900, y1 + 150, "Kept story"),
        block(
            base_id + 1,
            "plain text",
            x1,
            y1 + 170,
            x1 + 900,
            y1 + 800,
            prose(500),
        ),
    ]
    return blocks


# ------------------------------------------------------------------
# dropped_article_recovery
# ------------------------------------------------------------------


def test_recovers_a_story_dropped_as_advertisement():
    kept = existing_article()
    dropped = story_blocks(1, 2400, 3000)

    articles = [article_of(kept)]

    result = recover_dropped_articles(articles, kept + dropped)

    assert len(result) == 2
    assert set(result[1].block_ids) == {b.id for b in dropped}


def test_recovers_a_forecast_report_dropped_as_weather():
    kept = existing_article()
    dropped = story_blocks(1, 2400, 3000, role="weather")

    result = recover_dropped_articles([article_of(kept)], kept + dropped)

    assert len(result) == 2


def test_existing_articles_are_never_modified():
    kept = existing_article()
    dropped = story_blocks(1, 2400, 3000)

    original = article_of(kept)
    before = list(original.block_ids)

    result = recover_dropped_articles([original], kept + dropped)

    assert result[0] is original
    assert result[0].block_ids == before


def test_no_recovery_when_the_page_has_no_articles_at_all():
    """A full-page advertisement stays a full-page advertisement."""
    dropped = story_blocks(1, 200, 200)

    assert recover_dropped_articles([], dropped) == []


def test_utility_box_grid_is_never_recovered():
    """An almanac/listings grid must not become an article."""
    kept = existing_article()

    grid = []
    bid = 1
    for row in range(6):
        for col in range(3):
            x1 = 2200 + col * 550
            y1 = 2000 + row * 300
            grid.append(
                block(
                    bid,
                    "title",
                    x1,
                    y1,
                    x1 + 500,
                    y1 + 90,
                    "Grid cell heading",
                    "utility_box",
                )
            )
            grid.append(
                block(
                    bid + 1,
                    "plain text",
                    x1,
                    y1 + 100,
                    x1 + 500,
                    y1 + 280,
                    prose(300),
                    "utility_box",
                )
            )
            bid += 2

    result = recover_dropped_articles([article_of(kept)], kept + grid)

    assert len(result) == 1


def test_masthead_is_never_recovered():
    kept = existing_article()
    chrome = story_blocks(1, 2400, 3000, role="masthead")

    result = recover_dropped_articles([article_of(kept)], kept + chrome)

    assert len(result) == 1


def test_loose_paragraphs_without_a_headline_are_not_recovered():
    kept = existing_article()

    loose = [
        block(1 + i, "plain text", 2400, 3000 + i * 420,
              3300, 3400 + i * 420, prose(400), "advertisement")
        for i in range(4)
    ]

    result = recover_dropped_articles([article_of(kept)], kept + loose)

    assert len(result) == 1


# ------------------------------------------------------------------
# boundary_decomposer
# ------------------------------------------------------------------


def test_clean_article_is_left_alone():
    kept = existing_article()
    articles = [article_of(kept)]

    result = decompose_overlapping_articles(articles, kept)

    assert len(result) == 1
    assert result[0].block_ids == articles[0].block_ids


def test_rectangle_swallowing_a_neighbour_is_recut():
    """
    An L-shaped article -- body down the left, photo across the top
    right -- whose min/max rectangle covers a neighbouring story.
    """
    body = [
        block(1, "title", 100, 3000, 1000, 3200, "Left story"),
        block(2, "plain text", 100, 3220, 1000, 4000, prose(500)),
        block(3, "plain text", 100, 4020, 1000, 4800, prose(500)),
    ]
    photo = [block(4, "figure", 2000, 3000, 3800, 4000)]

    neighbour = [
        block(10, "title", 2000, 4200, 3800, 4400, "Right story"),
        block(11, "plain text", 2000, 4420, 3800, 4800, prose(500)),
    ]

    swallowing = article_of(body + photo, 1)
    other = article_of(neighbour, 2)

    all_blocks = body + photo + neighbour

    result = decompose_overlapping_articles(
        [swallowing, other], all_blocks
    )

    assert len(result) == 3

    # every block survives the re-cut
    seen = set()
    for a in result:
        seen |= set(a.block_ids)
    assert seen == {b.id for b in all_blocks}


def test_decomposition_never_loses_a_block():
    body = [
        block(1, "title", 100, 3000, 1000, 3200, "Left story"),
        block(2, "plain text", 100, 3220, 1000, 4800, prose(500)),
    ]
    photo = [block(4, "figure", 2000, 3000, 3800, 4000)]
    neighbour = [
        block(10, "title", 2000, 4200, 3800, 4400, "Right story"),
        block(11, "plain text", 2000, 4420, 3800, 4800, prose(500)),
    ]
    all_blocks = body + photo + neighbour

    result = decompose_overlapping_articles(
        [article_of(body + photo, 1), article_of(neighbour, 2)],
        all_blocks,
    )

    seen = set()
    for a in result:
        seen |= set(a.block_ids)

    assert seen == {b.id for b in all_blocks}


def test_pieces_of_a_split_never_cover_each_other():
    """
    The guarantee that stops the pass relocating the problem instead
    of fixing it: after a re-cut, no resulting rectangle may enclose
    another resulting article's blocks.
    """
    body = [
        block(1, "title", 100, 3000, 1000, 3200, "Left story"),
        block(2, "plain text", 100, 3220, 1000, 4000, prose(500)),
        block(3, "plain text", 100, 4020, 1000, 4800, prose(500)),
    ]
    photo = [block(4, "figure", 2000, 3000, 3800, 4000)]
    neighbour = [
        block(10, "title", 2000, 4200, 3800, 4400, "Right story"),
        block(11, "plain text", 2000, 4420, 3800, 4800, prose(500)),
    ]
    all_blocks = body + photo + neighbour

    result = decompose_overlapping_articles(
        [article_of(body + photo, 1), article_of(neighbour, 2)],
        all_blocks,
    )

    by_id = {b.id: b for b in all_blocks}

    rects = [
        (
            min(by_id[i].x1 for i in a.block_ids),
            min(by_id[i].y1 for i in a.block_ids),
            max(by_id[i].x2 for i in a.block_ids),
            max(by_id[i].y2 for i in a.block_ids),
        )
        for a in result
    ]

    for i, a in enumerate(result):
        for j, other in enumerate(result):
            if i == j:
                continue
            r = rects[i]
            for bid in other.block_ids:
                b = by_id[bid]
                inside = (
                    b.x1 >= r[0]
                    and b.y1 >= r[1]
                    and b.x2 <= r[2]
                    and b.y2 <= r[3]
                )
                assert not inside, (
                    f"article {a.article_id}'s rectangle encloses "
                    f"block {bid} owned by article {other.article_id}"
                )
