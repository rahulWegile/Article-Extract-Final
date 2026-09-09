"""
Re-links a headline block a DIFFERENT article claims back to a
HEADLESS neighbour it actually sits directly above.

WHY THIS EXISTS
---------------

_reassign_orphan_blocks (article_grouper.py) already fixes a block
whose own article claim doesn't hold up geometrically, but only when
the block is nearly ISOLATED from every one of its own article's
other blocks (ORPHAN_ISOLATION_FLOOR requires essentially no column
overlap with its own siblings). That is deliberately conservative --
loosening it risks the exact "block moved into an unrelated article
across the page" failures its own history describes.

Confirmed on a real Urdu (THE INQUILAB, doc_000182 page 2) document,
this leaves a real case uncaught: a theft story's headline
("...زیورات اور نقدی چوری...", classed "figure" -- a decorative
Nastaliq styling the layout detector boxed as an image, the same
confirmed failure layout_gap_recovery.py's recover_misclassified_text_
blocks exists for) was claimed by an unrelated ADG-visit article two
rows above it, sharing enough column overlap with that article's own
blocks (~47%) to never be flagged as isolated -- while sitting with
essentially ZERO gap directly above the theft story's own topmost
block, which had no headline of its own as a result.

WHAT THIS DOES
--------------

For every article with NO title/figure-classed block of its own (a
headless article -- a real, complete story missing only its
headline), look for a title/figure block owned by ANOTHER article
that sits with a tight vertical gap directly above this article's own
topmost block, sharing most of its column width. Move that ONE block
across if the donor article keeps a headline of its own afterward.

SAFETY
------

- Only ever moves a block INTO a headless article -- an article that
  already has its own headline is never a target, so this cannot
  create the "one story split by proximity" failure a general
  nearest-neighbour move would risk.
- Never leaves the donor headless in turn -- a donor with only one
  headline-class block is never a source, so this cannot just swap
  which article is missing one.
- The gap and column-overlap thresholds are tight (an ordinary
  headline-to-first-line spacing, most of the column width), so this
  only fires on a genuine "this heading belongs directly below it"
  layout, never on two unrelated blocks that merely share a column
  somewhere on the page.
"""

from pipeline.article.article_grouper import Article


HEADING_CLASSES = ("title", "figure")

# The two blocks must share this much column width -- same convention
# as elsewhere in this pipeline (layout_gap_recovery.py's
# COLUMN_OVERLAP, dropped_article_recovery.py's tight-pair recovery).
COLUMN_OVERLAP = 0.5

# Heading directly above the headless article's own topmost block: an
# ordinary headline-to-first-line spacing, not a jump across other
# content. Same value as dropped_article_recovery.py's tight-pair
# recovery, for the same reason -- both target one confirmed shape,
# "heading immediately above body".
MAX_GAP = 40.0


def _cls(block):
    return (getattr(block, "cls", "") or "").strip().lower()


def _is_heading(block):
    return _cls(block) in HEADING_CLASSES


def _is_title(block):
    # Stricter than _is_heading, used only to decide whether an
    # article already HAS a headline of its own. A "figure" is
    # ambiguous -- it is exactly as likely to be the story's own real
    # photograph as a misclassified headline (the case this whole
    # module exists for) -- so an article's own figure block must
    # never count as "already has a headline" the way a genuine
    # title-classed block does. Confirmed on the real Urdu page this
    # was built against: the theft story already owned a real photo
    # (a "figure" block) and, without this distinction, was never
    # recognised as headless at all.
    return _cls(block) == "title"


def _column_overlap_ratio(a, b):
    width_a = max(1.0, float(a.x2 - a.x1))
    width_b = max(1.0, float(b.x2 - b.x1))

    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)

    return max(0.0, overlap) / min(width_a, width_b)


def relink_headless_headlines(articles):
    """
    Returns a new list of Article objects with any confirmed
    heading moved from its donor to the headless article it actually
    belongs to. Articles this pass does not touch are returned by
    reference, unchanged -- same convention as
    dropped_article_recovery.recover_dropped_articles.
    """

    if len(articles) < 2:
        return articles

    blocks_by_article = {
        article.article_id: list(article.blocks) for article in articles
    }

    relinked = []

    for headless in articles:

        own_blocks = blocks_by_article[headless.article_id]

        if not own_blocks or any(_is_title(b) for b in own_blocks):
            continue

        topmost = min(own_blocks, key=lambda b: b.y1)

        best_owner_id = None
        best_heading = None
        best_gap = None

        for owner in articles:

            if owner.article_id == headless.article_id:
                continue

            for candidate in blocks_by_article[owner.article_id]:

                if not _is_heading(candidate):
                    continue

                if candidate.y2 > topmost.y1:
                    continue

                gap = topmost.y1 - candidate.y2

                if gap > MAX_GAP:
                    continue

                if _column_overlap_ratio(candidate, topmost) < COLUMN_OVERLAP:
                    continue

                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_heading = candidate
                    best_owner_id = owner.article_id

        if best_heading is None:
            continue

        donor_remaining = [
            b for b in blocks_by_article[best_owner_id]
            if b.id != best_heading.id
        ]

        if not any(_is_title(b) for b in donor_remaining):
            # Moving this block would leave the DONOR headless in
            # turn -- never trade one headless article for another.
            continue

        blocks_by_article[best_owner_id] = donor_remaining
        blocks_by_article[headless.article_id] = (
            [best_heading] + own_blocks
        )

        relinked.append(
            (best_heading.id, best_owner_id, headless.article_id)
        )

    if not relinked:
        return articles

    result = [
        Article(
            article_id=article.article_id,
            blocks=blocks_by_article[article.article_id],
            block_ids=[
                block.id for block in blocks_by_article[article.article_id]
            ],
            confidence=article.confidence,
        )
        for article in articles
    ]

    print()
    print("=" * 60)
    print("HEADLESS HEADLINE RELINK")
    print("=" * 60)

    for block_id, donor_id, target_id in relinked:
        print(
            f"  block {block_id}: article {donor_id} -> "
            f"headless article {target_id}"
        )

    print(f"Headlines re-linked : {len(relinked)}")
    print("=" * 60)
    print()

    return result
