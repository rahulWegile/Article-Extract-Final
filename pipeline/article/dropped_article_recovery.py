"""
Recovers a whole news story that the grouping model threw away by
giving every one of its blocks a non-article role.

WHY THIS EXISTS
---------------

The grouping prompts define a set of roles that are, correctly, hard
exclusions -- "NEVER place advertisements inside news articles",
"These roles MUST NOT appear inside articles". ArticleGrouper honours
that by dropping any block whose role is in IGNORE_ROLES, and every
repair pass it runs afterwards (_reassign_orphan_blocks,
_reassign_orphan_title_roots, _recover_unclaimed_kicker_titles) only
ever looks at blocks that kept an article-eligible role.

The result is that a single wrong role classification is terminal. If
the model labels one story's blocks `advertisement`, that story does
not become a worse article -- it produces no article at all, no
boundary, no crop, and no warning, because the grouper's unclaimed-
block warning also filters IGNORE_ROLES out.

Confirmed on a real Tamil page (Dinamani, page 1): a personal-finance
column -- bordered box, author mugshot, "certified financial advisor"
credit line, small-print disclaimer at the foot -- had all eight of
its blocks (headline, byline, portrait, four body paragraphs and the
disclaimer) returned as `advertisement`, and the page's five
boundaries left the entire bottom-right quadrant empty. On page 2 of
the same document a temperature-forecast STORY (three headings plus
eight body paragraphs) was returned as `weather` and disappeared the
same way.

WHAT THIS DOES
--------------

Blocks that no article owns are clustered by proximity, and a cluster
is promoted to a new article ONLY when it is shaped like a printed
news story on its own: a headline carrying real recognised text, at
least MIN_BODY_BLOCKS paragraphs of substantial prose, a compact
footprint, and no encroachment on any existing article.

This pass is purely ADDITIVE. It reads existing articles but never
edits, re-cuts, reorders or drops one, so a page whose grouping is
already correct is returned unchanged. The only way it can be wrong
is by inventing an article, which is why every threshold below is set
where a real display/classified advertisement fails it.

Deliberately NOT recovered, whatever their geometry: masthead, page
header/footer, page number, logo, section header, comic. Those are
page chrome; a mistake there is not a lost story.

Runs before article_splitter.py so that anything it recovers is
re-checked by the same over-merge logic every other article gets, and
before boundary_decomposer.py so a recovered article participates in
rectangle de-confliction like any other.
"""

from pipeline.article.article_grouper import Article


# Classes that represent real printed content.
CONTENT_CLASSES = {
    "title",
    "plain text",
    "figure",
    "figure_caption",
}

# Page chrome, and reference panels. A block holding one of these
# roles is never eligible for recovery and is not even allowed to
# join a candidate cluster -- a masthead sitting above a dropped
# story must not drag the masthead into the recovered article.
#
# `utility_box` is here because it is the model's specific finding
# that a region is an almanac, horoscope, crossword, TV listing,
# stock table or schedule -- a grid of many small independent items,
# which is exactly what must not be turned into an article. Measured
# on two Gujarati pages: an almanac grid (42 blocks, 14 headings) and
# a listings grid (41 blocks, 6 headings) were the only false
# positives this pass produced across 116 pages, and both were
# >90% utility_box.
#
# Note that `weather` is deliberately NOT here. The model uses it
# both for a temperature widget and for a written forecast REPORT --
# a Tamil page had a three-heading, eight-paragraph forecast story
# dropped under it -- so weather is left to the story-shape tests
# below rather than excluded outright.
NEVER_RECOVER_ROLES = {
    "masthead",
    "page_header",
    "page_footer",
    "page_number",
    "section_header",
    "logo",
    "comic",
    "utility_box",
}

# ----------------------------------------------------------------
# Story-shape thresholds.
#
# These are the whole safety margin of this pass, so they are set
# where an advertisement fails rather than where a story passes.
# A display ad is mostly artwork with short slogan text; a classified
# or recruitment ad is dense but is set as many short listing lines,
# not as a few long prose paragraphs. Requiring BOTH several
# paragraphs AND a high average paragraph length is what separates
# the two.
# ----------------------------------------------------------------

# A headline must carry this much recognised text. An empty title
# block (a common OCR failure on large display type) proves nothing.
MIN_HEADLINE_CHARS = 12

# A body block only counts as a paragraph at this length.
MIN_BODY_BLOCK_CHARS = 120

# Paragraphs required, and total prose required across them.
MIN_BODY_BLOCKS = 3
MIN_BODY_CHARS = 600

# Mean characters per counted paragraph. Listing-style ad copy is
# made of many short lines and fails this even when it is long in
# aggregate.
MIN_MEAN_BODY_CHARS = 150

# One printed story carries a kicker, a headline and a subheadline --
# a handful of headings at most. A cluster with more headings than
# this is a GRID of separate items (an almanac, a results table, a
# classifieds column), and recovering it would fuse many small
# unrelated items into one invented article. Measured: real dropped
# stories carried 1-3 headings, the false positives 6 and 14.
MAX_HEADLINE_BLOCKS = 4

# Companion limit on sheer size, for a grid whose cells the layout
# detector did not read as headings. Real dropped stories measured
# 10-12 blocks; the grids, 41 and 42.
MAX_CLUSTER_BLOCKS = 24

# The cluster's own blocks must fill this much of the rectangle they
# span, so a few fragments scattered across a quadrant never become
# an "article".
MIN_FILL_RATIO = 0.35

# The recovered rectangle must be at least this fraction of the page.
MIN_PAGE_AREA_RATIO = 0.01

# A candidate cluster is abandoned if its rectangle covers this much
# of any existing article's block -- the space is not actually free
# and recovering it would double-claim printed content.
MAX_ENCROACH_RATIO = 0.30

# Share of the page this pass is allowed to add. Recovery assumes a
# mostly-correct page from which a story or two was wrongly dropped.
# When the candidates add up to more than this, that assumption is
# false: the page is a full-page advertisement, a public notice, or a
# prospectus/supplement page that the model classified as non-
# editorial in its entirety, and the honest reading is that the model
# was right. Measured on a Gujarati offer-document page, which
# without this cap shredded into 30 invented "articles" and took the
# page from 2 swallowed-content defects to 35.
#
# The whole page is abandoned rather than trimmed to the largest few
# candidates: the signal is about what KIND of page this is, so
# partial recovery would keep the same wrong premise.
MAX_RECOVERED_PAGE_AREA_RATIO = 0.25

# Proximity used to cluster free blocks, as a fraction of page width.
# Wide enough to bind a story's own columns across a gutter, far
# short of the jump to an unrelated item.
CLUSTER_GAP_PAGE_FRACTION = 0.025
CLUSTER_GAP_MIN = 60.0

# ----------------------------------------------------------------
# TIGHT HEADLINE+BODY PAIR RECOVERY
#
# The cluster-based recovery above is built for a multi-paragraph
# story (MIN_BODY_BLOCKS paragraphs) and, on a crowded page, its
# proximity clustering can merge a real one-paragraph dropped story
# together with OTHER unrelated dropped content sitting nearby into
# one blob that correctly fails _looks_like_a_story (several
# unrelated stories is not one story) -- so the real story is never
# recovered either. Confirmed on a real Urdu (THE INQUILAB,
# doc_000182 page 1) document: a one-paragraph Election Commission
# story (a headline classed "figure" -- a real headline the layout
# detector boxed as an image, the same confirmed Nastaliq styling
# failure layout_gap_recovery.py's recover_misclassified_text_blocks
# exists for, which this block fell just under the width floor of --
# directly above one 835-character body block) proximity-clustered
# with two OTHER separate dropped items and two loose graphics into
# a 7-block blob spanning most of the page's lower half.
#
# This pass looks for a much narrower, purely LOCAL pattern instead:
# one heading-like block (title, or figure -- see above) sitting
# DIRECTLY above one substantial free paragraph, sharing most of its
# column width, separated by only an ordinary headline-to-first-line
# gap. Recovered as its OWN two-block article, independent of
# whatever looser cluster either block would otherwise fall into --
# confirmed on the same real page that this fires on exactly the one
# genuine pair (57, 31) and nothing else, out of 6 candidate headings
# and 3 candidate paragraphs on that page.
# ----------------------------------------------------------------

# A single recovered paragraph must carry this much text on its own
# -- there is no second/third paragraph here to average against (see
# MIN_MEAN_BODY_CHARS above for the multi-paragraph case), so this is
# deliberately close to what a real short news item actually runs,
# not a stray caption or a classified-ad line.
TIGHT_PAIR_MIN_BODY_CHARS = 200

# The two blocks must share this much column width for the pairing
# to mean "this heading belongs to this paragraph" -- same convention
# as COLUMN_OVERLAP elsewhere in the pipeline (layout_gap_recovery.py,
# local_grouper.py), tightened since this pass has no multi-paragraph
# shape check to fall back on if the pairing itself is wrong.
TIGHT_PAIR_COLUMN_OVERLAP = 0.5

# Heading directly above body: the vertical gap between them must be
# tight, an ordinary headline-to-first-line spacing, not a jump
# across unrelated content in between.
TIGHT_PAIR_MAX_GAP = 40.0


def _column_overlap_ratio(a, b):
    width_a = max(1.0, a.x2 - a.x1)
    width_b = max(1.0, b.x2 - b.x1)

    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)

    return max(0.0, overlap) / min(width_a, width_b)


def _recover_tight_headline_body_pairs(free_blocks, owned_blocks):
    """
    See TIGHT HEADLINE+BODY PAIR RECOVERY above. `free_blocks` must
    already exclude anything the cluster-based pass recovered, so a
    block is never claimed twice.
    """

    headings = [
        block for block in free_blocks
        if _cls(block) in ("title", "figure")
    ]

    paragraphs = [
        block for block in free_blocks
        if _cls(block) == "plain text"
        and len(_text(block)) >= TIGHT_PAIR_MIN_BODY_CHARS
    ]

    consumed = set()

    recovered = []

    for heading in headings:

        if heading.id in consumed:
            continue

        best = None
        best_gap = None

        for paragraph in paragraphs:

            if paragraph.id in consumed:
                continue

            if paragraph.y1 < heading.y2:
                continue

            gap = paragraph.y1 - heading.y2

            if gap > TIGHT_PAIR_MAX_GAP:
                continue

            if (
                _column_overlap_ratio(heading, paragraph)
                < TIGHT_PAIR_COLUMN_OVERLAP
            ):
                continue

            if best_gap is None or gap < best_gap:
                best = paragraph
                best_gap = gap

        if best is None:
            continue

        group = [heading, best]

        if _encroaches(group, owned_blocks):
            continue

        consumed.add(heading.id)
        consumed.add(best.id)

        group.sort(key=lambda block: getattr(block, "reading_order", 0))

        recovered.append(group)

    return recovered


def _text(block):
    return (getattr(block, "text", "") or "").strip()


def _cls(block):
    return (getattr(block, "cls", "") or "").strip().lower()


def _role(block):
    return (getattr(block, "role", "") or "").strip().lower()


def _rect(blocks):
    return (
        min(b.x1 for b in blocks),
        min(b.y1 for b in blocks),
        max(b.x2 for b in blocks),
        max(b.y2 for b in blocks),
    )


def _block_rect(block):
    return (block.x1, block.y1, block.x2, block.y2)


def _area(rect):
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _intersection(a, b):
    return (
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    )


def _gap(block_a, block_b):
    dx = max(0.0, max(block_a.x1, block_b.x1) - min(block_a.x2, block_b.x2))
    dy = max(0.0, max(block_a.y1, block_b.y1) - min(block_a.y2, block_b.y2))

    return (dx * dx + dy * dy) ** 0.5


def _cluster(free_blocks, gap_limit):
    """Single-link clustering on edge-to-edge distance."""

    groups = [[block] for block in free_blocks]

    merged = True

    while merged:

        merged = False

        for i in range(len(groups)):

            for j in range(i + 1, len(groups)):

                if any(
                    _gap(a, b) <= gap_limit
                    for a in groups[i]
                    for b in groups[j]
                ):
                    groups[i] = groups[i] + groups[j]

                    del groups[j]

                    merged = True

                    break

            if merged:
                break

    return groups


def _page_size(blocks, page_image):

    if page_image is not None:

        try:
            height, width = page_image.shape[:2]

            if width > 0 and height > 0:
                return float(width), float(height)

        except Exception:
            pass

    return (
        max((b.x2 for b in blocks), default=1.0),
        max((b.y2 for b in blocks), default=1.0),
    )


def _looks_like_a_story(group, page_area):
    """
    True when this cluster is shaped like a printed news story that
    could stand on its own page position.
    """

    if len(group) > MAX_CLUSTER_BLOCKS:
        return False

    headlines = [
        block
        for block in group
        if _cls(block) == "title"
        and len(_text(block)) >= MIN_HEADLINE_CHARS
    ]

    if not headlines:
        return False

    if len(headlines) > MAX_HEADLINE_BLOCKS:
        return False

    paragraphs = [
        block
        for block in group
        if _cls(block) == "plain text"
        and len(_text(block)) >= MIN_BODY_BLOCK_CHARS
    ]

    if len(paragraphs) < MIN_BODY_BLOCKS:
        return False

    total_chars = sum(len(_text(block)) for block in paragraphs)

    if total_chars < MIN_BODY_CHARS:
        return False

    if total_chars / len(paragraphs) < MIN_MEAN_BODY_CHARS:
        return False

    rect = _rect(group)

    rect_area = _area(rect)

    if rect_area <= 0:
        return False

    if rect_area < MIN_PAGE_AREA_RATIO * page_area:
        return False

    filled = sum(_area(_block_rect(block)) for block in group)

    if filled / rect_area < MIN_FILL_RATIO:
        return False

    return True


def _encroaches(group, owned_blocks):
    """
    True when this cluster's rectangle covers printed content that
    already belongs to an article.
    """

    rect = _rect(group)

    for block in owned_blocks:

        block_area = _area(_block_rect(block))

        if block_area <= 0:
            continue

        covered = _area(_intersection(_block_rect(block), rect))

        if covered / block_area > MAX_ENCROACH_RATIO:
            return True

    return False


def recover_dropped_articles(articles, blocks, page_image=None):
    """
    Promote story-shaped clusters of unowned blocks to new articles.

    Existing articles are returned untouched and in their original
    order; recovered articles are appended in reading order. Article
    ids are renumbered over the combined list.
    """

    if not blocks:
        return articles

    if not articles:
        # The model found no article anywhere on this page. That is
        # its verdict that the page carries no editorial content at
        # all -- a full-page advertisement, a notice page, a picture
        # spread. Recovery exists to repair a page that is mostly
        # right, and has no basis for overturning a whole-page
        # judgement; doing so was measured inventing 14 "articles"
        # out of one Gujarati legal-notice page.
        return articles

    owned_ids = set()

    for article in articles:
        owned_ids.update(article.block_ids)

    owned_blocks = [block for block in blocks if block.id in owned_ids]

    free_blocks = [
        block
        for block in blocks
        if block.id not in owned_ids
        and _cls(block) in CONTENT_CLASSES
        and _role(block) not in NEVER_RECOVER_ROLES
    ]

    if not free_blocks:
        return articles

    page_width, page_height = _page_size(blocks, page_image)

    page_area = max(1.0, page_width * page_height)

    gap_limit = max(
        CLUSTER_GAP_MIN,
        CLUSTER_GAP_PAGE_FRACTION * page_width,
    )

    recovered = []

    rejected = 0

    for group in _cluster(free_blocks, gap_limit):

        if not _looks_like_a_story(group, page_area):
            continue

        if _encroaches(group, owned_blocks):
            rejected += 1
            continue

        group.sort(key=lambda block: getattr(block, "reading_order", 0))

        recovered.append(group)

    # See TIGHT HEADLINE+BODY PAIR RECOVERY above -- a narrower pass
    # for a one-paragraph story the cluster pass above either merged
    # into an unrelated blob (and so correctly rejected) or never
    # clustered at all. Runs on whatever the cluster pass above did
    # NOT already recover, so nothing is ever claimed twice.
    cluster_recovered_ids = {
        block.id for group in recovered for block in group
    }

    tight_pair_free_blocks = [
        block
        for block in free_blocks
        if block.id not in cluster_recovered_ids
    ]

    for group in _recover_tight_headline_body_pairs(
        tight_pair_free_blocks, owned_blocks,
    ):
        recovered.append(group)

    if not recovered:
        return articles

    # Page-level sanity check: see MAX_RECOVERED_PAGE_AREA_RATIO.
    recovered_area = sum(_area(_rect(group)) for group in recovered)

    if recovered_area > MAX_RECOVERED_PAGE_AREA_RATIO * page_area:

        print()
        print("=" * 60)
        print("DROPPED ARTICLE RECOVERY")
        print("=" * 60)
        print(
            f"Skipped: {len(recovered)} story-shaped cluster(s) would "
            f"add {100.0 * recovered_area / page_area:.0f}% of the "
            f"page, over the {100.0 * MAX_RECOVERED_PAGE_AREA_RATIO:.0f}"
            f"% cap -- treating this as a non-editorial page (full-page "
            f"advertisement / notice / supplement) and trusting the "
            f"model's classification."
        )
        print("=" * 60)
        print()

        return articles

    recovered.sort(key=lambda group: (_rect(group)[1], _rect(group)[0]))

    result = list(articles)

    for group in recovered:

        result.append(
            Article(
                article_id=0,
                blocks=group,
                block_ids=[block.id for block in group],
                confidence=1.0,
            )
        )

    for index, article in enumerate(result, start=1):
        article.article_id = index

    print()
    print("=" * 60)
    print("DROPPED ARTICLE RECOVERY")
    print("=" * 60)
    print(
        f"Stories recovered from non-article roles : {len(recovered)}"
    )

    for group in recovered:

        rect = _rect(group)

        roles = sorted({_role(block) or "none" for block in group})

        print(
            f"  +{len(group)} blocks "
            f"({rect[0]:.0f},{rect[1]:.0f})-({rect[2]:.0f},{rect[3]:.0f}) "
            f"dropped as: {', '.join(roles)}"
        )

    if rejected:
        print(
            f"Story-shaped clusters left alone (would overlap an "
            f"existing article) : {rejected}"
        )

    print(f"Total articles after recovery : {len(result)}")
    print("=" * 60)
    print()

    return result
