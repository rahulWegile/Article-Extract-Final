import difflib
import re

from pipeline.block_parser import LayoutBlock


class PageCleaner:
    """
    Clean and annotate layout blocks before exporting them.

    Responsibilities
    ----------------
    ✓ Remove noise blocks
    ✓ Detect newspaper masthead
    ✓ Detect true page headers
    ✓ Assign column ids
    ✓ Detect section headers
    ✓ Detect image captions
    """

    def clean(
        self,
        blocks,
        page_width,
        page_height,
        page_number=None,
        is_rtl=False,
    ):
        #
        # Merge duplicate detections
        #
        # Must run before remove_noise: a duplicate's losing box is
        # picked by comparing OCR confidence, which remove_noise's
        # class-based keep-without-text rule doesn't have a way to
        # express.
        #

        blocks = self.merge_duplicate_detections(
            blocks
        )

        #
        # Remove noise
        #

        blocks = self.remove_noise(
            blocks,
            page_height=page_height,
            page_width=page_width,
        )

        #
        # Detect masthead
        #
        # The masthead (the paper's big front-page title block) only
        # exists on page 1. Later pages commonly repeat the paper's
        # name in a small running header strip -- that's a distinct,
        # legitimate case already handled by detect_page_header below,
        # and must not also be tagged "masthead".
        #

        if page_number is None or page_number == 1:

            self.detect_masthead(
                blocks,
                page_height,
                page_width=page_width,
            )

        else:

            print(
                "Mastheads Detected : 0 (not page 1)"
            )

        #
        # Detect page header
        #

        self.detect_page_header(
            blocks,
            page_height,
            page_number=page_number,
        )

        #
        # Detect section headers
        #

        self.detect_section_headers(
            blocks,
        )

        #
        # Assign article columns
        #

        self.assign_columns(
            blocks,
            page_width,
            is_rtl=is_rtl,
        )

        #
        # Detect image captions
        #

        self.detect_captions(
            blocks
        )

        return blocks

    # -----------------------------------------------------
    # Merge Duplicate Detections
    # -----------------------------------------------------

    def merge_duplicate_detections(
        self,
        blocks,
    ):
        """
        Collapse two layout boxes that describe the SAME physical
        headline/paragraph at a different crop extent -- e.g. a
        headline boxed once in full and again as just its first
        word. The layout detector's own class-agnostic NMS
        (pipeline/layout_detector.py) already removes near-identical
        boxes across classes, but that's a plain IoU check, so it
        can't catch a smaller box that sits almost entirely inside a
        bigger one without the two being near-identical in size --
        confirmed on a real Telugu (Namasthe Telangana) page, where
        one headline was boxed three times ("వెంటాడిన మృత్యువు" in
        full, then again slightly misread, then again as just its
        first word "వెంటాడిన").

        Only merges when BOTH the geometry (one box almost entirely
        inside the other) AND the OCR'd text agree it's the same
        content. Geometry alone would also catch a real caption or
        title block that legitimately sits inside a much bigger
        figure's box (e.g. a masthead graphic with its name printed
        on it) -- that is not a duplicate and must not be dropped,
        so a match requires real textual corroboration, not just
        overlapping rectangles.
        """

        CONTAINMENT_THRESHOLD = 0.9

        SIMILARITY_THRESHOLD = 0.6

        def normalized(text):
            return re.sub(
                r"[^\w]",
                "",
                (text or "").strip().lower(),
            )

        def contained_area_ratio(a, b):

            ix1 = max(a.x1, b.x1)
            iy1 = max(a.y1, b.y1)
            ix2 = min(a.x2, b.x2)
            iy2 = min(a.y2, b.y2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)

            intersection = iw * ih

            area_a = max(
                1, (a.x2 - a.x1) * (a.y2 - a.y1)
            )

            area_b = max(
                1, (b.x2 - b.x1) * (b.y2 - b.y1)
            )

            return intersection / min(area_a, area_b)

        def same_text(text_a, text_b):

            norm_a = normalized(text_a)
            norm_b = normalized(text_b)

            if not norm_a or not norm_b:
                return False

            if norm_a in norm_b or norm_b in norm_a:
                return True

            return (
                difflib.SequenceMatcher(
                    None, norm_a, norm_b,
                ).ratio()
                >= SIMILARITY_THRESHOLD
            )

        dropped = set()

        carved = []

        next_block_id = (
            max((b.id for b in blocks), default=0) + 1
        )

        for i, block_a in enumerate(blocks):

            if id(block_a) in dropped:
                continue

            for block_b in blocks[i + 1:]:

                if id(block_b) in dropped:
                    continue

                if (
                    contained_area_ratio(block_a, block_b)
                    < CONTAINMENT_THRESHOLD
                ):
                    continue

                if not same_text(
                    getattr(block_a, "text", ""),
                    getattr(block_b, "text", ""),
                ):
                    continue

                loser = (
                    block_b
                    if block_a.ocr_confidence
                    >= block_b.ocr_confidence
                    else block_a
                )

                winner = block_a if loser is block_b else block_b

                dropped.add(id(loser))

                # The loser is usually just a noisier re-detection of
                # the SAME text the winner already covers -- but when
                # the loser's own box extends well above the winner's
                # top edge (Block 29 enclosing Block 23, with real
                # headline text in the gap above Block 23), that top
                # strip is not covered by the winner at all. Dropping
                # the loser outright would silently lose that
                # headline, so carve the uncovered strip out as its
                # own title block instead of discarding it with the
                # rest.
                if (
                    winner.y1 - loser.y1
                    >= self.MIN_ENCLOSURE_HEADLINE_HEIGHT
                ):

                    carved.append(
                        LayoutBlock(
                            id=next_block_id,
                            cls="title",
                            x1=loser.x1,
                            y1=loser.y1,
                            x2=loser.x2,
                            y2=winner.y1,
                            confidence=loser.confidence,
                        )
                    )

                    next_block_id += 1

                if loser is block_a:
                    break

        merged = [
            block
            for block in blocks
            if id(block) not in dropped
        ] + carved

        removed = len(blocks) - len(merged) + len(carved)

        if removed:
            print(
                f"Duplicate Detections Merged : {removed}"
            )

        if carved:
            print(
                f"Headline Regions Carved From Duplicates : "
                f"{len(carved)}"
            )

        return merged

    # -----------------------------------------------------
    # Remove Noise
    # -----------------------------------------------------

    # A single text-bearing block whose own height exceeds this
    # fraction of the page height is treated as a layout-detector
    # artifact, never real content -- see remove_noise below.
    MAX_TEXT_BLOCK_HEIGHT_RATIO = 0.6

    # Same idea, for width: a real newspaper column never runs this
    # wide, so a text-bearing block that does is the layout detector
    # failing to segment several columns as one blob -- confirmed on
    # a real Urdu (THE INQUILAB, doc_000172 page 3) document, where a
    # recovered "title" block at (0,850)-(1055,1192) on a 1846px-wide
    # page (57% of page width, only 11% of page height -- so the
    # existing HEIGHT check alone never caught it) sat directly on
    # top of four properly-detected column blocks and was handed to
    # the LLM as if it were real content of its own.
    MAX_TEXT_BLOCK_WIDTH_RATIO = 0.55

    # A block wider than the ratio above is still legitimate when it
    # is a single-line horizontal banner headline -- confirmed real
    # case (see layout_gap_recovery.py's SINGLE_LINE_MIN_ASPECT,
    # same threshold and same reasoning): a real Bengali banner
    # headline ran the full width of its story as one line, 1959x136,
    # aspect 14.4. The confirmed bad block above was aspect ~3.1
    # (1055x342) -- nowhere near ribbon-shaped -- so this exemption
    # does not readmit it.
    BANNER_MIN_ASPECT = 8.0

    # A text-bearing block whose box contains at least this fraction
    # of another (smaller) text-bearing block's own area counts as
    # "enclosing" it, for the multi-block check below.
    ENCLOSURE_CONTAINS_RATIO = 0.6

    # A block enclosing at least this many OTHER, mutually distinct
    # (non-overlapping) text-bearing blocks is a layout-detector blob
    # covering several real blocks at once, regardless of its own
    # width/height ratios -- the confirmed case above enclosed four.
    MIN_ENCLOSED_BLOCKS = 2

    # Before discarding an enclosing blob outright, check whether it
    # extends at least this far ABOVE the topmost block it encloses --
    # a real headline sitting over a body block the detector also
    # boxed separately (Block 29 enclosing Block 23, with its own
    # headline text in the gap above Block 23's top edge). Below this
    # height the leftover strip is just rounding/padding around the
    # enclosed blocks, not a distinct headline worth keeping.
    MIN_ENCLOSURE_HEADLINE_HEIGHT = 20

    def remove_noise(
        self,
        blocks,
        page_height=None,
        page_width=None,
    ):

        def contains_ratio(container, contained):

            ix1 = max(container.x1, contained.x1)
            iy1 = max(container.y1, contained.y1)
            ix2 = min(container.x2, contained.x2)
            iy2 = min(container.y2, contained.y2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)

            intersection = iw * ih

            contained_area = max(
                1,
                (contained.x2 - contained.x1)
                * (contained.y2 - contained.y1),
            )

            return intersection / contained_area

        def mutually_overlapping(a, b):

            return (
                max(contains_ratio(a, b), contains_ratio(b, a))
                >= self.ENCLOSURE_CONTAINS_RATIO
            )

        cleaned = []

        next_block_id = (
            max((b.id for b in blocks), default=0) + 1
        )

        headline_blocks_carved = 0

        visual_classes = {
            "object",
            "image",
            "photo",
            "figure",
        }

        #
        # Classes kept for their GEOMETRY even when OCR read nothing
        # off them.
        #
        # A "title" is where an article begins. Every downstream stage
        # -- the grouping prompt, local_grouper's root detection, and
        # the final article crop's top edge -- needs that box to exist
        # even when its text is unreadable, and the article-extraction
        # model re-reads the crop image anyway, so an empty string
        # here costs nothing.
        #
        # Dropping these was silently turning an OCR failure into a
        # LAYOUT failure: confirmed on real Gujarati (દિવ્ય ભાસ્કર)
        # pages, where headline boxes the layout detector found at
        # IoU 1.00 were deleted here after Tesseract returned a single
        # character for them, and the articles beneath them were then
        # cropped starting below their own headline. Tesseract's side
        # of that is fixed in tesseract_engine._ocr_block; this is the
        # guard that stops the same shape of OCR failure from
        # removing a real article boundary again.
        #
        structural_classes = {
            "title",
        }

        keep_without_text = (
            visual_classes | structural_classes
        )

        # Classes a layout detector's own noise can plausibly mis-tag
        # as spanning most of the page -- see the oversized-block
        # check below. Deliberately excludes visual_classes: a real
        # full-page photo/graphic is a normal, legitimate thing to
        # detect, so height alone must never disqualify a "figure".
        text_bearing_classes = {
            "title",
            "plain text",
            "figure_caption",
        }

        # Classes the WIDTH check (below) applies to -- deliberately
        # excludes "title". A real newspaper headline routinely spans
        # 40%-80% of page width across 2 or 3 lines of large type
        # (e.g. a 3-column headline at ~0.46 width ratio, ~6.75
        # aspect), which is exactly the shape MAX_TEXT_BLOCK_WIDTH_
        # RATIO exists to catch for a mis-segmented multi-column BODY
        # blob instead. Confirmed on a real Odia (Sambad, doc_000177
        # page 3) page: a legitimate 2-line, 3-column headline
        # ("ସଂସଦର ଆଇନକୁ ଅପେକ୍ଷା ନକରି ପଦକ୍ଷେପ ନେଇଛୁ: ସିଜେଆଇ") at width
        # ratio ~0.46 and aspect ~6.75 (under BANNER_MIN_ASPECT) was
        # deleted by this check, leaving the article's body with no
        # headline. The HEIGHT check and the enclosure check above/
        # below still apply to "title" -- only the WIDTH check is
        # body-only.
        WIDTH_FILTER_CLASSES = {
            "plain text",
            "figure_caption",
        }

        oversized_removed = 0

        for block in blocks:

            cls = getattr(
                block,
                "cls",
                "",
            ).lower()

            text = getattr(
                block,
                "text",
                "",
            ).strip()

            #
            # Remove abandoned detections
            #

            if cls == "abandon":
                continue

            #
            # Remove tiny OCR garbage.
            #
            # IMPORTANT:
            # Do not remove visual blocks, and do not remove the
            # structural blocks whose box carries meaning on its own
            # (see keep_without_text above).
            #

            if (
                len(text) <= 2
                and cls not in keep_without_text
            ):
                continue

            #
            # Remove CMYK / printer marks
            #

            if text.upper() in {
                "CMYK",
                "CMY",
                "MYK",
                "C MY K",
                "C MV K",
            }:
                continue

            #
            # Remove implausibly oversized text-bearing detections.
            #
            # A real newspaper headline, paragraph or caption never
            # spans most of the page's height -- print layout always
            # breaks that much vertical space into several stacked/
            # columned blocks. A single "title"/"plain text"/
            # "figure_caption" box that tall is the layout detector
            # itself failing to segment a dense or unusually-shaped
            # region (confirmed across Urdu, Kannada and English real
            # pages -- e.g. a real Urdu (THE INQUILAB) page where a
            # "title" block was returned at (423,340)-(1846,3004) on a
            # 1846x3004 page, 89% of the page's own height, with
            # garbled OCR text, sitting on top of that same region's
            # real headline+body+image blocks which were ALSO
            # detected at their own correct, much smaller sizes).
            #
            # Left in place, this kind of block still gets grouped
            # into some article (its class survives remove_noise
            # either way), and because BoundaryBuilder/
            # article_region_reconstructor.py trust each article's
            # own blocks completely and union ALL of them, that one
            # phantom block alone stretches the article's final
            # boundary to nearly the entire page -- exactly the
            # "random"-looking, story-swallowing boundaries this
            # guards against. It is always redundant with the
            # properly-sized blocks covering the same real content,
            # never the only copy of anything, so dropping it costs
            # no real text.
            #
            # Purely geometric and class-based -- no language check,
            # so it applies identically to every script.
            #

            if (
                cls in text_bearing_classes
                and page_height
                and (block.y2 - block.y1)
                > self.MAX_TEXT_BLOCK_HEIGHT_RATIO * page_height
            ):
                oversized_removed += 1
                continue

            #
            # Remove implausibly WIDE text-bearing detections.
            #
            # Same failure mode as the height check above, along the
            # other axis -- see MAX_TEXT_BLOCK_WIDTH_RATIO. Exempts a
            # single-line horizontal banner headline (BANNER_MIN_ASPECT),
            # which is legitimately close to full page width.
            #

            block_width = block.x2 - block.x1
            block_height = max(1, block.y2 - block.y1)
            block_aspect = block_width / block_height

            if (
                cls in WIDTH_FILTER_CLASSES
                and page_width
                and block_width
                > self.MAX_TEXT_BLOCK_WIDTH_RATIO * page_width
                and block_aspect < self.BANNER_MIN_ASPECT
            ):
                oversized_removed += 1
                continue

            #
            # Remove a text-bearing block that ENCLOSES several other,
            # mutually distinct text-bearing blocks -- regardless of
            # its own width/height ratios. See ENCLOSURE_CONTAINS_RATIO/
            # MIN_ENCLOSED_BLOCKS: this is the same "detector drew one
            # blob over several real blocks" failure the width/height
            # checks above target, caught here by content instead of
            # by shape, for a blob whose own bbox doesn't happen to
            # trip either ratio.
            #

            if cls in text_bearing_classes:

                enclosed = []

                for other in blocks:

                    if other is block:
                        continue

                    other_cls = getattr(
                        other, "cls", "",
                    ).lower()

                    if other_cls not in text_bearing_classes:
                        continue

                    if (
                        contains_ratio(block, other)
                        < self.ENCLOSURE_CONTAINS_RATIO
                    ):
                        continue

                    if any(
                        mutually_overlapping(other, kept)
                        for kept in enclosed
                    ):
                        continue

                    enclosed.append(other)

                if enclosed:

                    # Don't discard a real headline along with the
                    # blob -- if this block extends well above the
                    # topmost block it encloses, that leftover strip
                    # is a plausible title region none of the enclosed
                    # blocks cover. Carve it out as its own block
                    # instead of dropping it with the rest.
                    enclosed_top = min(b.y1 for b in enclosed)

                    has_headline_gap = (
                        enclosed_top - block.y1
                        >= self.MIN_ENCLOSURE_HEADLINE_HEIGHT
                    )

                    # Normally only a blob enclosing >= 2 distinct
                    # blocks is a confirmed detector artifact worth
                    # discarding -- but a single enclosed block with a
                    # real headline gap above it (Block 29 enclosing
                    # just Block 23, with headline text in the gap) is
                    # just as much evidence the enclosing box isn't
                    # its own real content, so also fires with only
                    # one enclosed block in that specific case.
                    if (
                        len(enclosed) >= self.MIN_ENCLOSED_BLOCKS
                        or has_headline_gap
                    ):

                        if has_headline_gap:

                            cleaned.append(
                                LayoutBlock(
                                    id=next_block_id,
                                    cls="title",
                                    x1=block.x1,
                                    y1=block.y1,
                                    x2=block.x2,
                                    y2=enclosed_top,
                                    confidence=block.confidence,
                                )
                            )

                            next_block_id += 1
                            headline_blocks_carved += 1

                        oversized_removed += 1
                        continue

            cleaned.append(
                block
            )

        print()
        print("=" * 60)
        print("PAGE CLEANER")
        print("=" * 60)

        print(
            f"Original Blocks : {len(blocks)}"
        )

        print(
            f"Clean Blocks    : {len(cleaned)}"
        )

        if oversized_removed:
            print(
                f"Oversized Detector Artifacts Removed : "
                f"{oversized_removed}"
            )

        if headline_blocks_carved:
            print(
                f"Headline Regions Carved From Enclosures : "
                f"{headline_blocks_carved}"
            )

        print("=" * 60)

        return cleaned

    # -----------------------------------------------------
    # Masthead Detection
    # -----------------------------------------------------

    def detect_masthead(
        self,
        blocks,
        page_height,
        page_width=None,
    ):

        mastheads = [
            "THE ASIAN AGE",
            "THE HINDU",
            "TIMES OF INDIA",
            "INDIAN EXPRESS",
            "HINDUSTAN TIMES",
            "THE TELEGRAPH",
            "DECCAN CHRONICLE",
            "SIASAT",
            "INQUILAB",
        ]

        # Devanagari has no case, so these are matched against the
        # raw (non-uppercased) text below instead of `.upper()`.
        mastheads_devanagari = [
            "दैनिक भास्कर",
            "दैनिक जागरण",
            "अमर उजाला",
            "नवभारत टाइम्स",
            "हिंदुस्तान",
            "हिन्दुस्तान",
            "राजस्थान पत्रिका",
            "पंजाब केसरी",
            "जनसत्ता",
            "नई दुनिया",
            "पत्रिका",
        ]

        # Urdu has no case either, so these are matched against the
        # raw text, same as the Devanagari names above.
        mastheads_urdu = [
            "سیاست",
            "انقلاب",
        ]

        top_limit = page_height * 0.12

        detected = 0

        for block in blocks:

            if not hasattr(
                block,
                "type",
            ):
                block.type = block.cls

            if not hasattr(
                block,
                "is_global",
            ):
                block.is_global = False

            if not hasattr(
                block,
                "column",
            ):
                block.column = -1

            raw_text = getattr(
                block,
                "text",
                "",
            )

            text = raw_text.upper()

            #
            # Masthead must be near top.
            #

            matched_name = False

            if block.y2 <= top_limit:

                if any(
                    name in text
                    for name in mastheads
                ) or any(
                    name in raw_text
                    for name in mastheads_devanagari
                ) or any(
                    name in raw_text
                    for name in mastheads_urdu
                ):

                    matched_name = True

            #
            # Pure geometric fallback.
            #
            # A block sitting right at the top of the page and
            # spanning almost the full page width is a masthead
            # banner even when OCR found no text -- e.g. artistic
            # logo banners like The Siasat Daily, which the layout
            # detector boxes as a "figure" with an empty OCR string.
            #

            geometric_match = False

            if (
                not matched_name
                and page_width
                and block.y1 < page_height * 0.15
            ):

                block_width = (
                    block.x2 - block.x1
                )

                if block_width >= 0.75 * page_width:

                    geometric_match = True

            if matched_name or geometric_match:

                block.type = "masthead"

                block.is_global = True

                block.column = -1

                detected += 1

        print(
            f"Mastheads Detected : {detected}"
        )

    # -----------------------------------------------------
    # Page Header Detection
    # -----------------------------------------------------

    def detect_page_header(
        self,
        blocks,
        page_height,
        page_number=None,
    ):
        """
        Detect TRUE newspaper page metadata/header blocks.

        IMPORTANT:

        Do NOT classify a large article headline/body block as
        a page header merely because it contains a date/month/day.

        This prevents oversized blocks such as:

            "Unfortunate, no barricade can erase '31 martyrs'
             legacy: CM Lockdown in Srinagar imposed to block
             Martyrs' marches Day"

        from becoming global page headers.
        """

        keywords = [
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
            "SATURDAY",
            "SUNDAY",

            "WWW",
            "PRICE",
            "VOL",
            "VOLUME",
            "ISSUE",

            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]

        # Devanagari equivalents. Checked against the raw text
        # (case-insensitivity doesn't apply to this script).
        keywords_devanagari = [
            "सोमवार",
            "मंगलवार",
            "बुधवार",
            "गुरुवार",
            "शुक्रवार",
            "शनिवार",
            "रविवार",

            "मूल्य",
            "कीमत",
            "पृष्ठ",
            "अंक",
            "वर्ष",
            "संस्करण",

            "जनवरी",
            "फरवरी",
            "मार्च",
            "अप्रैल",
            "मई",
            "जून",
            "जुलाई",
            "अगस्त",
            "सितंबर",
            "सितम्बर",
            "अक्टूबर",
            "नवंबर",
            "नवम्बर",
            "दिसंबर",
            "दिसम्बर",
        ]

        # Urdu equivalents. Urdu script has no case either, so these
        # are matched against the raw text, same as Devanagari above.
        keywords_urdu = [
            "چہارشنبہ",
            "پنجشنبہ",
            "جمعہ",
            "ہفتہ",
            "اتوار",
            "پیر",
            "منگل",
            "بدھ",
            "جمعرات",

            "قیمت",
            "جلد",
            "شمارہ",
            "صفحہ",

            "جنوری",
            "فروری",
            "مارچ",
            "اپریل",
            "مئی",
            "جون",
            "جولائی",
            "اگست",
            "ستمبر",
            "اکتوبر",
            "نومبر",
            "دسمبر",
        ]

        top_limit = page_height * 0.15

        detected = 0

        #
        # Estimate page width from blocks.
        #

        if blocks:

            estimated_page_width = max(
                getattr(
                    b,
                    "x2",
                    0,
                )
                for b in blocks
            )

        else:

            estimated_page_width = 1

        if estimated_page_width <= 0:
            estimated_page_width = 1

        #
        # Metadata patterns.
        #

        metadata_patterns = [
            r"\bVOL\.?\s*\d+",
            r"\bVOLUME\s*\d+",
            r"\bISSUE\s*\d+",
            r"\bPRICE\s*[:₹]?\s*\d+",
            r"\bWWW\.",
            r"\bPAGE\s+\d+\b",
            r"\bEDITION\b",
        ]

        # Devanagari equivalents (checked against the raw text).
        metadata_patterns_devanagari = [
            r"मूल्य\s*[:₹]?\s*\d+",
            r"कीमत\s*[:₹]?\s*\d+",
            r"पृष्ठ\s+\d+",
            r"अंक\s*\d+",
            r"वर्ष\s*\d+",
        ]

        # Urdu equivalents (checked against the raw text).
        metadata_patterns_urdu = [
            r"قیمت\s*[:₹]?\s*\d+",
            r"جلد\s*\d+",
            r"شمارہ\s*\d+",
            r"صفحہ\s+\d+",
        ]

        #
        # Standalone date/day patterns.
        #

        standalone_day_or_date = re.compile(
            r"""
            ^
            \s*
            (
                MONDAY|
                TUESDAY|
                WEDNESDAY|
                THURSDAY|
                FRIDAY|
                SATURDAY|
                SUNDAY|
                JAN(?:UARY)?|
                FEB(?:RUARY)?|
                MAR(?:CH)?|
                APR(?:IL)?|
                MAY|
                JUN(?:E)?|
                JUL(?:Y)?|
                AUG(?:UST)?|
                SEP(?:TEMBER)?|
                OCT(?:OBER)?|
                NOV(?:EMBER)?|
                DEC(?:EMBER)?
            )
            (
                \s+\d{1,2}
            )?
            (
                \s*,?\s+\d{4}
            )?
            \s*
            $
            """,
            re.VERBOSE,
        )

        # Devanagari equivalent (day and/or month name, optionally
        # followed by a day number and/or year). Devanagari has no
        # case, so this is matched against the raw text, not upper().
        standalone_day_or_date_devanagari = re.compile(
            r"""
            ^
            \s*
            (
                सोमवार|मंगलवार|बुधवार|गुरुवार|
                शुक्रवार|शनिवार|रविवार|
                जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|
                जुलाई|अगस्त|सितंबर|सितम्बर|
                अक्टूबर|नवंबर|नवम्बर|दिसंबर|दिसम्बर
            )
            (
                \s+\d{1,2}
            )?
            (
                \s*,?\s+\d{4}
            )?
            \s*
            $
            """,
            re.VERBOSE,
        )

        # Urdu equivalent (day and/or month name, optionally followed
        # by a day number and/or year). Urdu script has no case, so
        # this is matched against the raw text, not upper().
        standalone_day_or_date_urdu = re.compile(
            r"""
            ^
            \s*
            (
                چہارشنبہ|پنجشنبہ|جمعہ|ہفتہ|اتوار|پیر|منگل|بدھ|جمعرات|
                جنوری|فروری|مارچ|اپریل|مئی|جون|
                جولائی|اگست|ستمبر|اکتوبر|نومبر|دسمبر
            )
            (
                \s+\d{1,2}
            )?
            (
                \s*,?\s+\d{4}
            )?
            \s*
            $
            """,
            re.VERBOSE,
        )

        #
        # Article-like indicators.
        #
        # If the block looks like an actual news headline,
        # do NOT make it global.
        #

        article_indicators = [
            ":",
            ";",
            "?",
            "!",
            "'",
            '"',
            " CM ",
            " PM ",
            " CONGRESS ",
            " POLICE ",
            " COURT ",
            " GOVERNMENT ",
            " MINISTER ",
            " LEADER ",
            " SAYS ",
            " SAID ",
            " LOCKDOWN ",
            " CUSTODY ",
            " PUBLISHERS ",
            " SRINAGAR ",
        ]

        for block in blocks:

            #
            # Never overwrite an already-global block.
            #

            if getattr(
                block,
                "is_global",
                False,
            ):
                continue

            #
            # Pure geometric fallback.
            #
            # A thin horizontal strip sitting right at the top of
            # page 1 and spanning at least half the page width is
            # page metadata chrome (e.g. a date/edition bar) even
            # without OCR text or a keyword match.
            #

            if (
                page_number is None
                or page_number == 1
            ) and block.y1 < page_height * 0.16:

                strip_height = max(
                    1,
                    block.y2 - block.y1,
                )

                strip_width = max(
                    1,
                    block.x2 - block.x1,
                )

                if (
                    strip_width
                    >= 0.50 * estimated_page_width
                    and strip_height
                    <= page_height * 0.05
                ):

                    block.type = "page_header"

                    block.is_global = True

                    block.column = -1

                    detected += 1

                    continue

            text = getattr(
                block,
                "text",
                "",
            ).strip()

            if not text:
                continue

            text_upper = text.upper()

            #
            # Must be near top of page.
            #

            if block.y2 > top_limit:
                continue

            #
            # Geometry.
            #

            block_height = max(
                1,
                block.y2 - block.y1,
            )

            block_width = max(
                1,
                block.x2 - block.x1,
            )

            width_ratio = (
                block_width
                / estimated_page_width
            )

            height_ratio = (
                block_height
                / page_height
            )

            #
            # -------------------------------------------------
            # CRITICAL FIX
            # -------------------------------------------------
            #
            # Large blocks are NOT page headers.
            #
            # This prevents block 3 from becoming global.
            #

            if (
                block_height
                > page_height * 0.05
            ):
                continue

            #
            # Very wide blocks are usually article/header
            # regions rather than compact page metadata.
            #

            if width_ratio > 0.75:
                continue

            #
            # Very tall relative blocks are not metadata.
            #

            if height_ratio > 0.05:
                continue

            #
            # Long text is almost certainly article content.
            #

            if len(text) > 100:
                continue

            #
            # -------------------------------------------------
            # Article-like text detection
            # -------------------------------------------------
            #

            article_like = False

            for indicator in article_indicators:

                if indicator in text_upper:

                    article_like = True

                    break

            if article_like:
                continue

            #
            # -------------------------------------------------
            # Keyword detection
            # -------------------------------------------------
            #

            matched_keyword = any(
                word in text_upper
                for word in keywords
            ) or any(
                word in text
                for word in keywords_devanagari
            ) or any(
                word in text
                for word in keywords_urdu
            )

            if not matched_keyword:
                continue

            #
            # -------------------------------------------------
            # Metadata pattern detection
            # -------------------------------------------------
            #

            metadata_match = any(
                re.search(
                    pattern,
                    text_upper,
                )
                for pattern in metadata_patterns
            ) or any(
                re.search(
                    pattern,
                    text,
                )
                for pattern in metadata_patterns_devanagari
            ) or any(
                re.search(
                    pattern,
                    text,
                )
                for pattern in metadata_patterns_urdu
            )

            #
            # Standalone day/date.
            #

            standalone_match = bool(
                standalone_day_or_date.fullmatch(
                    text_upper
                )
            ) or bool(
                standalone_day_or_date_devanagari.fullmatch(
                    text
                )
            ) or bool(
                standalone_day_or_date_urdu.fullmatch(
                    text
                )
            )

            #
            # A keyword alone is NOT sufficient.
            #
            # It must look like actual newspaper metadata.
            #

            if not (
                metadata_match
                or standalone_match
            ):
                continue

            #
            # -------------------------------------------------
            # Mark as page header
            # -------------------------------------------------
            #

            block.type = "page_header"

            block.is_global = True

            block.column = -1

            detected += 1

        print(
            f"Page Headers : {detected}"
        )

    # -----------------------------------------------------
    # Assign Columns
    # -----------------------------------------------------

    def assign_columns(
        self,
        blocks,
        page_width,
        is_rtl=False,
    ):

        article_blocks = [
            b
            for b in blocks
            if not getattr(
                b,
                "is_global",
                False,
            )
        ]

        if not article_blocks:
            return

        #
        # Newspaper page is divided into
        # six logical column regions.
        #

        column_width = (
            page_width / 6
        )

        for block in article_blocks:

            center = (
                block.x1
                + block.x2
            ) / 2

            raw_column = min(
                int(
                    center
                    / column_width
                ),
                5,
            )

            # For a right-to-left script (Urdu), column 0 must stay
            # "the first column a reader's eye reaches" -- the
            # RIGHTMOST print column, not the geometrically leftmost
            # one -- since the grouping prompt uses the column number
            # itself as a reading-flow signal (see LanguagePipeline
            # .is_rtl and sort_blocks.py for the matching reading-
            # order fix).
            block.column = (
                5 - raw_column
                if is_rtl
                else raw_column
            )

    # -----------------------------------------------------
    # Section Headers
    # -----------------------------------------------------

    def detect_section_headers(
        self,
        blocks,
    ):

        pattern = re.compile(
            r"^[A-Z ]+\s+\d+$"
        )

        # Devanagari has no uppercase, so the Latin-only pattern
        # above never matches a Hindi section header (e.g. "खेल 4").
        # Matched against the raw text instead of `.upper()`.
        pattern_devanagari = re.compile(
            r"^[ऀ-ॿ ]+\s*\d+$"
        )

        detected = 0

        for block in blocks:

            if getattr(
                block,
                "is_global",
                False,
            ):
                continue

            text = getattr(
                block,
                "text",
                "",
            ).strip()

            if not text:
                continue

            if pattern.match(
                text.upper()
            ) or pattern_devanagari.match(
                text
            ):

                block.type = (
                    "section_header"
                )

                detected += 1

        print(
            f"Section Headers : {detected}"
        )

    # -----------------------------------------------------
    # Caption Detection
    # -----------------------------------------------------

    def detect_captions(
        self,
        blocks,
    ):

        detected = 0

        for block in blocks:

            if getattr(
                block,
                "is_global",
                False,
            ):
                continue

            text = getattr(
                block,
                "text",
                "",
            ).strip()

            if not text:
                continue

            text_upper = text.upper()

            #
            # Caption indicators.
            #

            caption_patterns = [
                r"^PHOTO\s*:",
                r"^FILE\s*PHOTO",
                r"^FILE\s*:",
                r"^PHOTO\s+BY",
                r"^COURTESY\s+",
                r"^PIC\s*:",
                r"^IMAGE\s*:",
            ]

            # Devanagari equivalents, checked against the raw text.
            caption_patterns_devanagari = [
                r"^फोटो\s*:",
                r"^फ़ोटो\s*:",
                r"^फाइल\s*फोटो",
                r"^तस्वीर\s*:",
                r"^चित्र\s*:",
                r"^सौजन्य\s*:",
                r"^तस्वीर\s*सौजन्य",
            ]

            is_caption = any(
                re.search(
                    pattern,
                    text_upper,
                )
                for pattern in caption_patterns
            ) or any(
                re.search(
                    pattern,
                    text,
                )
                for pattern in caption_patterns_devanagari
            )

            if is_caption:

                block.type = "caption"

                detected += 1

        print(
            f"Captions Detected : {detected}"
        )