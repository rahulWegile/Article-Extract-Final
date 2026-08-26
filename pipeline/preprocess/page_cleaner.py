import re


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
    ):
        #
        # Remove noise
        #

        blocks = self.remove_noise(
            blocks
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
        )

        #
        # Detect image captions
        #

        self.detect_captions(
            blocks
        )

        return blocks

    # -----------------------------------------------------
    # Remove Noise
    # -----------------------------------------------------

    def remove_noise(
        self,
        blocks,
    ):

        cleaned = []

        visual_classes = {
            "object",
            "image",
            "photo",
            "figure",
        }

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
            # Keep visual blocks even when OCR
            # returns no text.
            #

            if (
                len(text) == 0
                and cls not in visual_classes
            ):
                continue

            #
            # Remove tiny OCR garbage.
            #
            # IMPORTANT:
            # Do not remove visual blocks.
            #

            if (
                len(text) <= 2
                and cls not in visual_classes
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

        print("=" * 60)

        return cleaned

    # -----------------------------------------------------
    # Masthead Detection
    # -----------------------------------------------------

    def detect_masthead(
        self,
        blocks,
        page_height,
    ):

        mastheads = [
            "THE ASIAN AGE",
            "THE HINDU",
            "TIMES OF INDIA",
            "INDIAN EXPRESS",
            "HINDUSTAN TIMES",
            "THE TELEGRAPH",
            "DECCAN CHRONICLE",
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

            if block.y2 > top_limit:
                continue

            if any(
                name in text
                for name in mastheads
            ) or any(
                name in raw_text
                for name in mastheads_devanagari
            ):

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

            block.column = min(
                int(
                    center
                    / column_width
                ),
                5,
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