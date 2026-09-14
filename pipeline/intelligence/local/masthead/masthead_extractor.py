from __future__ import annotations

import difflib
import re
from collections import Counter
from datetime import datetime, timedelta

import cv2

from pipeline.intelligence.local.masthead.date_parser import parse_date
from pipeline.intelligence.local.masthead.registry import (
    MastheadTemplate,
    load_registry,
    save_registry,
)
from pipeline.intelligence.local.masthead.script_detector import corroborates

GENERIC_TOP_CROP_FRACTION = 0.12
NAME_SIMILARITY_THRESHOLD = 0.80
AMBIGUOUS_MATCH_MARGIN = 0.05
OCR_CONFIDENCE_FLOOR = 0.55
MAX_FUTURE_SLACK_DAYS = 1
CONFIRMATIONS_REQUIRED = 3
LEARN_REGION_PADDING = 0.01

# Printed running-folio detection (see LocalMastheadExtractor.
# extract_folio_number / build_printed_page_map below). Purely
# positional -- no newspaper-specific template -- so it generalizes
# to any masthead instead of needing a registry entry per paper.
FOLIO_TOP_BAND_FRACTION = 0.08
FOLIO_MARGIN_FRACTION = 0.15
FOLIO_MIN_CONFIDENCE = 0.5
FOLIO_MIN_OFFSET_SAMPLES = 3
FOLIO_DOMINANT_OFFSET_RATIO = 0.7


def _normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", text.upper()).strip()


def _normalize_for_match(text: str) -> str:
    return _normalize(text).replace(" ", "")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _pad_region(bbox, padding):
    x1, y1, x2, y2 = bbox
    return (
        max(0.0, x1 - padding),
        max(0.0, y1 - padding),
        min(1.0, x2 + padding),
        min(1.0, y2 + padding),
    )


def _full_width_band(bbox, padding):
    _, y1, _, y2 = bbox
    return (
        0.0,
        max(0.0, y1 - padding),
        1.0,
        min(1.0, y2 + padding),
    )


def _fields_match(a, b):
    return (
        _normalize(a.get("newspaper_name", ""))
        == _normalize(b.get("newspaper_name", ""))
        and _normalize(a.get("edition", ""))
        == _normalize(b.get("edition", ""))
        and a.get("publish_date") == b.get("publish_date")
        and _normalize(a.get("language", "")) == _normalize(b.get("language", ""))
    )


def _crop_region(image, region):
    height, width = image.shape[:2]
    x1f, y1f, x2f, y2f = region

    x1 = max(0, int(x1f * width))
    y1 = max(0, int(y1f * height))
    x2 = min(width, int(x2f * width))
    y2 = min(height, int(y2f * height))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


class LocalMastheadExtractor:

    def __init__(self, ocr_engine, registry=None):
        self.reader = ocr_engine.reader
        self.templates = (
            registry if registry is not None else load_registry()
        )
        self._last_attempt = None

    def _ocr_region_lines(self, image, region):
        crop = _crop_region(image, region)

        if crop is None or crop.size == 0:
            return [], 0.0

        result = self.reader(crop)

        if result is None:
            return [], 0.0

        txts = getattr(result, "txts", None)
        if txts is None:
            txts = []

        scores = getattr(result, "scores", None)
        if scores is None:
            scores = []

        lines = [str(t).strip() for t in txts if str(t).strip()]

        confidences = []
        for score in scores:
            try:
                confidences.append(float(score))
            except (TypeError, ValueError):
                continue

        confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return lines, confidence

    def _ocr_region_text(self, image, region):
        lines, confidence = self._ocr_region_lines(image, region)
        return " ".join(lines), confidence

    def _ocr_region_lines_with_boxes(self, image, region):
        height, width = image.shape[:2]
        crop = _crop_region(image, region)

        if crop is None or crop.size == 0:
            return []

        x_offset = int(region[0] * width)
        y_offset = int(region[1] * height)

        result = self.reader(crop)

        if result is None:
            return []

        txts = getattr(result, "txts", None)
        if txts is None:
            txts = []

        scores = getattr(result, "scores", None)
        if scores is None:
            scores = []

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            boxes = []

        lines = []

        for index, text in enumerate(txts):
            text = str(text).strip()
            if not text:
                continue

            try:
                confidence = float(scores[index])
            except (IndexError, TypeError, ValueError):
                confidence = 0.0

            if index >= len(boxes):
                continue

            xs = [float(point[0]) for point in boxes[index]]
            ys = [float(point[1]) for point in boxes[index]]

            bbox = (
                (x_offset + min(xs)) / width,
                (y_offset + min(ys)) / height,
                (x_offset + max(xs)) / width,
                (y_offset + max(ys)) / height,
            )

            lines.append((text, confidence, bbox))

        return lines

    def _match_template(self, masthead_lines):
        normalized_lines = [
            _normalize_for_match(line) for line in masthead_lines if line
        ]

        scored = []

        for template in self.templates:
            variants = [
                _normalize_for_match(variant)
                for variant in template.name_ocr_variants
            ]

            best_ratio = max(
                (
                    difflib.SequenceMatcher(None, line, variant).ratio()
                    for line in normalized_lines
                    for variant in variants
                ),
                default=0.0,
            )

            scored.append((template, best_ratio))

        scored.sort(key=lambda item: item[1], reverse=True)

        if not scored or scored[0][1] < NAME_SIMILARITY_THRESHOLD:
            return None, "no registry entry cleared the name-match threshold"

        if (
            len(scored) > 1
            and (scored[0][1] - scored[1][1]) < AMBIGUOUS_MATCH_MARGIN
        ):
            return None, (
                f"ambiguous match: {scored[0][0].template_id} "
                f"({scored[0][1]:.2f}) vs {scored[1][0].template_id} "
                f"({scored[1][1]:.2f})"
            )

        return scored[0][0], None

    def _attempt(self, image, masthead_lines, masthead_confidence):
        masthead_text = " ".join(masthead_lines)

        template, reject_reason = self._match_template(masthead_lines)

        if template is None:
            print(f"Local masthead extractor: {reject_reason}")
            return None, None

        _, template_confidence = self._ocr_region_text(
            image, template.masthead_region
        )

        date_text, date_confidence = self._ocr_region_text(
            image, template.date_region
        )

        edition_text, edition_confidence = self._ocr_region_text(
            image, template.edition_region
        )

        mean_confidence = sum(
            (
                masthead_confidence,
                template_confidence,
                date_confidence,
                edition_confidence,
            )
        ) / 4

        if mean_confidence < OCR_CONFIDENCE_FLOOR:
            print(
                f"Local masthead extractor: mean OCR confidence "
                f"{mean_confidence:.2f} below floor "
                f"{OCR_CONFIDENCE_FLOOR:.2f}"
            )
            return template, None

        publish_date = parse_date(date_text)

        if not publish_date:
            print(
                f"Local masthead extractor: could not parse a single "
                f"unambiguous date from {date_text!r}"
            )
            return template, None

        parsed = datetime.strptime(publish_date, "%Y-%m-%d")

        if parsed > datetime.now() + timedelta(days=MAX_FUTURE_SLACK_DAYS):
            print(
                f"Local masthead extractor: parsed date {publish_date} "
                f"is implausibly far in the future"
            )
            return template, None

        normalized_edition_text = _normalize(edition_text)

        if not any(
            pattern in normalized_edition_text
            for pattern in template.edition_patterns
        ):
            print(
                f"Local masthead extractor: edition text "
                f"{edition_text!r} matched none of "
                f"{template.edition_patterns}"
            )
            return template, None

        if not corroborates(masthead_text, template.language):
            print(
                f"Local masthead extractor: script of {masthead_text!r} "
                f"doesn't corroborate expected language "
                f"'{template.language}' -- flagging, not blocking"
            )

        return template, {
            "newspaper_name": template.newspaper_name,
            "edition": template.edition,
            "publish_date": publish_date,
            "language": template.language,
        }

    def extract_metadata(self, image_path):
        if not self.templates:
            self._last_attempt = (str(image_path), None, None)
            return None

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Local masthead extractor: unable to read {image_path}")
            self._last_attempt = (str(image_path), None, None)
            return None

        masthead_lines, masthead_confidence = self._ocr_region_lines(
            image, (0.0, 0.0, 1.0, GENERIC_TOP_CROP_FRACTION)
        )

        if not masthead_lines:
            print("Local masthead extractor: no text in top-of-page crop")
            self._last_attempt = (str(image_path), None, None)
            return None

        template, result = self._attempt(
            image, masthead_lines, masthead_confidence
        )

        self._last_attempt = (str(image_path), template, result)

        if template is not None and template.verified and result is not None:
            return result

        if template is not None and not template.verified:
            print(
                f"Local masthead extractor: matched unverified "
                f"template '{template.template_id}' "
                f"({template.confirmation_count}/{CONFIRMATIONS_REQUIRED} "
                f"confirmations), not routing until confirmed"
            )

        return None

    def extract_folio_number(self, image_path):
        """
        Best-effort extraction of the printed running folio (page)
        number from a page image.

        Newspaper continuation markers ("Continued from P 1", "Turn
        to Page 14") quote the PRINTED page number, which frequently
        differs from the physical PDF page index once front-page ads
        or unnumbered jacket pages are involved (see
        build_printed_page_map). This reads the folio the same way a
        human would: a short standalone number sitting in the outer
        margin of the running head, near the top of the page.

        Deliberately positional/generic -- no newspaper name or
        layout is hardcoded, so it works for an unbounded universe of
        papers, unlike the template registry used for masthead
        identification above. Returns None on anything but a single,
        unambiguous digit-only candidate, so callers can safely fall
        back to treating the printed number as already being the
        physical page index.
        """
        image = cv2.imread(str(image_path))

        if image is None:
            return None

        lines = self._ocr_region_lines_with_boxes(
            image, (0.0, 0.0, 1.0, FOLIO_TOP_BAND_FRACTION)
        )

        candidates = set()

        for text, confidence, bbox in lines:

            if confidence < FOLIO_MIN_CONFIDENCE:
                continue

            stripped = text.strip()

            if not re.fullmatch(r"\d{1,3}", stripped):
                continue

            x1, _, x2, _ = bbox
            x_center = (x1 + x2) / 2

            in_left_margin = x_center <= FOLIO_MARGIN_FRACTION
            in_right_margin = x_center >= (1 - FOLIO_MARGIN_FRACTION)

            if not (in_left_margin or in_right_margin):
                continue

            candidates.add(int(stripped))

        if len(candidates) != 1:
            return None

        return candidates.pop()

    def record_gemini_result(self, image_path, gemini_metadata):
        if not isinstance(gemini_metadata, dict):
            return

        if not gemini_metadata.get("newspaper_name") or not gemini_metadata.get(
            "publish_date"
        ):
            return

        last = self._last_attempt

        if last is None or last[0] != str(image_path):
            return

        _, template, tentative_result = last

        if template is not None:
            if template.verified:
                return

            if tentative_result and _fields_match(
                tentative_result, gemini_metadata
            ):
                template.confirmation_count += 1

                print(
                    f"Local masthead extractor: '{template.template_id}' "
                    f"confirmed by Gemini agreement "
                    f"({template.confirmation_count}/"
                    f"{CONFIRMATIONS_REQUIRED})"
                )

                if template.confirmation_count >= CONFIRMATIONS_REQUIRED:
                    template.verified = True
                    print(
                        f"Local masthead extractor: '{template.template_id}' "
                        f"auto-verified, will now route locally"
                    )

                save_registry(self.templates)

            return

        self._learn_new_template(image_path, gemini_metadata)

    def _refine_line_text(self, image, bbox):
        crop = _crop_region(image, _full_width_band(bbox, LEARN_REGION_PADDING))

        if crop is None or crop.size == 0:
            return ""

        result = self.reader(crop)

        if result is None:
            return ""

        txts = getattr(result, "txts", None)
        if txts is None:
            txts = []

        return " ".join(str(t).strip() for t in txts if str(t).strip())

    def _find_line(self, image, lines, matches_fn):
        def _bbox_height(bbox):
            return bbox[3] - bbox[1]

        exact_matches = [line for line in lines if matches_fn(line[0])]

        if exact_matches:
            return min(exact_matches, key=lambda line: _bbox_height(line[2]))

        refined_matches = []

        for line in lines:
            refined_text = self._refine_line_text(image, line[2])
            if refined_text and matches_fn(refined_text):
                refined_matches.append((refined_text, line[1], line[2]))

        if not refined_matches:
            return None

        return min(refined_matches, key=lambda line: _bbox_height(line[2]))

    def _learn_new_template(self, image_path, gemini_metadata):
        image = cv2.imread(str(image_path))

        if image is None:
            return

        lines = self._ocr_region_lines_with_boxes(
            image, (0.0, 0.0, 1.0, GENERIC_TOP_CROP_FRACTION)
        )

        if not lines:
            return

        name_target = _normalize_for_match(gemini_metadata["newspaper_name"])

        name_line = max(
            lines,
            key=lambda line: difflib.SequenceMatcher(
                None, _normalize_for_match(line[0]), name_target
            ).ratio(),
            default=None,
        )

        if name_line is None:
            return

        name_ratio = difflib.SequenceMatcher(
            None, _normalize_for_match(name_line[0]), name_target
        ).ratio()

        if name_ratio < NAME_SIMILARITY_THRESHOLD:
            print(
                "Local masthead extractor: could not locate the masthead "
                "line for auto-learning, skipping"
            )
            return

        edition_value = gemini_metadata.get("edition") or ""
        date_value = gemini_metadata.get("publish_date") or ""

        date_line = self._find_line(
            image, lines, lambda text: parse_date(text) == date_value
        )

        edition_search_lines = lines

        if date_line is not None:
            date_y_center = (date_line[2][1] + date_line[2][3]) / 2

            nearby_lines = [
                line
                for line in lines
                if abs(
                    (line[2][1] + line[2][3]) / 2 - date_y_center
                )
                < 0.02
            ]

            if nearby_lines:
                edition_search_lines = nearby_lines

        edition_line = (
            self._find_line(
                image,
                edition_search_lines,
                lambda text: _normalize(edition_value) in _normalize(text),
            )
            if edition_value
            else None
        )

        if edition_line is None or date_line is None:
            print(
                "Local masthead extractor: could not locate date/edition "
                "lines for auto-learning, skipping"
            )
            return

        template_id = _slug(
            f"{gemini_metadata['newspaper_name']}_{edition_value}"
        )

        if any(t.template_id == template_id for t in self.templates):
            return

        template = MastheadTemplate(
            template_id=template_id,
            newspaper_name=gemini_metadata["newspaper_name"],
            edition=edition_value,
            language=gemini_metadata.get("language") or "English",
            name_ocr_variants=[name_line[0]],
            masthead_region=_pad_region(name_line[2], LEARN_REGION_PADDING),
            date_region=_full_width_band(date_line[2], LEARN_REGION_PADDING),
            edition_region=_full_width_band(
                edition_line[2], LEARN_REGION_PADDING
            ),
            edition_patterns=[_normalize(edition_value)],
            edition_primary_token=_normalize(edition_value),
            sample_source=str(image_path),
            verified=False,
            notes="auto-learned from a Gemini fallback result",
            confirmation_count=1,
        )

        self.templates.append(template)
        save_registry(self.templates)

        print(
            f"Local masthead extractor: learned draft template "
            f"'{template.template_id}' from Gemini "
            f"({CONFIRMATIONS_REQUIRED - 1} more agreeing document(s) "
            "needed before it routes locally)"
        )


def build_printed_page_map(folio_extractor, page_paths):
    """
    Map printed/folio page numbers -> physical PDF page index (1-based),
    by OCR-reading the running folio off every rendered page of a
    document.

    This is what lets a continuation marker like "Continued from P 1"
    resolve to the correct PDF page even when printed page 1 is not
    physical PDF page 1 (e.g. an unnumbered front jacket ad pushes the
    real front page a few PDF pages in).

    A folio number that resolves to more than one physical page (most
    commonly a supplement/section insert that restarts its own
    numbering, e.g. a "DelhiTimes" page 3 alongside the main section's
    page 3) is dropped rather than guessed -- callers should look up a
    printed page number in this map and fall back to treating it as
    the physical page index unchanged when it's absent, exactly as if
    no offset existed.
    """
    raw: dict[int, set[int]] = {}

    for index, page_path in enumerate(page_paths, start=1):

        try:
            folio = folio_extractor.extract_folio_number(page_path)
        except Exception:
            folio = None

        if folio is None:
            continue

        raw.setdefault(folio, set()).add(index)

    printed_page_map = {
        folio: next(iter(pdf_pages))
        for folio, pdf_pages in raw.items()
        if len(pdf_pages) == 1
    }

    # -----------------------------------------------------------
    # Extrapolate the front page(s).
    #
    # A newspaper's own front page conventionally carries no printed
    # folio number at all -- there is nothing there for
    # extract_folio_number to read -- so "printed page 1" (the most
    # common continuation target of all: the classic front-page jump
    # story) can never appear in the map above from direct OCR alone.
    #
    # When several OTHER pages agree on the same (pdf_index - folio)
    # offset, that offset almost certainly holds for the front page
    # too, so it is used to fill in any printed number still missing
    # from the map. This is arithmetic derived from THIS document's
    # own OCR evidence, not a per-newspaper rule, so it stays generic.
    #
    # Guarded so a document that mixes independently paginated
    # sections (e.g. a magazine insert restarting its own numbering)
    # -- where no single offset dominates -- safely contributes
    # nothing here rather than guessing. Even a wrong guess would only
    # ever cost a missed match, never a false one: this map only
    # selects which page to search on, and the text/slug scoring
    # thresholds in the repair pass are what actually decide whether
    # two crops are linked.
    # -----------------------------------------------------------

    offset_votes = Counter(
        pdf_index - folio
        for folio, pdf_index in printed_page_map.items()
    )

    if offset_votes:

        dominant_offset, dominant_count = offset_votes.most_common(1)[0]
        total_votes = sum(offset_votes.values())

        if (
            dominant_count >= FOLIO_MIN_OFFSET_SAMPLES
            and dominant_count / total_votes >= FOLIO_DOMINANT_OFFSET_RATIO
        ):

            max_pdf_page = len(page_paths)
            claimed_pdf_pages = set(printed_page_map.values())

            for printed in range(1, max_pdf_page + 1):

                if printed in printed_page_map:
                    continue

                inferred_pdf_page = printed + dominant_offset

                if not (1 <= inferred_pdf_page <= max_pdf_page):
                    continue

                if inferred_pdf_page in claimed_pdf_pages:
                    continue

                printed_page_map[printed] = inferred_pdf_page

    return printed_page_map
