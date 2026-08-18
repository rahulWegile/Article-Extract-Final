from pipeline.intelligence.local import field_extractors


def _ocr_result(lines_text: list[str]) -> dict:
    """Build a fake RapidOCREngine.process_article_crop-shaped result."""
    lines = [
        {
            "text": text,
            "confidence": 0.9,
            "bbox": {"x1": 0.0, "y1": float(index * 40), "x2": 300.0, "y2": float(index * 40 + 30)},
        }
        for index, text in enumerate(lines_text)
    ]
    return {
        "text": "\n".join(lines_text),
        "confidence": 0.9,
        "lines": lines,
        "line_count": len(lines),
    }


def test_headline_prefers_short_title_case_line():

    ocr_result = _ocr_result([
        "SC asks UP SIT to file report in three weeks",
        "By Our Correspondent",
        "The Supreme Court on Monday asked the Uttar Pradesh Special "
        "Investigation Team to submit its report within three weeks, "
        "noting that the matter required urgent attention from the state.",
    ])

    headline, subheadline = field_extractors.extract_headline_and_subheadline(ocr_result)

    assert headline == "SC asks UP SIT to file report in three weeks", headline


def test_headline_falls_back_to_first_line_when_no_strong_candidate():

    ocr_result = _ocr_result([
        "the meeting continued late into the evening without a clear resolution "
        "as both sides refused to compromise on the terms of the agreement.",
    ])

    headline, subheadline = field_extractors.extract_headline_and_subheadline(ocr_result)

    assert headline.startswith("the meeting continued")


def test_extract_author_detects_byline():

    text = "By Rahul Kumar\nNEW DELHI: The minister said today that..."

    author = field_extractors.extract_author(text)

    assert author == "Rahul Kumar", author


def test_extract_author_detects_agency():

    text = "NEW DELHI: The report was filed by PTI on Tuesday evening."

    author = field_extractors.extract_author(text)

    assert author == "PTI", author


def test_extract_author_none_when_no_signal():

    text = "The government announced the policy on Tuesday without further comment."

    author = field_extractors.extract_author(text)

    assert author is None


def test_extract_location_and_date_from_dateline():

    text = "NEW DELHI, Aug 12: The government today announced a new policy."

    location, date = field_extractors.extract_location_and_date(text)

    assert location == "NEW DELHI", location
    assert date is not None and "Aug 12" in date, date


def test_extract_location_returns_none_without_dateline():

    text = "The government today announced a new policy without a location tag."

    location, date = field_extractors.extract_location_and_date(text)

    assert location is None
    assert date is None


def test_classify_content_type_article_by_default():

    content_type = field_extractors.classify_content_type(
        "A" * 200, caption_score=0.0, has_figure_block=False
    )

    assert content_type == "article"


def test_classify_content_type_photo_caption():

    content_type = field_extractors.classify_content_type(
        "PTI Photo: Workers repair the damaged road after heavy rain.",
        caption_score=field_extractors.caption_score(
            "PTI Photo: Workers repair the damaged road after heavy rain."
        ),
        has_figure_block=True,
    )

    assert content_type == "photo_caption", content_type


def test_classify_content_type_advertisement():

    content_type = field_extractors.classify_content_type(
        "Visit www.example-shop.com for the best deals. Call now for booking!",
        caption_score=0.0,
        has_figure_block=False,
    )

    assert content_type == "advertisement", content_type


def test_detect_continuation_finds_marker():

    article = {
        "article_text": "The story develops further. More on Page 5",
    }

    continuation = field_extractors.detect_continuation(article)

    assert continuation["is_continued"] is True
    assert continuation["next_page"] == 5


def test_detect_continuation_no_marker():

    article = {
        "article_text": "The story ends here with a clear conclusion.",
    }

    continuation = field_extractors.detect_continuation(article)

    assert continuation["is_continued"] is False
    assert continuation["next_page"] is None


if __name__ == "__main__":

    test_headline_prefers_short_title_case_line()
    test_headline_falls_back_to_first_line_when_no_strong_candidate()
    test_extract_author_detects_byline()
    test_extract_author_detects_agency()
    test_extract_author_none_when_no_signal()
    test_extract_location_and_date_from_dateline()
    test_extract_location_returns_none_without_dateline()
    test_classify_content_type_article_by_default()
    test_classify_content_type_photo_caption()
    test_classify_content_type_advertisement()
    test_detect_continuation_finds_marker()
    test_detect_continuation_no_marker()

    print("=" * 60)
    print("ALL FIELD EXTRACTOR TESTS PASSED")
    print("=" * 60)
