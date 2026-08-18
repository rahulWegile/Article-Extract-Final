from pipeline.intelligence.local.continuation_matching import (
    text_overlap_score,
    score_continuation_pair,
    continuation_marker_target_page,
    match_all_continuations,
)


SOURCE_ARTICLE = {
    "page": 1,
    "article_id": "article_005",
    "content_type": "article",
    "headline": "Mamata urges EC to act on poll violence",
    "article_text": (
        "Officials said the situation had become intolerable in several "
        "districts and demanded immediate deployment of central forces "
        "in sensitive booths ahead of the upcoming phase of polling in "
        "the state. More on Page 3"
    ),
    "entities": ["Mamata Banerjee", "Election Commission"],
    "topics": ["politics"],
    "keywords": ["Election Commission", "poll violence", "central forces"],
}

TARGET_ARTICLE = {
    "page": 3,
    "article_id": "article_008",
    "content_type": "article",
    "headline": "",
    "article_text": (
        "had become intolerable in several districts and demanded "
        "immediate deployment of central forces in sensitive booths "
        "ahead of the upcoming phase of polling in the state and "
        "additional forces were rushed in overnight."
    ),
    "entities": ["Mamata Banerjee"],
    "topics": ["politics"],
    "keywords": ["central forces", "polling", "poll violence"],
}

UNRELATED_ARTICLE = {
    "page": 3,
    "article_id": "article_009",
    "content_type": "article",
    "headline": "Local cricket league finals begin this weekend",
    "article_text": (
        "The district cricket association announced the schedule for "
        "the finals of the annual league, with matches to be played at "
        "the municipal stadium starting Saturday morning."
    ),
    "entities": [],
    "topics": ["sports"],
    "keywords": ["cricket", "league"],
}


def test_text_overlap_score_high_for_true_continuation():

    score, shared, jaccard, ngram = text_overlap_score(
        SOURCE_ARTICLE["article_text"],
        TARGET_ARTICLE["article_text"],
    )

    assert shared >= 3, f"expected shared tokens, got {shared}"
    assert score > 0.15, f"expected non-trivial overlap score, got {score}"


def test_text_overlap_score_low_for_unrelated_articles():

    score, shared, jaccard, ngram = text_overlap_score(
        SOURCE_ARTICLE["article_text"],
        UNRELATED_ARTICLE["article_text"],
    )

    assert score < 0.15, f"expected near-zero overlap score, got {score}"


def test_continuation_marker_target_page_detected():

    target_page = continuation_marker_target_page(SOURCE_ARTICLE)

    assert target_page == 3, f"expected page 3, got {target_page}"


def test_score_continuation_pair_prefers_true_target():

    matching_score = score_continuation_pair(SOURCE_ARTICLE, TARGET_ARTICLE)
    unrelated_score = score_continuation_pair(SOURCE_ARTICLE, UNRELATED_ARTICLE)

    assert matching_score["score"] > unrelated_score["score"], (
        f"expected {matching_score['score']} > {unrelated_score['score']}"
    )


def test_match_all_continuations_links_known_pair():

    articles = [
        dict(SOURCE_ARTICLE),
        dict(TARGET_ARTICLE),
        dict(UNRELATED_ARTICLE),
    ]

    links, pending, report = match_all_continuations(articles)

    assert len(links) == 1, f"expected exactly one link, got {links}"
    assert links[0]["source_page"] == 1
    assert links[0]["source_article_id"] == "article_005"
    assert links[0]["target_page"] == 3
    assert links[0]["target_article_id"] == "article_008"


if __name__ == "__main__":

    test_text_overlap_score_high_for_true_continuation()
    test_text_overlap_score_low_for_unrelated_articles()
    test_continuation_marker_target_page_detected()
    test_score_continuation_pair_prefers_true_target()
    test_match_all_continuations_links_known_pair()

    print("=" * 60)
    print("ALL CONTINUATION MATCHING TESTS PASSED")
    print("=" * 60)
