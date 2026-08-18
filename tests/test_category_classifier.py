from pipeline.intelligence.local.category_classifier import CategoryClassifier


def test_category_classifier_distinguishes_politics_and_sports():

    classifier = CategoryClassifier()

    politics_category = classifier.classify(
        "The Prime Minister addressed parliament on the new tax bill "
        "amid opposition protests over the proposed reforms."
    )

    sports_category = classifier.classify(
        "The cricket team clinched the trophy after a last-over thriller "
        "in front of a packed home stadium."
    )

    assert politics_category == "politics", politics_category
    assert sports_category == "sports", sports_category
    assert politics_category != sports_category


def test_category_classifier_falls_back_to_other_for_empty_text():

    classifier = CategoryClassifier()

    assert classifier.classify("") == "other"
    assert classifier.classify("   ") == "other"


if __name__ == "__main__":

    test_category_classifier_distinguishes_politics_and_sports()
    test_category_classifier_falls_back_to_other_for_empty_text()

    print("=" * 60)
    print("ALL CATEGORY CLASSIFIER TESTS PASSED")
    print("=" * 60)
