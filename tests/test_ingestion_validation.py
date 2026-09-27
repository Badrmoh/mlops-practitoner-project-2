from pathlib import Path

import pytest

from legrag.ingestion.pipeline import extract_articles_from_rows, iter_native_rows, load_doc


MAX_ARTICLE_TEXT_LENGTH = 5_000


@pytest.fixture(scope="session")
def articles():
    doc_path = Path("law.pdf.json")
    if not doc_path.exists():
        pytest.skip("law.pdf.json not found; run src/legrag/ingestion/convert.py first")
    return extract_articles_from_rows(iter_native_rows(load_doc(doc_path)))


def test_article_numbers_are_unique_and_contiguous(articles):
    article_numbers = [article.article_number for article in articles]
    duplicates = sorted({number for number in article_numbers if article_numbers.count(number) > 1})
    assert duplicates == []

    expected = set(range(min(article_numbers), max(article_numbers) + 1))
    missing = sorted(expected - set(article_numbers))
    assert missing == []


def test_every_record_has_non_empty_arabic(articles):
    empty_arabic = [article.article_number for article in articles if not article.text_ar.strip()]
    assert empty_arabic == []


def test_no_record_exceeds_sane_length(articles):
    oversized = [
        (article.article_number, len(article.text_en), len(article.text_ar))
        for article in articles
        if max(len(article.text_en), len(article.text_ar)) > MAX_ARTICLE_TEXT_LENGTH
    ]
    assert oversized == []


def test_repealed_articles_are_flagged(articles):
    repeal_records = [
        article
        for article in articles
        if "no longer exists" in article.text_en.casefold()
    ]

    assert repeal_records, "expected at least one repealed article record"
    unflagged = [article.article_number for article in repeal_records if not article.is_repealed]
    assert unflagged == []
