import pytest

from copilot.text.arabic import (
    arabizi_to_arabic,
    has_arabic,
    looks_like_arabizi,
    normalize,
    plain_tokenize,
    tokenize,
)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("أحسن", "احسن"),
        ("إسكندرية", "اسكندريه"),
        ("مدرسة", "مدرسه"),
        ("على", "علي"),
        ("مَطْعَم", "مطعم"),  # diacritics
        ("مطــــعم", "مطعم"),  # tatweel
        ("٦ أكتوبر", "6 اكتوبر"),  # Eastern Arabic digits
        ("حلووووو", "حلوو"),  # elongation
    ],
)
def test_orthographic_variants_collapse(a, b):
    assert normalize(a) == normalize(b)


def test_definite_article_and_proclitics_stripped():
    assert tokenize("المطعم") == tokenize("مطعم")
    assert tokenize("بالمعادي") == tokenize("المعادي")


def test_dialect_expansion_adds_catalogue_terms():
    toks = tokenize("عايز اكل رخيص في المعادي")
    assert "مطعم" in toks and "خصم" in toks and "معادي" in toks
    assert "عايز" not in toks  # filler is dropped


def test_arabizi_detection_and_transliteration():
    assert looks_like_arabizi("3ayez") and looks_like_arabizi("smou7a")
    assert not looks_like_arabizi("restaurant")
    assert arabizi_to_arabic("mat3am") == "مطعم"
    assert arabizi_to_arabic("7amam") == "حامام"  # character fallback for unknown words


def test_arabizi_query_reaches_arabic_index_terms():
    toks = tokenize("3ayez gym fe madinet nasr")
    assert "جيم" in toks and "نصر" in toks


def test_english_is_kept_for_english_titles():
    assert tokenize("cheap massage in Maadi")[:1] == ["cheap"]
    assert "maadi" in tokenize("cheap massage in Maadi")


def test_has_arabic():
    assert has_arabic("gym في المعادي") and not has_arabic("gym in maadi")


def test_plain_tokenize_is_a_true_baseline():
    # no normalisation: hamza and ta marbuta variants stay distinct
    assert plain_tokenize("أحسن مدرسة") != plain_tokenize("احسن مدرسه")


def test_elongation_never_touches_digits():
    # regression: ORD-10001 became ORD-1001 and "under 1000" became "under 100"
    assert normalize("ORD-10001") == "ord-10001"
    assert normalize("تحت 1000") == "تحت 1000"
    assert normalize("حلووووو") == "حلوو"
