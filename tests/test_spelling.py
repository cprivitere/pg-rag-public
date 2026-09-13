"""Tests pgrag.rag.spelling.correct_query typo correction against a fake word
vocab: transposed and in-sentence typos are corrected, while vocab tokens,
short tokens, tokens with no close match, and low-ratio words are left alone."""

import pytest

from pgrag.rag import spelling


@pytest.fixture(autouse=True)
def fake_vocab(monkeypatch):
    monkeypatch.setattr(
        spelling,
        "_word_vocab",
        lambda: {"mushroom", "mortaferus", "cheese", "sword", "shroom", "slate"},
    )


def test_corrects_transposed_typo():
    assert spelling.correct_query("msurhoom") == "mushroom"


def test_corrects_typo_in_sentence():
    out = spelling.correct_query("Whast is this highest level msurhoom?")
    assert "mushroom" in out


def test_vocab_tokens_untouched():
    q = "what is the highest level mushroom"
    assert spelling.correct_query(q) == q


def test_short_tokens_never_corrected():
    out = spelling.correct_query("wht is ths msurhoom")
    assert out == "wht is ths mushroom"


def test_no_close_match_left_alone():
    out = spelling.correct_query("what is the xyzzyblorp item")
    assert "xyzzyblorp" in out


def test_low_ratio_words_not_corrected():
    assert spelling.correct_query("locate") == "locate"
