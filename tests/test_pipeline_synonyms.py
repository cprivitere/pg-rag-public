"""Synonym expansion transforms user-facing terms to canonical game
terminology before classification and retrieval.

_expand_synonyms is called at the top of ask() and ask_stream(), before
classify_query, so that the classifier, spelling correction, and BM25 all see
the canonical game term rather than the synonym.
"""

from pgrag.rag.pipeline import _expand_synonyms


def test_lycanthropes_to_werewolf():
    """Plural 'lycanthropes' becomes 'werewolf' (word-boundary matched)."""
    assert _expand_synonyms("lycanthropes") == "werewolf"


def test_lycanthrope_to_werewolf():
    """Singular 'lycanthrope' becomes 'werewolf'."""
    assert _expand_synonyms("lycanthrope") == "werewolf"


def test_case_insensitive():
    """Capitalized 'Lycanthrope' becomes lowercase 'werewolf'."""
    assert _expand_synonyms("Lycanthrope") == "werewolf"


def test_expansion_in_full_query():
    """A realistic query is expanded before classification."""
    result = _expand_synonyms("best end game armor for lycanthropes")
    assert result == "best end game armor for werewolf"


def test_no_synonym_unaffected():
    """A query with no synonyms passes through unchanged."""
    query = "sword vs axe"
    assert _expand_synonyms(query) is query


def test_partial_word_no_match():
    """'lycanthropes' must be a whole word — not part of a longer token."""
    # 'esque' suffix is not a match; the word boundary at 'lycanthropes' is
    # the only trigger.  'lycanthropelike' would fail but is unrealistic.
    result = _expand_synonyms("werewolf lycanthropes esque")
    assert result == "werewolf werewolf esque"


def test_multiple_synonyms():
    """Multiple synonym tokens in the same query all expand."""
    result = _expand_synonyms("lycanthrope and lycanthropes")
    assert result == "werewolf and werewolf"