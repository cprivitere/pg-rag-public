"""Tests the embed_batch overflow fallback: on _InputTooLong the batch is
bisected, isolating only the genuinely over-window texts instead of
re-embedding the whole batch one-at-a-time (the O(n) regression in REVIEW.md)."""
import pgrag.embeddings.llama_embeddings as emb


def _resp(vectors):
    """Build a llama.cpp /embedding response for a batch of 1-vector texts."""
    return [{"embedding": [v]} for v in vectors]


def test_embed_batch_bisects_on_too_long_isolating_offender(monkeypatch):
    """A 4-text batch where the full-batch post 400s but the two halves
    succeed must bisect, not serialize all 4 via _embed_one."""
    calls = {"count": 0, "budgets": []}

    def fake_post(texts, budget):
        calls["count"] += 1
        calls["budgets"].append(budget)
        # Full batch (4 texts) -> too long; halves (2 texts) -> ok.
        if len(texts) == 4:
            raise emb._InputTooLong
        return _resp([f"v{t}" for t in texts])

    monkeypatch.setattr(emb, "_post", fake_post)
    monkeypatch.setattr(emb, "validate_embeddings", lambda v: v)

    out = emb.embed_batch(["a", "b", "c", "d"])
    assert out == ["va", "vb", "vc", "vd"]          # order preserved
    assert calls["count"] == 3                     # 1 full + 2 halves, NOT 4
    assert all(b == emb.MAX_EMBED_CHARS for b in calls["budgets"])  # no shrink


def test_embed_batch_leaf_falls_through_to_embed_one(monkeypatch):
    """A single-text batch that 400s at MAX_EMBED_CHARS shrinks via _embed_one."""
    calls = {"budgets": []}

    def fake_post(texts, budget):
        calls["budgets"].append(budget)
        if budget == emb.MAX_EMBED_CHARS:
            raise emb._InputTooLong
        return _resp([f"v{budget}"])

    monkeypatch.setattr(emb, "_post", fake_post)
    monkeypatch.setattr(emb, "validate_embeddings", lambda v: v)

    out = emb.embed_batch(["toolong"])
    assert out == [f"v{emb._TRUNC_STEPS[1]}"]   # shrunk to next budget
    assert emb.MAX_EMBED_CHARS in calls["budgets"]
    assert emb._TRUNC_STEPS[1] in calls["budgets"]