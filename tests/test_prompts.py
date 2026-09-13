"""Tests pgrag.rag.prompts.build_prompt: per-query-type instruction sections
(comparison / lookup / general / entity / leveling) are included or excluded
as expected, the question and context are embedded, and the prompt encourages
reasoning from stated facts while blocking fabrication."""

from pgrag.rag.prompts import build_prompt


def test_comparison_prompt_includes_instructions():
    prompt = build_prompt(
        "what is the highest level cheese?", "context here", query_type="comparison"
    )
    assert "COMPARISON QUESTION DETECTED" in prompt
    assert "compare values" in prompt.lower() or "Compare" in prompt


def test_general_prompt_no_comparison_section():
    prompt = build_prompt("tell me about cheese", "context here", query_type="general")
    assert "COMPARISON QUESTION DETECTED" not in prompt


def test_lookup_prompt_no_comparison_section():
    prompt = build_prompt(
        "what level is statehelm sewer cheese?", "context here", query_type="lookup"
    )
    assert "COMPARISON QUESTION DETECTED" not in prompt


def test_prompt_contains_context():
    prompt = build_prompt("question", "my context", query_type="general")
    assert "my context" in prompt


def test_prompt_contains_question():
    prompt = build_prompt("my question", "context", query_type="general")
    assert "my question" in prompt


def test_leveling_prompt_includes_ranking_instructions():
    prompt = build_prompt(
        "What is the most efficient way to level Cheesemaking to level 25?",
        "context here",
        query_type="entity",
    )
    assert "LEVELING QUESTION DETECTED" in prompt
    assert "Rank those recipes/activities by XP efficiency" in prompt
    assert "XP Range Calculation" in prompt
    assert "Target Cumulative XP - Start Cumulative XP = Range XP needed" in prompt
    assert "Recipe Progression Path" in prompt
    assert "Optimal Strategy" in prompt


def test_entity_prompt_without_leveling_has_no_leveling_section():
    prompt = build_prompt(
        "What is Cheesemaking?",
        "context here",
        query_type="entity",
    )
    assert "LEVELING QUESTION DETECTED" not in prompt
    assert "ENTITY QUESTION DETECTED" in prompt


def test_prompt_allows_reasoning_but_blocks_fabrication():
    """The LLM may infer intent and synthesize from stated facts, but must
    not invent values. This rebalance is what lets it answer 'most efficient
    way to level X' from recipe XP numbers."""
    prompt = build_prompt("most efficient way to level Cheesemaking", "ctx", query_type="entity")
    # reasoning from context is encouraged...
    assert "connect information across documents" in prompt
    assert "conclusions that follow from the stated facts" in prompt
    assert "Figure out what the user is really asking" in prompt
    # ...but fabrication is still prohibited
    assert "NEVER fabricate" in prompt
    assert "do not invent facts" in prompt
    assert "Arithmetic directly derived from stated context values" in prompt


def test_old_overrestrictive_wording_removed():
    """'Do NOT guess, infer' conflated interpretation with fabrication and
    made the model refuse to synthesize. It must not come back."""
    prompt = build_prompt("tell me about cheese", "ctx", query_type="general")
    assert "Do NOT guess, infer" not in prompt


def test_entity_prompt_lists_all_named_characters():
    prompt = build_prompt("Tell me about the Gardening skill.", "ctx", query_type="entity")
    assert "reproduce EVERY name" in prompt


def test_comparison_prompt_quotes_descriptions_and_concludes_firmly():
    prompt = build_prompt(
        "Which is more damaging, Sword or Unarmed?", "ctx", query_type="comparison"
    )
    assert "Quote each compared item's own description verbatim" in prompt
    assert "Conclude firmly" in prompt
