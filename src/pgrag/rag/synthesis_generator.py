"""Synthesis generator — use LLM to create curated docs from scattered chunks."""

from pgrag.rag.llm import LLMServerError, generate

SYNTHESIS_PROMPT = """You are a game assistant for Project Gorgon. The user asked a question but the search results are scattered across multiple sources.

Synthesize the following information into a clear, comprehensive answer. Create a well-organized document that combines all the relevant information.

User Question: {query}

Scattered Information Sources:
{sources}

Create a comprehensive, well-organized response that:
1. Directly answers the user's question
2. Combines information from all sources
3. Removes redundancy
4. Presents information in a clear, logical order
5. Uses markdown formatting for readability

Response:"""


def synthesize_answer(query: str, results: list, generation=None) -> str:
    """Synthesize scattered results into a coherent answer.

    Args:
        query: Original user query
        results: List of search result documents

    Returns:
        Synthesized answer as string
    """
    # Format sources from results
    sources_text = ""
    for i, result in enumerate(results, 1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        name = metadata.get("name", "Unknown")

        sources_text += f"\nSource {i} ({source} - {name}):\n{text[:500]}...\n"

    prompt = SYNTHESIS_PROMPT.format(query=query, sources=sources_text)

    try:
        response = generate(prompt, **(generation or {}))
        return response
    except LLMServerError:
        # If LLM fails, return a simple concatenation
        return _fallback_synthesis(query, results)


def _fallback_synthesis(query: str, results: list) -> str:
    """Fallback synthesis when LLM is unavailable.

    Args:
        query: Original user query
        results: List of search result documents

    Returns:
        Simple concatenated answer
    """
    parts = [f"Based on multiple sources, here's what I found about: {query}\n"]

    for _i, result in enumerate(results, 1):
        text = result.get("text", "")
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        name = metadata.get("name", "Unknown")

        # Take first 200 chars of each source
        excerpt = text[:200].replace("\n", " ").strip()
        if len(text) > 200:
            excerpt += "..."

        parts.append(f"**{name}** ({source}): {excerpt}")

    return "\n\n".join(parts)
