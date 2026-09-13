"""Synthesis detector — identify scattered answers for synthesis."""


def detect_scattered_answer(results: list, threshold: int = 3) -> bool:
    """Detect if search results indicate scattered/fragmented knowledge.

    Args:
        results: List of search result documents
        threshold: Minimum number of results to consider scattered

    Returns:
        True if results appear scattered across multiple sources
    """
    if len(results) < threshold:
        return False

    # Check for multiple different sources
    sources = set()
    for result in results:
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        sources.add(source)

    # If we have results from 3+ different sources, it's scattered
    if len(sources) >= 3:
        return True

    # Check for low relevance scores (high distances)
    distances = []
    for result in results:
        distance = result.get("distance", 0)
        if distance > 0:
            distances.append(distance)

    if distances:
        avg_distance = sum(distances) / len(distances)
        # High average distance indicates scattered results
        if avg_distance > 1.2 and len(results) >= threshold:
            return True

    return False


def should_synthesize(results: list, query_type: str = "general") -> bool:
    """Determine if synthesis should be triggered.

    Args:
        results: List of search result documents
        query_type: Type of query (comparison, lookup, general)

    Returns:
        True if synthesis should be attempted
    """
    # Don't synthesize for comparison queries - they use summaries
    if query_type == "comparison":
        return False

    # Check if results are scattered
    if detect_scattered_answer(results):
        return True

    # Check for mixed content types that might benefit from synthesis
    content_types = set()
    for result in results:
        metadata = result.get("metadata", {})
        table = metadata.get("table", "unknown")
        content_types.add(table)

    # If we have mixed wiki, curated, and recipe results, synthesis might help
    return len(content_types) >= 3
