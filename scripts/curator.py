"""Heuristic curation: regex-scan wiki pages for fragmented topics and emit
template curated docs. Deliberately NO LLM — deterministic and offline, so the
curated docs are stable retrieval anchors that don't vary run-to-run."""

from pathlib import Path

from pgrag.config import CURATED_DIR

WIKI_DIR = Path("data/wiki")


def scan_wiki_for_fragments():
    """Scan wiki files for fragmented knowledge patterns."""
    fragments = []

    if not WIKI_DIR.exists():
        return fragments

    # Look for common fragmented topics
    patterns = {
        "area_levels": ["level.*area", "area.*level", "zone.*level"],
        "skill_trainers": ["unlock.*level", "trains.*level", "teaches.*level"],
        "crafting_progressions": ["requires level", "skill level.*req"],
    }

    for txt_file in WIKI_DIR.glob("*.txt"):
        try:
            content = txt_file.read_text(encoding="utf-8")

            for topic, regexes in patterns.items():
                for pattern in regexes:
                    import re

                    if re.search(pattern, content, re.IGNORECASE):
                        fragments.append(
                            {"file": txt_file.name, "topic": topic, "preview": content[:200]}
                        )
                        break
        except Exception:
            continue

    return fragments


def create_curated_from_fragments(topic: str, fragments: list):
    """Build a deterministic template doc for one fragment topic (no LLM).

    Emits a `==Heading==` + "Auto-generated from N wiki sources" listing of up
    to the first 5 matching file snippets. Deliberately not LLM-curated: the
    output must be reproducible so the curator is offline-testable and the
    curated docs are stable retrieval anchors.
    """
    content = f"=={topic.replace('_', ' ').title()}==\n"
    content += f"Auto-generated from {len(fragments)} wiki sources.\n\n"

    for frag in fragments[:5]:  # Limit to 5 sources
        content += f"Source: {frag['file']}\n"
        content += f"{frag['preview']}\n\n"

    return content


def run_curator():
    """Main curator agent loop."""
    print("Starting curator agent...")

    # Ensure curated directory exists
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    # Scan for fragments
    fragments = scan_wiki_for_fragments()
    print(f"Found {len(fragments)} potential fragments")

    # Group by topic
    topics = {}
    for frag in fragments:
        topic = frag["topic"]
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(frag)

    # Create curated docs for each topic
    created = 0
    for topic, frags in topics.items():
        if len(frags) >= 3:  # Only create if enough sources
            curated_path = CURATED_DIR / f"{topic}_curated.txt"
            content = create_curated_from_fragments(topic, frags)

            # Regenerate when content changed (V20) — stale docs must not persist
            existed = curated_path.exists()
            if not existed or curated_path.read_text(encoding="utf-8") != content:
                curated_path.write_text(content, encoding="utf-8")
                print(f"{'Updated' if existed else 'Created'}: {curated_path.name}")
                created += 1

    print(f"Curator complete. Created {created} new curated documents.")
    return created


if __name__ == "__main__":
    run_curator()
