#!/usr/bin/env python3
"""Analyze stringliteral.json for missed player-facing mechanic prose.

Loads the full stringliteral.json, filters for long player-facing entries,
groups them by topic, and produces a report for manual review.

Output: docs/discovered-mechanic-prose.md (appended to)
"""

import json
import re
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
IL2CPP_DIR = Path(__file__).resolve().parent.parent / "data" / "il2cpp"
STRINGLIT_PATH = IL2CPP_DIR / "out_lean" / "Dump0" / "stringliteral.json"
DOT_PATH = Path(__file__).resolve().parent.parent / "docs" / "discovered-mechanic-prose.md"

# ── player-facing keywords ─────────────────────────────────────────────────
PLAYER_KEYWORDS = [
    "you",
    "your",
    "skill",
    "damage",
    "craft",
    "combat",
    "xp",
    "level",
    "buff",
    "vendor",
    "shop",
    "favor",
    "heal",
    "armor",
    "weapon",
    "quest",
    "train",
    "death",
    "dying",
    "curse",
    "loot",
    "corpse",
    "item",
    "recipe",
    "npc",
    "enemy",
    "monster",
    "player",
    "attack",
    "defense",
    "stamina",
    "power",
    "mana",
    "health",
    "crafting",
    "gathering",
    "mining",
    "forestry",
    "transmutation",
    "sword",
    "staff",
    "dagger",
    "club",
    "hammer",
    "axe",
    "bow",
    "crossbow",
    "shield",
    "potion",
    "elixir",
    "food",
    "drink",
    "ingredient",
    "material",
    "equip",
    "wear",
    "slot",
    "inventory",
    "bag",
    "backpack",
    "storage",
    "vault",
    "chest",
    "vendor",
    "buy",
    "sell",
    "trade",
    "auction",
    "consignment",
    "coin",
    "gold",
    "token",
    "point",
    "gift",
    "favor",
    "reputation",
    "standing",
    "title",
    "achievement",
    "medal",
    "trophy",
    "housing",
    "home",
    "deed",
    "plot",
    "decorate",
    "furniture",
    "pet",
    "mount",
    "costume",
    "emote",
    "dance",
    "music",
    "instrument",
    "gym",
    "sport",
    "golf",
    "game",
    "puzzle",
    "race",
    "event",
    "holiday",
    "festival",
    "season",
    "daily",
    "weekly",
    "monthly",
    "timer",
    "cooldown",
    "cooldown",
    "recovery",
    "regeneration",
    "resistance",
    "weakness",
    "vulnerability",
    "immunity",
    "absorb",
    "reflect",
    "block",
    "parry",
    "dodge",
    "evade",
    "critical",
    "glance",
    "hit",
    "miss",
    "stun",
    "slow",
    "snare",
    "root",
    "knockback",
    "knockdown",
    "pull",
    "push",
    "teleport",
    "portal",
    "travel",
    "fast travel",
    "waypoint",
    "bind",
    "respawn",
    "corpse run",
    "spirit",
    "ghost",
    "revive",
    "resurrect",
    "death",
    "penalty",
    "debt",
    "repair",
    "durability",
    "enchant",
    "augment",
    "infuse",
    "transmute",
    "salvage",
    "disenchant",
    "recycle",
    "upgrade",
    "downgrade",
    "reroll",
    "reforge",
    "reroll",
    "polish",
    "dye",
    "paint",
    "skin",
    "appearance",
    "glamour",
    "style",
    "fashion",
    "cosmetic",
    "visual",
    "particle",
    "effect",
    "animation",
    "sound",
    "music",
    "audio",
    "volume",
    "master",
    "brightness",
    "gamma",
    "resolution",
    "window",
    "fullscreen",
    "vsync",
    "fps",
    "frame",
    "rate",
    "quality",
    "detail",
    "shadow",
    "light",
    "fog",
    "ambient",
    "occlusion",
    "bloom",
    "hdr",
    "antialiasing",
    "filter",
    "texture",
    "shader",
    "model",
    "mesh",
    "terrain",
    "vegetation",
    "water",
    "sky",
    "weather",
    "time",
    "day",
    "night",
    "cycle",
    "season",
    "winter",
    "spring",
    "summer",
    "fall",
    "dungeon",
    "raid",
    "trial",
    "instance",
    "scenario",
    "mission",
    "objective",
    "goal",
    "task",
    "errand",
    "favor",
    "quest",
    "story",
    "lore",
    "history",
    "legend",
    "myth",
    "faction",
    "clan",
    "guild",
    "group",
    "party",
    "solo",
    "multiplayer",
    "co-op",
    "pvp",
    "duel",
    "arena",
    "battleground",
    "war",
    "conflict",
    "truce",
    "peace",
    "alliance",
    "enemy",
    "hostile",
    "aggressive",
    "passive",
    "neutral",
    "friendly",
    "ally",
    "companion",
    "follower",
    "minion",
    "summon",
    "fairy",
    "race",
    "class",
    "archetype",
    "role",
    "build",
    "spec",
    "talent",
    "passive",
    "active",
    "ability",
    "power",
    "move",
    "spell",
    "chant",
    "incantation",
    "prayer",
    "ritual",
    "technique",
    "form",
    "stance",
    "mode",
    "state",
    "transform",
    "shape",
    "shift",
    "rage",
]

# ── regex for format strings like {0} ─────────────────────────────────────
_FORMAT_RE = re.compile(r"\{[-]?[0-9]")

# ── stop words ─────────────────────────────────────────────────────────────
STOP_KEYWORDS = [
    "please report at",
    "mesh baker",
    "texture packer",
    "submesh",
    "assetbundle",
    "vivox",
    "compute shader",
    "shader",
    "depth buffer",
    "render texture",
    "gpu",
    "graphics api",
    "render graph",
    "shader graph",
    "material",
    "prefab",
    "animator",
    "ik target",
    "bone",
    "skinned mesh",
]


def load_strings(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def is_player_facing(val: str) -> bool:
    """Check if a string looks like player-facing content."""
    low = val.lower()
    hits = sum(1 for kw in PLAYER_KEYWORDS if kw in low)
    return hits >= 3


def has_stop(val: str) -> bool:
    low = val.lower()
    return any(sw in low for sw in STOP_KEYWORDS)


def normalize(val: str) -> str:
    val = val.replace("\r", " ")
    val = val.replace("\n", " ")
    return " ".join(val.split())


def classify_topic(val: str) -> str:
    """Heuristic topic classification based on content."""
    low = val.lower()
    topics = []

    if any(
        w in low
        for w in [
            "shop",
            "vendor",
            "sell",
            "buy",
            "price",
            "purchase",
            "consignment",
            "customer",
            "item listing",
            "stack",
            "reserve",
        ]
    ):
        topics.append("shopping--vendors")
    if any(w in low for w in ["skill", "xp", "level", "train", "ability", "combat bar"]):
        topics.append("skills--combat")
    if any(
        w in low
        for w in [
            "armor",
            "health",
            "power",
            "regeneration",
            "food",
            "hunger",
            "thirst",
            "metabolism",
            "naked",
        ]
    ):
        topics.append("gear--survival")
    if any(w in low for w in ["death", "dying", "respawn", "corpse", "revive"]):
        topics.append("death--penalties")
    if any(w in low for w in ["curse", "cursed", "decurse"]):
        topics.append("curses--afflictions")
    if any(w in low for w in ["fairy", "fae", "fly", "wing", "racial"]):
        topics.append("races--fairy")
    if any(w in low for w in ["loot", "corpse", "container", "pick up", "drop"]):
        topics.append("loot--gathering")
    if any(w in low for w in ["quest", "goal", "directed", "tracker", "task"]):
        topics.append("quests--objectives")
    if any(w in low for w in ["hangout", "favor", "npc friend", "npc", "befriend"]):
        topics.append("npcs--social")
    if any(w in low for w in ["transmutation", "transmute", "attune", "enchant", "augment"]):
        topics.append("crafting--transmutation")
    if any(
        w in low
        for w in [
            "settings",
            "option",
            "enable",
            "disable",
            "toggle",
            "show",
            "display",
            "graphics",
            "quality",
            "font",
        ]
    ):
        topics.append("settings--ui")
    if any(
        w in low
        for w in [
            "chat",
            "message",
            "whisper",
            "tell",
            "say",
            "shout",
            "emote",
            "spoiler",
            "bracket",
        ]
    ):
        topics.append("communication--chat")
    if any(
        w in low
        for w in [
            "party",
            "group",
            "friend",
            "invite",
            "player",
            "other player",
            "nearby",
            "global",
        ]
    ):
        topics.append("social--multiplayer")
    if any(w in low for w in ["target", "tab", "cycle", "enemy select", "auto-target"]):
        topics.append("combat--targeting")
    if any(
        w in low
        for w in ["damage number", "combat log", "floaty", "healing number", "particle", "effect"]
    ):
        topics.append("combat--feedback")
    if any(w in low for w in ["sprint", "run", "shift key", "power drain", "movement"]):
        topics.append("movement--sprinting")
    if any(w in low for w in ["race", "mount", "animal"]):
        topics.append("movement--mounts")
    if any(w in low for w in ["map", "minimap", "landmark", "area"]):
        topics.append("navigation--map")
    if any(
        w in low
        for w in [
            "item",
            "equip",
            "inventory",
            "bag",
            "backpack",
            "container",
            "stack",
            "drag",
            "drop",
        ]
    ):
        topics.append("items--inventory")
    if any(w in low for w in ["gift", "give", "present", "favor"]):
        topics.append("items--gifts")

    if not topics:
        topics.append("uncategorized")

    return "/".join(sorted(set(topics)))


def main():
    print(f"Loading {STRINGLIT_PATH}...")
    data = load_strings(STRINGLIT_PATH)
    print(f"Total entries: {len(data)}")

    # Filter: long strings, no format/HTML, player-facing
    candidates = []
    for entry in data:
        val = entry.get("value", "")
        if not isinstance(val, str):
            continue
        norm = normalize(val)
        if len(norm) < 60:
            continue
        if "<" in norm:
            continue
        if _FORMAT_RE.search(norm):
            continue
        if has_stop(norm):
            continue
        if not is_player_facing(norm):
            continue
        candidates.append((entry["index"], norm))

    print(f"Candidates (>=60 chars, player-facing, no format/HTML): {len(candidates)}")

    # Deduplicate near-identical (same leading ~80 chars)
    seen_prefixes = set()
    deduped = []
    for idx, norm in candidates:
        prefix = norm[:80].lower()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        deduped.append((idx, norm))

    print(f"After dedup: {len(deduped)}")

    # Classify by topic
    by_topic: dict[str, list[tuple[int, str]]] = {}
    for idx, norm in deduped:
        topic = classify_topic(norm)
        by_topic.setdefault(topic, []).append((idx, norm))

    # Output report to terminal
    print("\n" + "=" * 70)
    print("TOPIC CLASSIFICATION REPORT")
    print("=" * 70)

    for topic in sorted(by_topic.keys()):
        items = by_topic[topic]
        print(f"\n{'─' * 60}")
        print(f"  {topic}  ({len(items)} items)")
        print(f"{'─' * 60}")
        for idx, norm in items[:10]:
            preview = norm[:150].replace("\n", " ")
            print(f"    [{idx}] {preview}...")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more")

    # Write report to docs
    DOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOT_PATH, "w", encoding="utf-8") as f:
        f.write("# Discovered Mechanic Prose — stringliteral.json\n\n")
        f.write("Generated by `scripts/analyze_stringliteral.py`\n")
        f.write(f"Total stringliteral entries: {len(data)}\n")
        f.write(f"Player-facing candidates (>=60 chars, deduped): {len(deduped)}\n\n")
        f.write("## Topic Groups\n\n")
        for topic in sorted(by_topic.keys()):
            items = by_topic[topic]
            f.write(f"### {topic}  ({len(items)} items)\n\n")
            for idx, norm in items:
                f.write(f"- `[{idx}]` {norm}\n")
            f.write("\n")
        f.write("## New MECHANIC_TOPICS Patterns to Add\n\n")
        f.write("<!-- Manually filled after review -->\n")
        f.write("| Topic | Title | Regex Pattern | Notes |\n")
        f.write("|-------|-------|---------------|-------|\n")

    print(f"\nReport written to {DOT_PATH}")
    print(f"Total candidates: {len(deduped)}")
    print(
        "\nNext step: review candidates above and add new MECHANIC_TOPICS patterns to decomp_builder.py"
    )


if __name__ == "__main__":
    main()
