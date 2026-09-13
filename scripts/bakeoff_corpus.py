"""Bake-off corpus generator.

Builds an embedder-bake-off eval corpus (data/bakeoff_corpus.json) from the
generated data/documents.json.

Why the corpus is cluster-sampled: an IR eval is only meaningful if each query's
gold (relevant) docs are *present* to be retrieved. A blind sample of 500 docs
from ~258k rarely contains an entity's cross-table footprint, so golds collapse
to 1 and MRR/NDCG/Recall degenerate. Instead:

- Group docs into *entity clusters*: a doc's full footprint across the source
  (every doc sharing the same case-normalized metadata.name across the entity
  tables: item <-> recipe <-> quest <-> wiki, ability <-> effect <-> skill, ...)
  plus its own multi-chunk page family. Names matching more than AMBIG_CAP docs
  are "ambiguous" and not grouped, so generic nouns can't form false clusters.
- Sample whole clusters (so each query's golds are all in the corpus), with a
  per-table doc budget (no table may exceed cap_share of the corpus) for a
  representative spread, and force one rich cluster per interesting query type
  so every query type is covered. Deterministic via SEED.
- Queries are a 3-tier, 40-query set, emitted needle -> fill -> paraphrase:
  - needle (hard, N_QUERIES-tier single-gold): the gold is one specific doc
    carrying a distinguishing structured fact (skill_level_req / value / skill
    / damage_type / stack_size / level / location), chosen for MAX same-name
    distractor pressure so MRR/hit@k keep dynamic range.
  - fill (multi-gold): whole-cluster golds capped at 10 so Recall@10/NDCG can
    actually reach 1.0 (uncapped 49-gold wiki clusters made the denominator
    structurally unreachable).
  - paraphrase: two surface phrasings target the SAME golds, so a model that
    matches one wording but not the other is a lexical matcher — the
    within-model a-vs-b variance is the semantic-vs-lexical signal.
- The corpus carries a fingerprint (gold stats, tier counts, type distribution,
  content hash) so a changed/regenerated run is detectable.

Run: uv run python scripts/bakeoff_corpus.py
Writes: data/bakeoff_corpus.json
"""

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DOCS_PATH = DATA / "documents.json"
OUT_PATH = DATA / "bakeoff_corpus.json"

TARGET = 500  # corpus docs (golds are by-construction present)
N_QUERIES = 40  # evaluation queries (needle + fill + 20 paraphrase; fill adapts so total == 40)
HARD_N = 9  # cap on the single-gold "needle" tier (8 templates exist)
SEED = 42
CAP_SHARE = 0.35  # no single table's docs >35% of the corpus sample
AMBIG_CAP = 30  # name with more entity-table matches: too generic to group
INTERESTING = ("recipe", "item", "wiki", "ability", "quest", "skillprofile", "effect")
# Needle queries: (type, metadata field, question). The gold is the ONE doc of
# that type whose text contains the field value, chosen for max same-name
# sibling distractor pressure (see build_queries). Field must appear in the doc
# text (verified) so the gold is retrievable.
HARD_TEMPLATES = [
    ("recipe", "skill_level_req", "What skill level is required to craft {name}?"),
    ("recipe", "skill", "What skill is used to craft {name}?"),
    ("item", "value", "What is the buy value of the item {name}?"),
    ("item", "stack_size", "What is the stack size of {name}?"),
    ("ability", "skill", "Which skill does the {name} ability belong to?"),
    ("ability", "damage_type", "What damage type is the {name} ability?"),
    ("ability", "level", "What level is required for the {name} ability?"),
    ("quest", "location", "Where does the quest {name} take place?"),
]
# Paraphrase tier: two surface phrasings per type, each query targeting the SAME
# golds (the whole cluster, capped like fill). A model that scores both is
# semantic; one that drops a->b is lexical.
PARAPHRASE_TEMPLATES = {
    "recipe": [
        "What materials are needed to craft {name}?",
        "What ingredients do I need to make {name}?",
    ],
    "item": ["What does the item {name} do?", "What is {name} used for?"],
    "ability": ["What does the ability {name} do?", "Describe the {name} ability"],
    "skillprofile": [
        "Tell me everything about the {name} skill",
        "What are the details of the {name} skill profile?",
    ],
    "wiki": [
        "What does the wiki say about {title}?",
        "What information is available about {title}?",
    ],
    "effect": ["What effect does {name} have?", "What does the {name} effect do?"],
    "quest": ["How do I start the quest {name}?", "How do I complete the quest {name}?"],
}
# Tables whose same-name records are plausibly the same entity (cross-source link).
CROSS_TABLES = {
    "item",
    "recipe",
    "quest",
    "ability",
    "effect",
    "skill",
    "skillprofile",
    "leveling",
    "npc",
    "wiki",
    "itemuse",
    "abilitykeyword",
}

CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")

FILL_TEMPLATES = {
    "wiki": "What information is available about {title}?",
    "skillprofile": "What are the details of the {name} skill profile?",
    "recipe": "What materials are needed to craft {name}?",
    "item": "What is {name} used for?",
    "ability": "What does the ability {name} do?",
    "quest": "How do I complete the quest {name}?",
    "effect": "What effect does {name} have?",
}


def base_key(doc_id):
    return CHUNK_SUFFIX.sub("", doc_id)


def doc_name(doc):
    meta = doc.get("metadata", {}) or {}
    return meta.get("name") or base_key(doc["id"])


def load_docs():
    with open(DOCS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_clusters(docs, by_id):
    """Return (clusters, owner): owner maps doc_id -> cluster index."""
    name_sets = {}
    for d in docs:
        if d.get("type") not in CROSS_TABLES:
            continue
        n = doc_name(d).strip()
        if n:
            name_sets.setdefault(n.lower(), []).append(d["id"])
    ambiguous = {n for n, ids in name_sets.items() if len(ids) > AMBIG_CAP}

    family = {}
    for d in docs:
        family.setdefault(base_key(d["id"]), []).append(d["id"])

    owner, clusters = {}, []
    for n, ids in name_sets.items():
        if n in ambiguous or not ids:
            continue
        cid = len(clusters)
        clusters.append(sorted(ids))
        for i in ids:
            owner[i] = cid
    for ids in family.values():
        un = [i for i in ids if i not in owner]
        if un:
            cid = len(clusters)
            clusters.append(sorted(un))
            for i in un:
                owner[i] = cid
    return clusters, owner


def cluster_type_count(ids, by_id):
    return Counter(by_id[i]["type"] for i in ids)


def pick_clusters(clusters, by_id, target, cap_share=CAP_SHARE):
    """Select whole clusters: force one rich cluster per interesting query type,
    then fill with a per-table doc budget so no table dominates. Returns cid list."""
    rng = random.Random(SEED)
    order = list(range(len(clusters)))
    rng.shuffle(order)
    order.sort(key=lambda c: -len(clusters[c]))  # richest first (seeded)

    total_types = Counter()
    for ids in clusters:
        total_types.update(by_id[i]["type"] for i in ids)
    tt = sum(total_types.values()) or 1
    cap = {t: max(1, int(target * min(total_types[t] / tt, cap_share))) for t in total_types}

    def has_type(ids, t):
        return any(by_id[i]["type"] == t for i in ids)

    forced = [
        c
        for t in INTERESTING
        for c in (next((x for x in order if has_type(clusters[x], t)), None),)
        if c is not None
    ]
    ordered_forced = []
    seen = set()
    for c in forced:
        if c not in seen:
            seen.add(c)
            ordered_forced.append(c)

    chosen = list(ordered_forced)
    used = Counter()
    for c in chosen:
        used.update(by_id[i]["type"] for i in clusters[c])
    chosen_set = set(chosen)

    def size():
        return sum(len(clusters[c]) for c in chosen)

    for c in order:
        if c in chosen_set:
            continue
        if size() >= target:
            break
        ids = clusters[c]
        nxt = Counter(used)
        ok = True
        for i in ids:
            t = by_id[i]["type"]
            nxt[t] += 1
            if nxt[t] > cap[t]:
                ok = False
                break
        if ok and size() + len(ids) <= target:
            chosen.append(c)
            used = nxt
    return chosen


def build_queries(docs, clusters, chosen_cids, n=N_QUERIES):
    """3-tier queries over chosen clusters, emitted needle -> fill -> paraphrase.

    Tiers share one `used` set (disjoint): needle golds are single docs chosen
    for max same-name distractor pressure; fill golds are whole clusters capped
    at 10; paraphrase pairs target the same capped cluster golds under two
    phrasings. Returns queries[:n]."""
    by_id = {d["id"]: d for d in docs}
    ordered = sorted(chosen_cids, key=lambda c: -len(clusters[c]))
    chosen_ids = [i for c in chosen_cids for i in clusters[c]]

    # name.lower -> doc ids in the chosen set (same-name sibling distractor count)
    siblings_by_name = {}
    for i in chosen_ids:
        siblings_by_name.setdefault(doc_name(by_id[i]).lower(), []).append(i)

    def anchor_of_type(ids, t):
        return next((by_id[i] for i in ids if by_id[i]["type"] == t), by_id[ids[0]])

    def fmt(t, doc, tmpl):
        if t == "wiki":
            title = base_key(doc["id"])
            if title.startswith("wiki_"):
                title = title[len("wiki_") :]
            return tmpl.format(title=title.replace("_", " "))
        return tmpl.format(name=doc_name(doc))

    queries, used = [], set()
    hard_ids = 0

    # --- Tier 1: needle — one specific doc carrying a distinguishing structured
    # fact; the gold is chosen for MAX same-name distractor pressure (not first
    # match), so MRR/hit range stays open. Cap leaves room for multi-gold fill
    # at small n.
    needle_cap = min(HARD_N, len(HARD_TEMPLATES), max(0, n - 2))
    for t, field, tmpl in HARD_TEMPLATES:
        if hard_ids >= needle_cap:
            break
        best = best_score = best_len = -1
        for c in ordered:
            for i in clusters[c]:
                d = by_id[i]
                if d["type"] != t or i in used:
                    continue
                v = (d.get("metadata", {}) or {}).get(field)
                if v is None or v == "":
                    continue
                if str(v).lower() not in (d.get("text", "") or "").lower():
                    continue
                score = len(siblings_by_name.get(doc_name(d).lower(), []))
                vlen = len(str(v))
                if best == -1 or score > best_score or (score == best_score and vlen > best_len):
                    best, best_score, best_len = i, score, vlen
        if best == -1:
            continue
        queries.append(
            {
                "id": f"q_hard_{hard_ids}",
                "text": tmpl.format(name=doc_name(by_id[best])),
                "expected_doc_ids": [best],
            }
        )
        used.add(best)
        hard_ids += 1

    # Tier sizes: paraphrase aims at 10 pairs (20 queries) capped by budget; fill
    # takes the remainder (n - needle - paraphrase) so the corpus hits N_QUERIES
    # regardless of how many needle templates fire (quest/location has no
    # in-corpus doc here, so hard can be 7 or 8). The -2 keeps >=2 fill slots in
    # small-n slices so the multi-gold fill tier is what the slice carries.
    para_pairs = min(10, max(0, (n - hard_ids - 2) // 2))
    fill_target = n - hard_ids - 2 * para_pairs

    # --- Tier 2: fill — whole-cluster golds capped at 10 so Recall@10/NDCG can
    # reach 1.0 (pre-cap 49-gold wiki clusters made the denominator unreachable).
    # Prefer non-interesting clusters (generic text) so interesting clusters stay
    # available for the paraphrase tier; spill into interesting only if needed.
    def fill_text(ids, covered):
        t = covered[0] if covered else None
        doc = anchor_of_type(ids, t) if t else by_id[ids[0]]
        if t and t in FILL_TEMPLATES:
            return fmt(t, doc, FILL_TEMPLATES[t])
        return f"Tell me about {doc_name(doc)}"

    fills = 0
    for prefer_interesting in (False, True):
        for c in ordered:
            if fills >= fill_target:
                break
            ids = clusters[c]
            if any(i in used for i in ids):
                continue
            covered = [t for t in INTERESTING if any(by_id[i]["type"] == t for i in ids)]
            if bool(covered) != prefer_interesting:
                continue
            queries.append(
                {
                    "id": f"q_fill_{fills}",
                    "text": fill_text(ids, covered),
                    "expected_doc_ids": list(ids)[:10],
                }
            )
            used.update(ids)
            fills += 1

    # --- Tier 3: paraphrase — two phrasings per cluster targeting the SAME golds
    # (whole cluster, capped). One cluster per type preferred, then repeat to the
    # pair budget; the a-vs-b MRR variance is the lexical-vs-semantic signal.
    pair_no = 0
    seen_types = set()
    for pass_ in range(2):
        for c in ordered:
            if pair_no >= para_pairs:
                break
            ids = clusters[c]
            if any(i in used for i in ids):
                continue
            covered = [t for t in INTERESTING if any(by_id[i]["type"] == t for i in ids)]
            if not covered:
                continue
            t = covered[0]
            if pass_ == 0 and t in seen_types:
                continue
            doc = anchor_of_type(ids, t)
            for s, tmpl in zip(("a", "b"), PARAPHRASE_TEMPLATES[t], strict=False):
                queries.append(
                    {
                        "id": f"q_para_{pair_no}_{s}",
                        "text": fmt(t, doc, tmpl),
                        "expected_doc_ids": list(ids)[:10],
                    }
                )
            used.update(ids)
            seen_types.add(t)
            pair_no += 1
        if pair_no >= para_pairs:
            break
    return queries[:n]


def fingerprint(corpus):
    rels = [len(q["expected_doc_ids"]) for q in corpus["queries"]]
    type_dist = Counter(d["type"] for d in corpus["docs"])
    blob = json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode()
    return {
        "seed": SEED,
        "n_docs": len(corpus["docs"]),
        "n_queries": len(rels),
        "golds_per_query_mean": round(sum(rels) / len(rels), 2) if rels else 0.0,
        "golds_per_query_min": min(rels) if rels else 0,
        "golds_per_query_max": max(rels) if rels else 0,
        "multi_gold_queries": sum(1 for r in rels if r > 1),
        "hard_queries": sum(1 for q in corpus["queries"] if q["id"].startswith("q_hard_")),
        "paraphrase_queries": sum(1 for q in corpus["queries"] if q["id"].startswith("q_para_")),
        "corpus_type_dist": dict(sorted(type_dist.items())),
        "content_hash": hashlib.sha256(blob).hexdigest()[:12],
    }


def main():
    docs = load_docs()
    sys.stderr.write(f"loaded {len(docs)} docs\n")
    by_id = {d["id"]: d for d in docs}

    clusters, _owner = build_clusters(docs, by_id)
    sys.stderr.write(
        f"{len(clusters)} entity clusters "
        f"(mean {sum(len(c) for c in clusters) / len(clusters):.2f} docs/cluster)\n"
    )

    chosen_cids = pick_clusters(clusters, by_id, TARGET)
    chosen_ids = [i for c in chosen_cids for i in clusters[c]]
    n_types = len({by_id[i]["type"] for i in chosen_ids})
    sys.stderr.write(
        f"selected {len(chosen_cids)} clusters / {len(chosen_ids)} docs ({n_types} tables)\n"
    )

    queries = build_queries(docs, clusters, chosen_cids)
    sys.stderr.write(f"built {len(queries)} queries\n")

    corpus = {
        "docs": [
            {
                "id": i,
                "type": by_id[i]["type"],
                "text": by_id[i].get("text", ""),
                "metadata": by_id[i].get("metadata", {}),
            }
            for i in chosen_ids
        ],
        "queries": queries,
    }
    idset = set(d["id"] for d in corpus["docs"])
    missing = [i for q in corpus["queries"] for i in q["expected_doc_ids"] if i not in idset]
    if missing:
        sys.stderr.write(f"ERROR: {len(missing)} golds not in corpus\n")
        sys.exit(1)
    corpus["fingerprint"] = fingerprint(corpus)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(corpus, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OUT_PATH)
    fp = corpus["fingerprint"]
    n_para = fp.get("paraphrase_queries", 0)
    n_fill = fp["n_queries"] - fp["hard_queries"] - n_para
    sys.stderr.write(
        f"wrote {OUT_PATH}: {fp['n_docs']} docs, {fp['n_queries']} queries "
        f"(hard {fp['hard_queries']}, fill {n_fill}, paraphrase {n_para}), "
        f"mean {fp['golds_per_query_mean']} golds/q "
        f"({fp['multi_gold_queries']}/{fp['n_queries']} multi-gold), "
        f"hash {fp['content_hash']}\n"
    )
    print(
        json.dumps(
            {
                "docs": fp["n_docs"],
                "queries": fp["n_queries"],
                "mean_golds": fp["golds_per_query_mean"],
                "multi_gold": fp["multi_gold_queries"],
                "hard": fp["hard_queries"],
                "paraphrase": n_para,
                "fill": n_fill,
                "hash": fp["content_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
