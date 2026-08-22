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
- A "needle" (hard) tier of HARD_N single-gold queries: the gold is one specific
  doc carrying a distinguishing structured fact (skill_level_req / value / skill
  / location), with same-name siblings as in-corpus distractors, so MRR/hit@k
  keep dynamic range instead of saturating at the entity-cluster ceiling.
- The corpus carries a fingerprint (gold stats, type distribution, content
  hash) so a changed/regenerated run is detectable.
- The corpus carries a fingerprint (gold stats, type distribution, content
  hash) so a changed/regenerated run is detectable.

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

TARGET = 500            # corpus docs (golds are by-construction present)
N_QUERIES = 24          # evaluation queries
HARD_N = 6              # of the queries: single-gold "needle" tier (MRR range)
SEED = 42
CAP_SHARE = 0.25        # no single table's docs >25% of the corpus sample
AMBIG_CAP = 30          # name with more entity-table matches: too generic to group
INTERESTING = ("recipe", "item", "wiki", "ability", "quest", "skillprofile", "effect")
# Hard-query templates: (type, metadata field, question). The gold is the ONE
# doc of that type whose text contains the field value; same-name siblings in
# the corpus act as distractors, restoring MRR/hit-range that the multi-gold
# entity clusters saturate. Field must appear in the doc text (verified) so the
# gold is retrievable.
HARD_TEMPLATES = [
    ("recipe", "skill_level_req", "What skill level is required to craft {name}?"),
    ("item", "value", "What is the buy value of the item {name}?"),
    ("ability", "skill", "Which skill does the {name} ability belong to?"),
    ("quest", "location", "Where does the quest {name} take place?"),
    ("ability", "damage_type", "What damage type is the {name} ability?"),
    ("recipe", "skill", "What skill is used to craft {name}?"),
]
# Tables whose same-name records are plausibly the same entity (cross-source link).
CROSS_TABLES = {
    "item", "recipe", "quest", "ability", "effect", "skill",
    "skillprofile", "leveling", "npc", "wiki", "itemuse", "abilitykeyword",
}

CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")

TYPE_TEMPLATES = {
    "recipe": "What materials are needed to craft {name}?",
    "item": "What does the item {name} do?",
    "wiki": "What does the wiki say about {title}?",
    "ability": "What does the ability {name} do?",
    "quest": "How do I start the quest {name}?",
    "skillprofile": "Tell me everything about the {name} skill",
    "effect": "What effect does {name} have?",
}
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
    order.sort(key=lambda c: -len(clusters[c]))      # richest first (seeded)

    total_types = Counter()
    for ids in clusters:
        total_types.update(by_id[i]["type"] for i in ids)
    tt = sum(total_types.values()) or 1
    cap = {t: max(1, int(target * min(total_types[t] / tt, cap_share)))
           for t in total_types}

    def has_type(ids, t):
        return any(by_id[i]["type"] == t for i in ids)

    forced = [c for t in INTERESTING
              for c in (next((x for x in order if has_type(clusters[x], t)), None),)
              if c is not None]
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
    """Queries targeting chosen clusters: golds = the whole cluster (all in corpus)."""
    by_id = {d["id"]: d for d in docs}
    ordered = sorted(chosen_cids, key=lambda c: -len(clusters[c]))

    def anchor_of_type(ids, t):
        return next((by_id[i] for i in ids if by_id[i]["type"] == t), by_id[ids[0]])

    def q_text(t, doc):
        if t == "wiki":
            title = base_key(doc["id"])
            if title.startswith("wiki_"):
                title = title[len("wiki_"):]
            return TYPE_TEMPLATES[t].format(title=title.replace("_", " "))
        return TYPE_TEMPLATES[t].format(name=doc_name(doc))

    queries, used = [], set()
    # Hard "needle" tier: single-gold queries whose gold is a specific doc that
    # carries a distinguishing structured fact (skill_level_req / value / skill /
    # location ...), while same-name siblings are in-corpus distractors. This
    # restores MRR/hit@k dynamic range the multi-gold clusters saturate.
    hard_added = 0
    for t, field, tmpl in HARD_TEMPLATES:
        if hard_added >= HARD_N or len(queries) >= n:
            break
        gold = None
        for c in ordered:
            for i in clusters[c]:
                d = by_id[i]
                if d["type"] != t or i in used:
                    continue
                v = (d.get("metadata", {}) or {}).get(field)
                if v is None or v == "":
                    continue
                if str(v).lower() in (d.get("text", "") or "").lower():
                    gold = i
                    break
            if gold is not None:
                break
        if gold is None:
            continue
        queries.append({"id": f"q_hard_{hard_added}",
                        "text": tmpl.format(name=doc_name(by_id[gold])),
                        "expected_doc_ids": [gold]})
        used.add(gold)
        hard_added += 1
    # fills from remaining chosen clusters
    for c in ordered:
        if len(queries) >= n:
            break
        ids = clusters[c]
        if any(i in used for i in ids):
            continue
        # pick the cluster's dominant interesting type for the query text, else generic
        covered = [t for t in INTERESTING if any(by_id[i]["type"] == t for i in ids)]
        t = covered[0] if covered else None
        doc = anchor_of_type(ids, t) if t else by_id[ids[0]]
        if t == "wiki":
            text = q_text("wiki", doc)
        elif t:
            text = FILL_TEMPLATES[t].format(name=doc_name(doc))
        else:
            text = "Tell me about {name}".format(name=doc_name(doc))
        queries.append({"id": f"q_fill_{len(queries)}",
                        "text": text, "expected_doc_ids": list(ids)})
        used.update(ids)
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
        "multi_gold_queries": sum(1 for r in rels if r > 1),
        "hard_queries": sum(1 for q in corpus["queries"] if q["id"].startswith("q_hard_")),
        "hard_queries": sum(1 for q in corpus["queries"] if q["id"].startswith("q_hard_")),
        "corpus_type_dist": dict(sorted(type_dist.items())),
        "content_hash": hashlib.sha256(blob).hexdigest()[:12],
    }


def main():
    docs = load_docs()
    sys.stderr.write(f"loaded {len(docs)} docs\n")
    by_id = {d["id"]: d for d in docs}

    clusters, _owner = build_clusters(docs, by_id)
    sys.stderr.write(f"{len(clusters)} entity clusters "
                     f"(mean {sum(len(c) for c in clusters) / len(clusters):.2f} docs/cluster)\n")

    chosen_cids = pick_clusters(clusters, by_id, TARGET)
    chosen_ids = [i for c in chosen_cids for i in clusters[c]]
    n_types = len({by_id[i]["type"] for i in chosen_ids})
    sys.stderr.write(f"selected {len(chosen_cids)} clusters / {len(chosen_ids)} docs "
                     f"({n_types} tables)\n")

    queries = build_queries(docs, clusters, chosen_cids)
    sys.stderr.write(f"built {len(queries)} queries\n")

    corpus = {
        "docs": [{"id": i, "type": by_id[i]["type"], "text": by_id[i].get("text", ""),
                  "metadata": by_id[i].get("metadata", {})} for i in chosen_ids],
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
    sys.stderr.write(f"wrote {OUT_PATH}: {fp['n_docs']} docs, {fp['n_queries']} queries, "
                     f"mean {fp['golds_per_query_mean']} golds/q "
                     f"({fp['multi_gold_queries']}/{fp['n_queries']} multi-gold), "
                     f"hash {fp['content_hash']}\n")
    print(json.dumps({"docs": fp["n_docs"], "queries": fp["n_queries"],
                      "mean_golds": fp["golds_per_query_mean"],
                      "multi_gold": fp["multi_gold_queries"],
                      "hash": fp["content_hash"]}))


if __name__ == "__main__":
    main()