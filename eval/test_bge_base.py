"""
eval/test_bge_base.py
Three-level test for bge-base-en-v1.5 vs bge-small-en-v1.5.

Level 1 -- Embedding sanity   : dim check, norm=1.0, cosine on known pairs
Level 2 -- Retrieval spot-check: 5 financial queries, show top chunks
Level 3 -- Side-by-side compare: same queries on 'papers' vs 'papers_base'

Run:
    python eval/test_bge_base.py            # all three levels
    python eval/test_bge_base.py --level 1  # just sanity
    python eval/test_bge_base.py --level 2  # just retrieval
    python eval/test_bge_base.py --level 3  # just comparison
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

TEST_QUERIES = [
    "What was Apple total revenue for fiscal year 2025?",
    "What are the main risk factors related to supply chain?",
    "What is the gross margin trend over recent quarters?",
    "Explain the company share repurchase program.",
    "What were the EPS diluted figures reported?",
]

SIMILAR_PAIRS = [
    ("net income", "profit after tax"),        # expect HIGH similarity
    ("revenue growth", "supply chain risk"),   # expect LOW similarity
    ("earnings per share", "EPS diluted"),     # expect VERY HIGH similarity
]


# ---------------------------------------------------------------------------
# LEVEL 1: Embedding sanity
# ---------------------------------------------------------------------------

def level1_sanity():
    print("\n" + "=" * 60)
    print("  LEVEL 1 -- Embedding Sanity Check")
    print("=" * 60)

    from indexing.embedder import Embedder
    emb = Embedder()  # picks up bge-base default

    vec = emb.embed_query("What is total revenue?")
    dim = len(vec)
    print(f"\n  Vector dimension : {dim}  (expected 768)  --> {'PASS' if dim == 768 else 'FAIL'}")

    norm = float(np.linalg.norm(vec))
    print(f"  L2 norm          : {norm:.6f}  --> {'normalised OK' if abs(norm - 1.0) < 1e-5 else 'NOT normalised'}")

    print("\n  Cosine similarity on known pairs (vectors are normalised so dot = cosine):")
    for a, b in SIMILAR_PAIRS:
        va = emb.embed_query(a)
        vb = emb.embed_query(b)
        sim = float(np.dot(va, vb))
        print(f"    {repr(a)} vs {repr(b)}: {sim:.4f}")


# ---------------------------------------------------------------------------
# LEVEL 2: Retrieval spot-check
# ---------------------------------------------------------------------------

def level2_retrieval(collection="papers_base"):
    print("\n" + "=" * 60)
    print(f"  LEVEL 2 -- Retrieval Spot-Check  [{collection}]")
    print("=" * 60)

    from retrieval.retriever import Retriever

    try:
        r = Retriever(collection_name=collection)
        total = r.collection.count()
    except Exception as e:
        print(f"\n  Could not load {repr(collection)}: {e}")
        return

    if total == 0:
        print(f"\n  Collection {repr(collection)} is empty -- re-index first.")
        return

    print(f"\n  {total} chunks in {repr(collection)}\n")

    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"  [{i}] {q}")
        results = r.retrieve(q, top_k=2)
        if not results:
            print("       --> No results\n")
            continue
        for rank, res in enumerate(results, 1):
            m = res["metadata"]
            score = res["similarity_score"]
            page = m.get("page_number", "?")
            doc = m.get("paper_id", "?")
            preview = res["text"][:90].strip()
            print(f"       #{rank}  score={score:.4f}  page={page}  doc={doc}")
            print(f"           \"{preview}...\"")
        print()


# ---------------------------------------------------------------------------
# LEVEL 3: Side-by-side small vs base
# ---------------------------------------------------------------------------

def level3_compare():
    print("\n" + "=" * 60)
    print("  LEVEL 3 -- Side-by-Side: bge-small vs bge-base")
    print("=" * 60)

    from indexing.embedder import Embedder
    import chromadb

    client = chromadb.PersistentClient(path="data/chroma_db")
    configs = [
        ("papers",      "BAAI/bge-small-en-v1.5"),
        ("papers_base", "BAAI/bge-base-en-v1.5"),
    ]
    available = {c.name for c in client.list_collections()}

    # Find paper_ids present in BOTH collections so comparison is fair
    id_sets = {}
    for col_name, _ in configs:
        if col_name not in available:
            continue
        col = client.get_collection(col_name)
        metas = col.get(include=["metadatas"])["metadatas"]
        id_sets[col_name] = {m["paper_id"] for m in metas if m and "paper_id" in m}

    if len(id_sets) < 2:
        print("  Need both collections to compare. Skipping.")
        return

    common_ids = sorted(id_sets["papers"] & id_sets["papers_base"])
    only_small = sorted(id_sets["papers"] - id_sets["papers_base"])
    only_base  = sorted(id_sets["papers_base"] - id_sets["papers"])

    print(f"\n  Common documents ({len(common_ids)}): {common_ids}")
    if only_small:
        print(f"  Only in papers (excluded): {only_small}")
    if only_base:
        print(f"  Only in papers_base (excluded): {only_base}")

    if not common_ids:
        print("\n  No common documents found -- index the same files into both collections first.")
        return

    wins = {"papers": 0, "papers_base": 0, "tie": 0}

    for query in TEST_QUERIES:
        print(f"\n  Q: {query}")
        scores = {}

        for col_name, model_name in configs:
            if col_name not in available:
                print(f"    [{col_name}] not found -- skipping")
                continue
            col = client.get_collection(col_name)

            emb = Embedder(model_name=model_name)
            qvec = emb.embed_query(query).tolist()

            # Restrict to common paper_ids only
            where = {"paper_id": {"$in": common_ids}} if len(common_ids) > 1 else {"paper_id": common_ids[0]}
            res = col.query(
                query_embeddings=[qvec],
                n_results=1,
                where=where,
                include=["documents", "distances"],
            )
            score = round(1.0 - res["distances"][0][0], 4) if res["distances"][0] else 0
            text = res["documents"][0][0][:80] if res["documents"][0] else "--"
            scores[col_name] = score
            label = "small" if "small" in model_name else "base "
            print(f"    [{label}] score={score:.4f}  \"{text}...\"")

        if "papers" in scores and "papers_base" in scores:
            s = scores["papers"]
            b = scores["papers_base"]
            if b > s + 0.005:
                print(f"    --> base  wins  (+{b - s:.4f})")
                wins["papers_base"] += 1
            elif s > b + 0.005:
                print(f"    --> small wins  (+{s - b:.4f})")
                wins["papers"] += 1
            else:
                print(f"    --> tie  (diff={abs(b - s):.4f})")
                wins["tie"] += 1

    print("\n" + "-" * 60)
    print(
        f"  RESULT  bge-base wins: {wins['papers_base']}  |  "
        f"bge-small wins: {wins['papers']}  |  ties: {wins['tie']}"
    )
    print(f"  (Compared only on {len(common_ids)} common documents)")
    print("-" * 60)



# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test bge-base vs bge-small")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=None,
                        help="Which level to run (default: all)")
    parser.add_argument("--collection", type=str, default="papers_base",
                        help="Collection for level 2 (default: papers_base)")
    args = parser.parse_args()

    levels = [args.level] if args.level else [1, 2, 3]

    if 1 in levels:
        level1_sanity()
    if 2 in levels:
        level2_retrieval(args.collection)
    if 3 in levels:
        level3_compare()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
