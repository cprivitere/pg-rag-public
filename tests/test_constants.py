"""Asserts versioned build/pipeline invariants by importing pgrag.config and
build constants and parsing source: embedding dim, context budget, TOP_K,
chroma path/collection names, embed/upsert batch sizes, and pipeline call order."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_v23_embedding_dim_in_config():
    from pgrag.config import EMBEDDING_DIM

    assert isinstance(EMBEDDING_DIM, int) and EMBEDDING_DIM > 0


def test_context_budget_in_config():
    from pgrag.config import CONTEXT_BUDGET

    assert isinstance(CONTEXT_BUDGET, int) and CONTEXT_BUDGET > 0


def test_general_top_k_40():
    """General-path TOP_K defaults to 40 (2x context) so the thinking-on
    model has more retrieved material to reason over. Contract change from
    V37's 20."""
    from scripts.pg_rag import Pipe

    assert Pipe.Valves().TOP_K == 40, "general TOP_K must default to 40"


def test_v2_chroma_path_consistent():
    import inspect

    import pgrag.rag.retriever
    from pgrag.vectorstore.build_index import build_index

    build_src = inspect.getsource(build_index)
    retrieve_src = inspect.getsource(pgrag.rag.retriever)
    assert 'path="data/chroma"' in build_src, "V2: build_index must use path=data/chroma"
    assert 'path="data/chroma"' in retrieve_src, "V2: retriever must use path=data/chroma"


def test_v5_embed_batch_size():
    from pgrag.vectorstore.build_index import EMBED_BATCH_SIZE

    assert EMBED_BATCH_SIZE <= 10000, "V5: EMBED_BATCH_SIZE must be ≤ 10000"


def test_v5_upsert_batch_size():
    from pgrag.vectorstore.build_index import BATCH_SIZE

    assert BATCH_SIZE <= 5000, "V5: BATCH_SIZE must be ≤ 5000"


def test_v8_collection_name_consistent():
    import inspect

    import pgrag.rag.retriever
    from pgrag.vectorstore.build_index import build_index

    build_src = inspect.getsource(build_index)
    retrieve_src = inspect.getsource(pgrag.rag.retriever)
    assert 'name="project_gorgon"' in build_src, "V8: build_index must use project_gorgon"
    assert 'name="project_gorgon"' in retrieve_src, "V8: retriever must use project_gorgon"


def test_v1_pipeline_sequence():
    main_py = ROOT / "src/pgrag/build.py"
    tree = ast.parse(main_py.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.append(node.func.id)
    idx_load = next(i for i, c in enumerate(calls) if c == "load_database")
    idx_wiki = next(i for i, c in enumerate(calls) if c == "load_wiki")
    idx_build = next(i for i, c in enumerate(calls) if c == "build_documents")
    assert idx_load < idx_wiki < idx_build, "V1: load_database → load_wiki → build_documents"
