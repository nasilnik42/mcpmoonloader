from __future__ import annotations
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import requests
from mcp.server.fastmcp import FastMCP

# ===== Settings =====
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
CHROMA_DIR = OUT_DIR / "chroma_db"
CHUNKS_JSONL = OUT_DIR / "chunks.jsonl"
PAGES_JSONL = OUT_DIR / "pages.jsonl"
COLLECTION_NAME = "blast_moonloader_lua"
GEMINI_API_KEY_FALLBACK = "кто прочитал тот гомосексуал"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-2")
OUTPUT_DIMENSIONALITY = int(os.getenv("GEMINI_OUTPUT_DIMENSIONALITY", "3072"))

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_./:-]+")

mcp = FastMCP(
    "moonloader-rag",
    instructions=(
        "Use this MCP server whenever you need MoonLoader Lua, SA:MP, SAMPFUNCS, "
        "BlastHack Wiki API documentation, function signatures, examples, events, "
        "render API, RakNet/SAMP events, or globals. Prefer moonloader_context for coding tasks "
        "and moonloader_search for targeted lookups."
    ),
)

_collection = None
_chunks: Optional[List[Dict[str, Any]]] = None
_pages: Optional[List[Dict[str, Any]]] = None
_chunks_by_name: Optional[Dict[str, List[Dict[str, Any]]]] = None
_doc_freq: Optional[Counter] = None
_avg_len: float = 1.0


def _api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY_FALLBACK


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunks() -> List[Dict[str, Any]]:
    global _chunks, _chunks_by_name, _doc_freq, _avg_len
    if _chunks is None:
        _chunks = _load_jsonl(CHUNKS_JSONL)
        _chunks_by_name = defaultdict(list)
        df = Counter()
        total_len = 0
        for d in _chunks:
            meta = d.get("metadata", {})
            keys = {
                str(meta.get("name", "")).lower(),
                str(meta.get("title", "")).lower(),
                str(meta.get("path", "")).lower(),
                str(meta.get("indexed_path", "")).lower(),
            }
            for k in keys:
                if k:
                    _chunks_by_name[k].append(d)
            toks = set(_tokens(d.get("text", "") + " " + json.dumps(meta, ensure_ascii=False)))
            total_len += len(toks)
            df.update(toks)
        _doc_freq = df
        _avg_len = max(1.0, total_len / max(1, len(_chunks)))
    return _chunks


def pages() -> List[Dict[str, Any]]:
    global _pages
    if _pages is None:
        _pages = _load_jsonl(PAGES_JSONL)
    return _pages


def collection():
    global _collection
    if _collection is None:
        _collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)
    return _collection


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def _query_embedding(query: str) -> List[float]:
    text = f"task: search result | query: {query}"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"
    body = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": OUTPUT_DIMENSIONALITY,
    }
    r = requests.post(
        url,
        headers={"Content-Type": "application/json", "x-goog-api-key": _api_key()},
        json=body,
        timeout=60,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini embedding error {r.status_code}: {r.text[:700]}")
    return r.json()["embedding"]["values"]


def _semantic_search(query: str, limit: int) -> List[Dict[str, Any]]:
    emb = _query_embedding(query)
    res = collection().query(
        query_embeddings=[emb],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        # Scale semantic score high enough so hybrid ranking prefers the vector result order,
        # while lexical matches still help when Gemini embedding is unavailable or exact terms matter.
        out.append({"text": doc, "metadata": meta, "distance": float(dist), "semantic_score": 100.0 / (1.0 + float(dist))})
    return out


def _lexical_search(query: str, limit: int) -> List[Dict[str, Any]]:
    docs = chunks()
    qtokens = _tokens(query)
    if not qtokens:
        return []
    df = _doc_freq or Counter()
    N = max(1, len(docs))
    scores = []
    qset = set(qtokens)
    for d in docs:
        meta = d.get("metadata", {})
        hay = " ".join([
            d.get("text", ""),
            str(meta.get("name", "")),
            str(meta.get("title", "")),
            str(meta.get("path", "")),
        ])
        toks = _tokens(hay)
        if not toks:
            continue
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        # BM25-ish scoring
        for t in qset:
            if t not in tf:
                continue
            idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            freq = tf[t]
            score += idf * (freq * 2.2) / (freq + 1.2 * (1 - 0.75 + 0.75 * dl / _avg_len))
        # Exact API/function boosts
        name = str(meta.get("name", "")).lower()
        title = str(meta.get("title", "")).lower()
        qlower = query.lower()
        if name and name in qlower:
            score += 10.0
        if title and title in qlower:
            score += 8.0
        if score > 0:
            scores.append((score, d))
    scores.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, d in scores[:limit]:
        out.append({"text": d.get("text", ""), "metadata": d.get("metadata", {}), "lexical_score": float(score)})
    return out


def _dedupe_rank(results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for r in results:
        m = r.get("metadata", {})
        key = f"{m.get('path')}#{m.get('chunk_index', 0)}"
        current = best.get(key)
        score = r.get("semantic_score", 0.0) + r.get("lexical_score", 0.0)
        r["score"] = score
        if current is None or score > current.get("score", 0):
            best[key] = r
    ranked = sorted(best.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked if limit <= 0 else ranked[:limit]


def _format_sources(results: List[Dict[str, Any]], max_chars_per_source: int, include_content: bool) -> List[Dict[str, Any]]:
    formatted = []
    for i, r in enumerate(results, 1):
        meta = dict(r.get("metadata", {}))
        text = r.get("text", "")
        if include_content:
            if max_chars_per_source and max_chars_per_source > 0:
                text_out = text[:max_chars_per_source]
            else:
                text_out = text
        else:
            text_out = ""
        formatted.append({
            "rank": i,
            "score": r.get("score"),
            "distance": r.get("distance"),
            "lexical_score": r.get("lexical_score"),
            "title": meta.get("title"),
            "name": meta.get("name"),
            "kind": meta.get("kind"),
            "url": meta.get("url"),
            "path": meta.get("path"),
            "chunk_index": meta.get("chunk_index"),
            "chunk_total": meta.get("chunk_total"),
            "content": text_out,
        })
    return formatted


@mcp.tool()
def moonloader_stats() -> Dict[str, Any]:
    """Return stats for the local BlastHack Wiki MoonLoader Lua RAG database."""
    ch = chunks()
    pg = pages()
    return {
        "collection": COLLECTION_NAME,
        "chroma_dir": str(CHROMA_DIR),
        "chroma_count": collection().count(),
        "chunks_jsonl_count": len(ch),
        "pages_jsonl_count": len(pg),
        "embedding_model": EMBED_MODEL,
        "output_dimensionality": OUTPUT_DIMENSIONALITY,
        "supports_unlimited_sources": True,
        "note": "Use moonloader_search(limit=0, max_chars_per_source=0) to return all ranked sources/content. Beware client context limits.",
    }


@mcp.tool()
def moonloader_search(
    query: str,
    limit: int = 20,
    mode: str = "hybrid",
    include_content: bool = True,
    max_chars_per_source: int = 4000,
) -> Dict[str, Any]:
    """Search BlastHack Wiki MoonLoader Lua docs.

    Args:
        query: Natural language question, API name, event name, or coding task.
        limit: Max number of sources. Use 0 or negative for unlimited/all ranked sources.
        mode: "hybrid", "vector", or "text". Hybrid uses Gemini Embedding 2 + lexical fallback.
        include_content: Include source text in results.
        max_chars_per_source: Truncate each source content. Use 0 or negative for full source text.

    Returns unlimited sources when limit <= 0, subject only to MCP/client message limits.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "query is empty", "sources": []}
    all_count = len(chunks())
    n = all_count if limit <= 0 else min(max(limit, 1), all_count)
    mode = (mode or "hybrid").lower()

    results: List[Dict[str, Any]] = []
    vector_error = None
    if mode in {"hybrid", "vector", "semantic"}:
        try:
            results.extend(_semantic_search(query, n))
        except Exception as e:
            vector_error = str(e)
            if mode in {"vector", "semantic"}:
                return {"error": vector_error, "sources": [], "fallback_available": "Use mode='text' or mode='hybrid'."}
    if mode in {"hybrid", "text", "lexical"}:
        results.extend(_lexical_search(query, n))

    ranked = _dedupe_rank(results, n)
    return {
        "query": query,
        "mode": mode,
        "limit_requested": limit,
        "returned": len(ranked),
        "vector_error": vector_error,
        "sources": _format_sources(ranked, max_chars_per_source, include_content),
    }


@mcp.tool()
def moonloader_get_doc(name_or_path: str, max_chars_per_chunk: int = 0) -> Dict[str, Any]:
    """Get all chunks for an exact MoonLoader API/page by function/event/global name, title, path, or URL.

    Examples: "sampRegisterChatCommand", "onSendRpc", "renderDrawBox", "moonloader/lua/renderDrawBox".
    max_chars_per_chunk <= 0 returns full chunk text.
    """
    q = (name_or_path or "").strip()
    if not q:
        return {"error": "name_or_path is empty", "chunks": []}
    qnorm = q.lower().strip("/")
    qnorm = qnorm.replace("https://wiki.blast.hk/", "")
    # Try exact keys first.
    by_name = _chunks_by_name if _chunks_by_name is not None else None
    chunks()  # initialize by_name
    by_name = _chunks_by_name or {}
    candidates = []
    for key, vals in by_name.items():
        if qnorm == key.strip("/") or qnorm.endswith("/" + key.strip("/")):
            candidates.extend(vals)
    # Fallback contains match by name/title/path.
    if not candidates:
        for d in chunks():
            meta = d.get("metadata", {})
            hay = " ".join(str(meta.get(k, "")) for k in ["name", "title", "path", "indexed_path", "url"]).lower()
            if qnorm in hay:
                candidates.append(d)
    # Sort chunks in page order.
    candidates.sort(key=lambda d: (d.get("metadata", {}).get("path", ""), int(d.get("metadata", {}).get("chunk_index", 0))))
    out = []
    for d in candidates:
        meta = d.get("metadata", {})
        text = d.get("text", "")
        if max_chars_per_chunk and max_chars_per_chunk > 0:
            text = text[:max_chars_per_chunk]
        out.append({
            "title": meta.get("title"),
            "name": meta.get("name"),
            "kind": meta.get("kind"),
            "url": meta.get("url"),
            "path": meta.get("path"),
            "chunk_index": meta.get("chunk_index"),
            "chunk_total": meta.get("chunk_total"),
            "content": text,
        })
    return {"query": q, "returned_chunks": len(out), "chunks": out}


@mcp.tool()
def moonloader_list_api(filter_text: str = "", limit: int = 0, offset: int = 0) -> Dict[str, Any]:
    """List indexed MoonLoader API/docs pages.

    Args:
        filter_text: Optional substring filter over name/title/path.
        limit: Max rows; 0 or negative means unlimited/all.
        offset: Pagination offset.
    """
    seen = {}
    for d in chunks():
        m = d.get("metadata", {})
        path = m.get("path")
        if not path or path in seen:
            continue
        seen[path] = {
            "name": m.get("name"),
            "title": m.get("title"),
            "kind": m.get("kind"),
            "url": m.get("url"),
            "path": path,
            "chunk_total": m.get("chunk_total"),
        }
    rows = sorted(seen.values(), key=lambda x: (x.get("kind") or "", x.get("name") or ""))
    ft = (filter_text or "").lower().strip()
    if ft:
        rows = [r for r in rows if ft in json.dumps(r, ensure_ascii=False).lower()]
    total = len(rows)
    offset = max(0, int(offset or 0))
    rows = rows[offset:]
    if limit and limit > 0:
        rows = rows[:limit]
    return {"total_matching": total, "offset": offset, "returned": len(rows), "items": rows}


@mcp.tool()
def moonloader_context(query: str, max_sources: int = 40, max_total_chars: int = 60000) -> Dict[str, Any]:
    """Return a consolidated context block for coding agents.

    Use this before writing MoonLoader Lua code. Set max_sources=0 for all ranked sources.
    Set max_total_chars=0 for unlimited text (client context limits may still apply).
    """
    search = moonloader_search(
        query=query,
        limit=max_sources,
        mode="hybrid",
        include_content=True,
        max_chars_per_source=0,
    )
    sources = search.get("sources", [])
    parts = []
    total = 0
    kept = []
    for s in sources:
        block = (
            f"[Source {s['rank']}] {s.get('title')} ({s.get('name')})\n"
            f"URL: {s.get('url')}\n"
            f"Path: {s.get('path')} chunk {s.get('chunk_index')}/{s.get('chunk_total')}\n"
            f"Content:\n{s.get('content', '')}\n"
        )
        if max_total_chars and max_total_chars > 0 and total + len(block) > max_total_chars:
            break
        parts.append(block)
        total += len(block)
        kept.append({k: s.get(k) for k in ["rank", "title", "name", "url", "path", "chunk_index", "chunk_total", "distance"]})
    header = (
        "MoonLoader Lua / BlastHack Wiki RAG context. Use these sources as primary truth.\n"
        "If writing code, respect requirements like SA:MP, SAMPFUNCS, lib.samp.events, wait(), main(), and return values.\n"
    )
    return {
        "query": query,
        "sources_returned_by_search": len(sources),
        "sources_in_context": len(kept),
        "context_chars": len(header) + sum(len(p) for p in parts),
        "context": header + "\n---\n" + "\n---\n".join(parts),
        "sources": kept,
        "vector_error": search.get("vector_error"),
    }


if __name__ == "__main__":
    mcp.run("stdio")
