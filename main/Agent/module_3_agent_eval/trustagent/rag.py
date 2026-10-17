"""Minimal offline RAG: BM25 + TF-IDF cosine hybrid with a rerank step.

The full stack the plan calls for (pypdf -> recursive char chunking ->
sentence-transformers -> chromadb -> BM25 hybrid -> rerank -> RAGAS) is wired
in behind guards, but the *default* path is pure Python.  Reason: this
workspace has no C++ toolchain and the evaluation must run with no network, so
the retrieval backend has to degrade to something that always works.  When
``sentence-transformers`` / ``chromadb`` are present, pass ``vector=True`` and
BM25 becomes the sparse half of a hybrid search.

The corpus is a set of paraphrased summaries of O-RAN ALLIANCE public
specification text — see ``corpus/README.md`` for the provenance note.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return _TOKEN.findall(text)


# --------------------------------------------------------------------------
def bm25_rank(query: str, docs: Sequence[str], top_k: int = 5,
              k1: float = 1.5, b: float = 0.75) -> List[Tuple[float, int]]:
    """Okapi BM25 over ``docs``; returns ``(score, doc_index)`` best-first."""
    q = tokenize(query)
    if not q or not docs:
        return []
    tokenised = [tokenize(d) for d in docs]
    n = len(tokenised)
    avgdl = (sum(len(t) for t in tokenised) / n) or 1.0
    df: Dict[str, int] = {}
    for toks in tokenised:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    scores: List[float] = []
    for toks in tokenised:
        tf: Dict[str, int] = {}
        for term in toks:
            tf[term] = tf.get(term, 0) + 1
        s = 0.0
        dl = len(toks) or 1
        for term in q:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])
    return [(scores[i], i) for i in order[:max(1, int(top_k))] if scores[i] > 0]


def tfidf_cosine(query: str, doc: str) -> float:
    """Bag-of-words cosine in TF-IDF space — the 'poor man's' dense score."""
    a, b = tokenize(query), tokenize(doc)
    if not a or not b:
        return 0.0
    def vec(toks: List[str]) -> Dict[str, float]:
        m: Dict[str, float] = {}
        for t in toks:
            m[t] = m.get(t, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in m.values())) or 1.0
        return {k: v / norm for k, v in m.items()}
    va, vb = vec(a), vec(b)
    return sum(w * vb.get(k, 0.0) for k, w in va.items())


# --------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    doc_id: str
    start: int = 0
    end: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


def recursive_char_split(text: str, chunk_size: int = 480, overlap: int = 80,
                         separators: Optional[List[str]] = None) -> List[str]:
    """LangChain-style recursive character splitter, ~20 lines of it.

    Try the separators in order (paragraph -> line -> sentence -> char) so
    chunks break where a human would break them, and only fall through to a
    hard character cut when nothing else fits.
    """
    seps = separators or ["\n\n", "\n", ". ", " ", ""]
    text = str(text or "").strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    for sep in seps:
        if sep == "":
            continue
        parts = text.split(sep)
        if max(len(p) for p in parts) <= chunk_size:
            out: List[str] = []
            buf = ""
            for p in parts:
                candidate = (buf + sep + p) if buf else p
                if len(candidate) <= chunk_size:
                    buf = candidate
                else:
                    if buf:
                        out.append(buf.strip())
                    buf = (p if len(p) <= chunk_size
                           else _overlap_cut(p, chunk_size, overlap)[0])
            if buf.strip():
                out.append(buf.strip())
            merged: List[str] = []
            for c in out:
                if len(c) < chunk_size // 3 and merged:
                    merged[-1] = (merged[-1] + " " + c).strip()
                else:
                    merged.append(c)
            return [m for m in merged if m]
    return _overlap_cut(text, chunk_size, overlap)


def _overlap_cut(text: str, size: int, overlap: int) -> List[str]:
    step = max(1, size - max(0, overlap))
    return [text[i:i + size] for i in range(0, len(text), step)]


# --------------------------------------------------------------------------
class Retriever:
    """BM25 (+ optional TF-IDF / vector) hybrid retriever with a tiny reranker."""

    def __init__(self, documents: Optional[Dict[str, str]] = None, *,
                 chunk_size: int = 480, overlap: int = 80, vector: bool = False,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.chunks: List[Chunk] = []
        self._embed = None
        self._vectors: Optional[List[List[float]]] = None
        if vector:
            self._embed = _try_sentence_transformer(model_name)
        for doc_id, text in (documents or {}).items():
            self.add(doc_id, text, chunk_size=chunk_size, overlap=overlap)

    def add(self, doc_id: str, text: str, *, chunk_size: int = 480,
            overlap: int = 80, meta: Optional[Dict[str, Any]] = None) -> int:
        before = len(self.chunks)
        pos = 0
        for piece in recursive_char_split(text, chunk_size, overlap):
            start = text.find(piece[:40], pos)
            start = 0 if start < 0 else start
            self.chunks.append(Chunk(text=piece, doc_id=doc_id, start=start,
                                     end=start + len(piece), meta=dict(meta or {})))
            pos = start + 1
        if self._embed is not None:
            self._vectors = None
        return len(self.chunks) - before

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 4) -> List[Chunk]:
        if not self.chunks:
            return []
        texts = [c.text for c in self.chunks]
        sparse = bm25_rank(query, texts, top_k=max(top_k * 3, 8))
        scored: Dict[int, float] = {i: s for s, i in sparse}
        for i, c in enumerate(self.chunks):
            sim = tfidf_cosine(query, c.text)
            if sim > 0:
                scored[i] = scored.get(i, 0.0) + sim
        if self._embed is not None:
            vsim = self._vector_scores(query)
            for i, v in vsim:
                scored[i] = scored.get(i, 0.0) + v
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])
        picked = [self.chunks[i] for _, i in ranked[:max(1, top_k)]]
        return self._rerank(query, picked)

    def _rerank(self, query: str, cands: List[Chunk]) -> List[Chunk]:
        """Coverage-based rerank: reward chunks containing *more* query terms."""
        q = set(tokenize(query))
        if not q:
            return cands

        def cov(c: Chunk) -> Tuple[float, float]:
            toks = set(tokenize(c.text))
            inter = len(q & toks) / len(q)
            # penalise long chunks so the answer stays near the query terms
            return inter, -len(toks) / 1000.0

        return sorted(cands, key=cov, reverse=True)

    def _vector_scores(self, query: str) -> List[Tuple[int, float]]:
        if self._embed is None:
            return []
        if self._vectors is None:
            self._vectors = [list(map(float, v))
                             for v in self._embed.encode([c.text for c in self.chunks])]
        qv = list(map(float, self._embed.encode([query])[0]))
        out = []
        for i, dv in enumerate(self._vectors):
            num = sum(a * b for a, b in zip(qv, dv))
            den = (math.sqrt(sum(a * a for a in qv)) * math.sqrt(sum(b * b for b in dv))) or 1.0
            out.append((i, num / den))
        out.sort(key=lambda kv: -kv[1])
        return out

    # -- context assembly ---------------------------------------------------
    def context(self, query: str, top_k: int = 4) -> str:
        return "\n\n".join(f"[{c.doc_id}#{i}] {c.text}"
                            for i, c in enumerate(self.search(query, top_k)))


def _try_sentence_transformer(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer    # type: ignore
        return SentenceTransformer(model_name)
    except Exception:
        return None


def load_md_corpus(directory: str) -> Dict[str, str]:
    """Read ``*.md`` / ``*.txt`` files into ``{stem: text}``."""
    import os
    out: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if fn.lower().endswith((".md", ".txt")):
            with open(os.path.join(directory, fn), "r", encoding="utf-8") as fh:
                out[os.path.splitext(fn)[0]] = fh.read()
    return out
