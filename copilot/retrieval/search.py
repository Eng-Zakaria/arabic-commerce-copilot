"""Hybrid retrieval: BM25 over dialect-aware tokens + dense embeddings, fused with RRF.

Why both: BM25 nails exact names, areas and Arabizi spellings the tokenizer understands;
embeddings catch paraphrases ("حاجة أفطر فيها" -> breakfast offers) that share no tokens.
Reciprocal Rank Fusion needs no score calibration between the two, so it is robust.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..text.arabic import tokenize

Tokenizer = Callable[[str], list[str]]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    rank: int


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75). Small and dependency-free; fine to ~100k documents."""

    def __init__(
        self,
        ids: Sequence[str],
        docs: Sequence[str],
        tokenizer: Tokenizer = tokenize,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.ids, self.tokenizer, self.k1, self.b = list(ids), tokenizer, k1, b
        self.tfs = [Counter(tokenizer(d)) for d in docs]
        self.lens = np.array([sum(tf.values()) for tf in self.tfs], dtype=float)
        self.avgdl = float(self.lens.mean()) if len(self.lens) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.tokenizer(query)
        if not q:
            return []
        scores = np.zeros(len(self.ids))
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
                    s += self.idf[t] * f * (self.k1 + 1) / norm
            scores[i] = s
        order = np.argsort(-scores)[:k]
        return [Hit(self.ids[i], float(scores[i]), r) for r, i in enumerate(order) if scores[i] > 0]


class Embedder:
    """OpenAI-compatible embeddings (Ollama locally, Azure OpenAI in azure mode), disk-cached."""

    QUERY_INSTRUCTION = "Instruct: Given a shopping search query, retrieve matching offers\nQuery: "

    def __init__(self, model: str | None = None, cache_dir: Path | None = None, client=None):
        self.model = model or os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
        self.cache_dir = cache_dir or Path(os.environ.get("EMBED_CACHE", "data/embeddings"))
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from cloudkit import llm_client

            self._client = llm_client()
        return self._client

    def _cache_path(self, texts: Sequence[str]) -> Path:
        h = hashlib.sha256(
            json.dumps([self.model, *texts], ensure_ascii=False).encode()
        ).hexdigest()
        return self.cache_dir / f"{h[:20]}.npy"

    def embed(self, texts: Sequence[str], batch: int = 32) -> np.ndarray:
        path = self._cache_path(texts)
        if path.exists():
            return np.load(path)
        vecs: list[list[float]] = []
        for i in range(0, len(texts), batch):
            resp = self.client.embeddings.create(model=self.model, input=list(texts[i : i + batch]))
            vecs.extend(d.embedding for d in resp.data)
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        return arr

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([self.QUERY_INSTRUCTION + query])[0]


class DenseIndex:
    def __init__(self, ids: Sequence[str], docs: Sequence[str], embedder: Embedder):
        self.ids, self.embedder = list(ids), embedder
        self.matrix = embedder.embed(list(docs))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        sims = self.matrix @ self.embedder.embed_query(query)
        order = np.argsort(-sims)[:k]
        return [Hit(self.ids[i], float(sims[i]), r) for r, i in enumerate(order)]


def rrf(rankings: Sequence[Sequence[Hit]], k: int = 10, c: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum 1 / (c + rank). c=60 per Cormack et al. 2009."""
    fused: dict[str, float] = {}
    for hits in rankings:
        for h in hits:
            fused[h.doc_id] = fused.get(h.doc_id, 0.0) + 1.0 / (c + h.rank + 1)
    order = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
    return [Hit(d, s, r) for r, (d, s) in enumerate(order)]


def dense_query(query: str) -> str:
    """Embedding models read Arabizi poorly; append its Arabic transliteration."""
    from ..text.arabic import arabizi_to_arabic, looks_like_arabizi, normalize

    extra = [
        arabizi_to_arabic(t)
        for t in normalize(query).split()
        if t.isascii() and looks_like_arabizi(t)
    ]
    return f"{query} ({' '.join(extra)})" if extra else query


class HybridRetriever:
    """Modes (used by the eval ablation):
    bm25 | dense | hybrid  - ranking only
    slots                  - hybrid ranking restricted to offers matching parsed slots
    """

    def __init__(self, bm25: BM25, dense: DenseIndex | None = None, depth: int = 50, facets=None):
        self.bm25, self.dense, self.depth = bm25, dense, depth
        self.facets = (
            facets or {}
        )  # doc_id -> {"category":..., "area":..., "city":..., "price":...}

    def _rank(self, query: str, k: int, mode: str) -> list[Hit]:
        if mode == "bm25" or self.dense is None:
            return self.bm25.search(query, k)
        if mode == "dense":
            return self.dense.search(dense_query(query), k)
        return rrf(
            [
                self.bm25.search(query, self.depth),
                self.dense.search(dense_query(query), self.depth),
            ],
            k,
        )

    def search(self, query: str, k: int = 10, mode: str = "slots") -> list[Hit]:
        if mode != "slots" or not self.facets:
            return self._rank(query, k, mode)
        from .query import parse

        slots = parse(query)
        ranked = self._rank(query, len(self.facets), "hybrid" if self.dense else "bm25")
        if slots.empty:
            return ranked[:k]

        def ok(doc_id: str) -> bool:
            f = self.facets[doc_id]
            return (
                (not slots.category or f["category"] == slots.category)
                and (not slots.area or f["area"] == slots.area)
                and (not slots.city or f["city"] == slots.city)
                and (not slots.max_price or f["price"] <= slots.max_price)
            )

        seen = {h.doc_id for h in ranked}
        # offers that match every slot but no ranking signal still qualify (e.g. pure facet queries)
        pool = [h for h in ranked if ok(h.doc_id)] + [
            Hit(d, 0.0, 10**6) for d in self.facets if d not in seen and ok(d)
        ]
        if slots.cheapest:
            pool.sort(key=lambda h: self.facets[h.doc_id]["price"])
        filtered = [Hit(h.doc_id, h.score, r) for r, h in enumerate(pool[:k])]
        return filtered or ranked[:k]  # never return nothing just because a slot was misread
