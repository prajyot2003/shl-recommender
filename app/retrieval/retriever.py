"""Retrieval over the catalog.

Design choice: **lexical-first hybrid**. In the traces, the signal that decides
a shortlist is overwhelmingly *named-skill overlap* ("Java" -> Core Java test,
"Docker" -> Docker test). Lexical methods (BM25 + fuzzy token match) nail that
directly, need no model download, and add no per-query latency — which matters
under the 30s call cap and cold-start hosting.

A semantic (embedding) path is available and *fused in when present*, to catch
vocabulary gaps ("plant operator" -> "safety & dependability"). It is optional:
if sentence-transformers / a local model is unavailable the retriever still works
on the lexical path alone. This keeps the MVP runnable everywhere and lets a
reviewer turn semantic on with one env flag.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..catalog.models import Assessment

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Scored:
    assessment: Assessment
    score: float


class Retriever:
    """Ranks catalog assessments against a free-text query."""

    def __init__(self, assessments: list[Assessment], use_semantic: bool = False):
        self._items = assessments
        self._docs = [a.search_document() for a in assessments]
        self._tokenized = [_tokenize(d) for d in self._docs]
        self._bm25 = self._build_bm25()
        self._semantic = _SemanticIndex.try_build(assessments) if use_semantic else None
        if use_semantic and self._semantic is None:
            logger.info("Semantic retrieval requested but unavailable; using lexical only.")

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi

            return BM25Okapi(self._tokenized)
        except Exception as e:
            logger.warning("BM25 unavailable (%s); using token-overlap fallback.", e)
            return None

    # -- scoring components -------------------------------------------------
    def _bm25_scores(self, q_tokens: list[str]) -> list[float]:
        if self._bm25 is not None:
            return list(self._bm25.get_scores(q_tokens))
        # Fallback: simple token-overlap count so retrieval never dies.
        qset = set(q_tokens)
        return [float(len(qset & set(doc))) for doc in self._tokenized]

    def _fuzzy_boost(self, query: str) -> list[float]:
        """Reward near-exact name matches; e.g. 'opq' -> 'OPQ32r'."""
        try:
            from rapidfuzz import fuzz
        except Exception:
            return [0.0] * len(self._items)
        q = query.lower()
        boosts = []
        for a in self._items:
            name = a.name.lower()
            partial = fuzz.partial_ratio(q, name) / 100.0
            boosts.append(partial if partial >= 0.85 else 0.0)
        return boosts

    @staticmethod
    def _minmax(xs: list[float]) -> list[float]:
        lo, hi = min(xs), max(xs)
        if hi - lo < 1e-9:
            return [0.0 for _ in xs]
        return [(x - lo) / (hi - lo) for x in xs]

    # -- public API ---------------------------------------------------------
    def search(self, query: str, k: int = 20) -> list[Scored]:
        if not query.strip():
            return []
        q_tokens = _tokenize(query)
        lexical = self._minmax(self._bm25_scores(q_tokens))
        fuzzy = self._fuzzy_boost(query)
        if self._semantic is not None:
            semantic = self._minmax(self._semantic.scores(query))
            fused = [0.55 * l + 0.30 * s + 0.15 * f
                     for l, s, f in zip(lexical, semantic, fuzzy)]
        else:
            fused = [0.75 * l + 0.25 * f for l, f in zip(lexical, fuzzy)]

        ranked = sorted(
            (Scored(a, s) for a, s in zip(self._items, fused)),
            key=lambda x: x.score,
            reverse=True,
        )
        return [r for r in ranked if r.score > 0][:k]

    def multi_search(self, queries: list[str], k: int = 20) -> list[Scored]:
        """Union of several sub-queries (one per extracted skill/facet),
        keeping each item's best score. Lets a multi-skill JD pull the right
        test for every skill instead of one blended query washing them out."""
        best: dict[str, Scored] = {}
        for q in queries:
            for hit in self.search(q, k=k):
                cur = best.get(hit.assessment.id)
                if cur is None or hit.score > cur.score:
                    best[hit.assessment.id] = hit
        return sorted(best.values(), key=lambda x: x.score, reverse=True)[:k]


class _SemanticIndex:
    """Optional dense-embedding index. Isolated so its heavy deps never break
    the lexical path."""

    def __init__(self, model, matrix, items):
        self._model = model
        self._matrix = matrix
        self._items = items

    @classmethod
    def try_build(cls, assessments: list[Assessment]):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            docs = [a.search_document() for a in assessments]
            mat = model.encode(docs, normalize_embeddings=True)
            return cls(model, np.asarray(mat), assessments)
        except Exception as e:
            logger.info("Semantic index not built: %s", e)
            return None

    def scores(self, query: str) -> list[float]:
        import numpy as np

        q = self._model.encode([query], normalize_embeddings=True)
        return list(np.asarray(q).dot(self._matrix.T)[0])
