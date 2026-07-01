"""The Catalog service: single source of truth the agent grounds against."""
from __future__ import annotations

import logging

from .loader import load_catalog
from .models import Assessment
from ..retrieval.retriever import Retriever, Scored

logger = logging.getLogger(__name__)


class Catalog:
    def __init__(self, assessments: list[Assessment], use_semantic: bool = False):
        self._items = assessments
        self._by_id = {a.id: a for a in assessments}
        # Normalized-name index for resolving LLM-selected names back to items.
        self._by_name = {self._norm(a.name): a for a in assessments}
        self._retriever = Retriever(assessments, use_semantic=use_semantic)

    @classmethod
    def bootstrap(cls, source_url: str | None, use_semantic: bool = False) -> "Catalog":
        return cls(load_catalog(source_url), use_semantic=use_semantic)

    @staticmethod
    def _norm(name: str) -> str:
        return " ".join(name.lower().split())

    def __len__(self) -> int:
        return len(self._items)

    def get(self, aid: str) -> Assessment | None:
        return self._by_id.get(aid)

    def resolve(self, identifier: str) -> Assessment | None:
        """Map an id OR a (possibly slightly off) name back to a catalog item.
        This is the gate that guarantees emitted URLs come from the catalog."""
        if identifier in self._by_id:
            return self._by_id[identifier]
        return self._by_name.get(self._norm(identifier))

    def search(self, query: str, k: int = 20) -> list[Scored]:
        return self._retriever.search(query, k=k)

    def multi_search(self, queries: list[str], k: int = 20) -> list[Scored]:
        return self._retriever.multi_search(queries, k=k)
