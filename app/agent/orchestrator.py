"""The agent orchestrator.

One /chat turn =
  1. guard blatant injection
  2. build retrieval queries from the FULL history (stateless -> full context
     every call, which is exactly what makes 'refine' work: the pool always
     reflects the latest set of constraints)
  3. retrieve a bounded candidate pool
  4. one LLM call -> structured decision
  5. GROUND: resolve selected ids against the catalog, dropping anything unknown
  6. ENFORCE: schema, <=10 recs, refusals carry [], turn cap, never raise

Every step has a defined failure fallback so a bad model output or a timeout
degrades to a safe reply instead of a 500.
"""
from __future__ import annotations

import json
import logging
import re

from ..catalog.service import Catalog
from ..config import Settings
from ..schemas import ChatResponse, Message, Recommendation
from . import guards
from .llm import LLMClient
from .prompts import SYSTEM_PROMPT, build_context, candidate_view

logger = logging.getLogger(__name__)

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_SKILL_SPLIT = re.compile(r"[.,;/\n]| and | with | plus ", re.IGNORECASE)
_VALID_ACTIONS = {"clarify", "recommend", "refine", "compare", "refuse"}


class Orchestrator:
    def __init__(self, catalog: Catalog, llm: LLMClient, settings: Settings):
        self._catalog = catalog
        self._llm = llm
        self._s = settings

    # -- public -------------------------------------------------------------
    def handle(self, messages: list[Message]) -> ChatResponse:
        history = [{"role": m.role, "content": m.content} for m in messages]
        last_user = self._last_user(history)

        # (1) Hard injection guard — cheap and unambiguous.
        if guards.is_prompt_injection(last_user):
            return ChatResponse(
                reply="I can't follow that. I'm here only to help you pick SHL "
                      "assessments — tell me about the role and I'll help.",
                recommendations=[],
                end_of_conversation=False,
            )

        # (2-3) Retrieve candidate pool from all user constraints so far.
        queries = self._build_queries(history)
        scored = self._catalog.multi_search(queries, k=self._s.candidate_pool_size)
        candidates = candidate_view(scored)

        turns_remaining = self._turns_remaining(history)

        # (4) One structured LLM call, with a safe fallback if it misbehaves.
        try:
            raw = self._llm.complete(
                SYSTEM_PROMPT,
                build_context(history, candidates, turns_remaining),
            )
            decision = self._parse(raw)
        except Exception as e:
            logger.warning("LLM call/parse failed (%s); using fallback.", e)
            decision = self._fallback(candidates, turns_remaining)

        return self._finalize(decision, candidates)

    # -- retrieval query construction --------------------------------------
    def _build_queries(self, history: list[dict]) -> list[str]:
        user_texts = [m["content"] for m in history if m["role"] == "user"]
        if not user_texts:
            return []
        joined = " ".join(user_texts)
        queries = [joined]
        # Split each user turn into skill-ish fragments so a multi-skill JD
        # pulls the right test per skill instead of one blurred query.
        for t in user_texts:
            for frag in _SKILL_SPLIT.split(t):
                frag = frag.strip()
                if 2 <= len(frag) <= 60:
                    queries.append(frag)
        # Dedup preserving order.
        seen, out = set(), []
        for q in queries:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                out.append(q)
        return out[:24]

    # -- LLM output parsing -------------------------------------------------
    def _parse(self, raw: str) -> dict:
        text = raw.strip()
        # Tolerate code fences / stray prose around the JSON.
        if "```" in text:
            text = text.replace("```json", "```").split("```")[1] \
                if text.count("```") >= 2 else text
        m = _JSON_OBJ.search(text)
        if not m:
            raise ValueError("no JSON object in LLM output")
        data = json.loads(m.group(0))
        action = str(data.get("action", "")).lower()
        if action not in _VALID_ACTIONS:
            action = "clarify"
        return {
            "action": action,
            "reply": str(data.get("reply", "")).strip(),
            "selected_ids": data.get("selected_ids") or [],
            "end_of_conversation": bool(data.get("end_of_conversation", False)),
        }

    def _fallback(self, candidates: list[dict], turns_remaining: int) -> dict:
        """Deterministic safety net when the LLM is unusable."""
        if candidates and turns_remaining <= self._s.max_turns:
            top = [c["id"] for c in candidates[:5]]
            return {
                "action": "recommend",
                "reply": "Here are the assessments that best match what you've "
                         "described so far.",
                "selected_ids": top,
                "end_of_conversation": False,
            }
        return {
            "action": "clarify",
            "reply": "Could you tell me the role and the key skills you're hiring "
                     "for? That lets me shortlist the right assessments.",
            "selected_ids": [],
            "end_of_conversation": False,
        }

    # -- grounding + schema enforcement ------------------------------------
    def _finalize(self, decision: dict, candidates: list[dict]) -> ChatResponse:
        action = decision["action"]
        reply = decision["reply"] or self._default_reply(action)
        end = decision["end_of_conversation"]

        # Refusals and clarifications never carry recommendations.
        if action in ("refuse", "clarify"):
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

        # GROUND: resolve every selected id/name to a real catalog item.
        allowed_ids = {c["id"] for c in candidates}
        recs: list[Recommendation] = []
        seen: set[str] = set()
        for ident in decision["selected_ids"]:
            ident = str(ident).strip()
            asset = self._catalog.resolve(ident)
            # Prefer items that came from the shown pool; but still accept a
            # valid catalog resolution (model may echo a name we indexed).
            if asset is None:
                continue
            if asset.id in seen:
                continue
            seen.add(asset.id)
            recs.append(Recommendation(**asset.to_recommendation()))
            if len(recs) >= self._s.max_recommendations:
                break

        # If the model claimed to recommend but grounding produced nothing,
        # fail safe to a clarify rather than an empty "here's your shortlist".
        if not recs:
            return ChatResponse(
                reply="I want to ground this in the catalog before recommending. "
                      "Could you confirm the role and must-have skills?",
                recommendations=[],
                end_of_conversation=False,
            )

        return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _last_user(history: list[dict]) -> str:
        for m in reversed(history):
            if m["role"] == "user":
                return m["content"]
        return ""

    def _turns_remaining(self, history: list[dict]) -> int:
        used = sum(1 for m in history if m["role"] in ("user", "assistant"))
        return max(0, self._s.max_turns - used)

    @staticmethod
    def _default_reply(action: str) -> str:
        return {
            "recommend": "Here is a shortlist based on what you've described.",
            "refine": "Updated the shortlist with your changes.",
            "compare": "Here's how those assessments differ, based on the catalog.",
        }.get(action, "How can I help you select SHL assessments?")
