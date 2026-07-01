"""LLM clients.

`OpenAICompatibleLLM` talks to any OpenAI-style /chat/completions endpoint
(Groq is the default — chosen for low latency under the 30s call cap). A single
call per /chat turn keeps us comfortably inside the timeout.

`StubLLM` is a dependency-free, deterministic stand-in that mimics the trace
behaviours (clarify / recommend / refine / compare / refuse). It exists so the
whole service — routing, grounding, schema, guards, turn-cap — can be run and
unit-tested with no API key. It is NOT meant to match a real model's quality.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def complete(self, system: str, user: str) -> str:
        import httpx

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "max_tokens": 900,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class StubLLM:
    """Offline heuristic decision-maker used for local dev and tests."""

    _LEGAL = re.compile(
        r"\b(legal(ly)?|lawsuit|liabilit|compliance (require|oblig)|"
        r"required under|satisf(y|ies) .*requirement|counsel|regulat)", re.I)
    _INJECT = re.compile(
        r"(ignore (all |your |previous )?instruction|system prompt|"
        r"you are now|disregard .*above|reveal .*prompt|jailbreak)", re.I)
    _OFFTOPIC = re.compile(
        r"\b(salary|how much should i pay|write (a |the )?job description|"
        r"weather|stock price|tell me a joke)\b", re.I)
    _VAGUE = re.compile(
        r"^\W*(i (need|want)|we need|looking for|need)\s+"
        r"(an?\s+)?(assessment|solution|test|something)\W*$", re.I)

    def complete(self, system: str, user: str) -> str:
        # `user` is the JSON context the orchestrator builds. Parse it back.
        try:
            ctx = json.loads(user)
        except json.JSONDecodeError:
            ctx = {}
        history = ctx.get("history", [])
        candidates = ctx.get("candidates", [])
        turns_remaining = ctx.get("turns_remaining", 8)
        last_user = ""
        for m in reversed(history):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        if self._INJECT.search(last_user) or self._OFFTOPIC.search(last_user):
            return self._json("refuse",
                              "I can only help with selecting SHL assessments. "
                              "I can't help with that request.", [], False)
        if self._LEGAL.search(last_user):
            return self._json("refuse",
                              "That's a legal/compliance question outside what I can advise on. "
                              "Your legal team is the right resource. I can help pick assessments.",
                              [], False)

        # Vague opener with no candidates -> clarify.
        if (self._VAGUE.search(last_user.strip()) or not candidates) and turns_remaining > 2:
            return self._json("clarify",
                              "Happy to help narrow that down. Who is the role for, "
                              "and what should the candidate be able to do?", [], False)

        top = [c["id"] for c in candidates[:5]]
        if not top:
            return self._json("clarify",
                              "Could you share the role and key skills you're hiring for?",
                              [], False)
        return self._json("recommend",
                          f"Here are {len(top)} assessments that fit what you described.",
                          top, False)

    @staticmethod
    def _json(action, reply, ids, eoc) -> str:
        return json.dumps({
            "action": action,
            "reply": reply,
            "selected_ids": ids,
            "end_of_conversation": eoc,
        })


def build_llm(settings) -> LLMClient:
    """Real client if an API key is present, otherwise the offline stub."""
    if settings.llm_api_key:
        return OpenAICompatibleLLM(
            settings.llm_base_url, settings.llm_api_key,
            settings.llm_model, settings.llm_timeout)
    logger.warning("No LLM_API_KEY set — using StubLLM (heuristic, offline).")
    return StubLLM()
