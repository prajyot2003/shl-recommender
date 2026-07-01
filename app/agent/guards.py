"""Cheap, conservative pre-filters.

These catch *blatant* prompt-injection and hard off-topic asks before we spend
an LLM call. They are intentionally narrow: the primary scope enforcement lives
in the system prompt (the traces show nuanced cases like a legal question raised
mid-conversation that still must keep the existing shortlist). Over-eager rules
here would cause false refusals and tank the behavior-probe score.
"""
from __future__ import annotations

import re

_INJECTION = re.compile(
    r"(ignore\s+(all\s+|your\s+|the\s+|previous\s+)?instruction"
    r"|disregard\s+(the\s+|all\s+)?(above|previous)"
    r"|reveal\s+(your\s+)?(system\s+)?prompt"
    r"|you\s+are\s+now\s+"
    r"|act\s+as\s+(?:a\s+)?(?:dan|jailbreak)"
    r"|print\s+your\s+(system\s+)?instructions)",
    re.IGNORECASE,
)


def is_prompt_injection(text: str) -> bool:
    return bool(_INJECTION.search(text or ""))
