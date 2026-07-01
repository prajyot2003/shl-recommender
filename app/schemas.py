"""API schemas. The response shape is the assignment's non-negotiable contract:
{ reply: str, recommendations: [{name, url, test_type}] (0..10), end_of_conversation: bool }
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must contain at least one item")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=10)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
