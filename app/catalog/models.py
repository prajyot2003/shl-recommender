"""Domain models for the SHL catalog.

Kept deliberately small and immutable. Every recommendation the agent can ever
emit is one of these objects, looked up from the in-memory catalog by ``id``.
That single invariant is what makes "URLs only come from the catalog"
enforceable rather than aspirational.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# SHL's canonical test-type codes. Used to (a) render human-readable "keys"
# when the source data only gives codes, and (b) let the agent reason about
# balance of a battery (e.g. "you have K tests but no P").
TEST_TYPE_LABELS: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def labels_for(codes: Iterable[str]) -> list[str]:
    """Map a set of type codes to their human labels, ignoring unknowns."""
    out: list[str] = []
    for c in codes:
        c = c.strip().upper()
        if c in TEST_TYPE_LABELS and TEST_TYPE_LABELS[c] not in out:
            out.append(TEST_TYPE_LABELS[c])
    return out


@dataclass(frozen=True)
class Assessment:
    """One catalog item (an SHL Individual Test Solution)."""

    id: str                       # stable slug derived from the URL
    name: str
    url: str
    test_types: tuple[str, ...]   # e.g. ("K",) or ("K", "S")
    description: str = ""
    keys: tuple[str, ...] = field(default_factory=tuple)   # human labels
    duration: str = ""
    languages: tuple[str, ...] = field(default_factory=tuple)
    remote_testing: bool | None = None
    adaptive: bool | None = None
    job_levels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def test_type_str(self) -> str:
        """The compact code string used in the API response, e.g. 'K,S'."""
        return ",".join(self.test_types)

    def search_document(self) -> str:
        """Flat text blob fed to the lexical index."""
        parts = [
            self.name,
            self.description,
            " ".join(self.keys),
            " ".join(labels_for(self.test_types)),
            " ".join(self.job_levels),
        ]
        return "  ".join(p for p in parts if p)

    def to_recommendation(self) -> dict[str, str]:
        """Exactly the three fields the assignment schema requires. No more:
        extra keys risk breaking a strict evaluator."""
        return {"name": self.name, "url": self.url, "test_type": self.test_type_str}
