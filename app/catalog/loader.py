"""Catalog ingestion.

Responsibilities:
  1. Obtain raw catalog data (live URL -> local cache -> bundled snapshot).
  2. Normalize heterogeneous field names into ``Assessment`` objects.
  3. Filter to Individual Test Solutions (Job Solutions are out of scope).

We never fail hard on ingestion: if the network is down we serve the last
cached copy, and if that is missing we serve the bundled snapshot. A recommender
that 500s on cold start scores zero, so degradation must be graceful.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .models import Assessment, labels_for

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_PATH = _DATA_DIR / "catalog_cache.json"
_SNAPSHOT_PATH = _DATA_DIR / "catalog_sample.json"

# Field-name aliases seen across possible catalog exports. Order = priority.
_NAME_KEYS = ("name", "title", "assessment_name", "product_name")
_URL_KEYS = ("link", "url", "product_url", "href")   # live catalog uses "link"
_TYPE_KEYS = ("test_type", "test_types", "type", "types", "test_type_codes")
_KEYS_KEYS = ("keys", "key", "categories")           # human labels in live catalog
_DESC_KEYS = ("description", "desc", "summary", "long_description", "details")
_DUR_KEYS = ("duration", "assessment_length", "length", "completion_time")
_LANG_KEYS = ("languages", "language", "langs")
_LEVEL_KEYS = ("job_levels", "job_level", "levels")


def _first(d: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    # Split on common delimiters for a stringly-typed field.
    return [p.strip() for p in re.split(r"[;,/|]", str(v)) if p.strip()]


def _slug_from_url(url: str, name: str) -> str:
    m = re.search(r"/view/([^/]+)/?", url)
    if m:
        return m.group(1).lower()
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"


def _parse_test_types(raw: Any) -> tuple[str, ...]:
    """Accept 'K,S', ['K','S'], 'Knowledge & Skills', etc. -> ('K','S')."""
    label_to_code = {
        "ability & aptitude": "A",
        "biodata & situational judgment": "B",
        "competencies": "C",
        "development & 360": "D",
        "assessment exercises": "E",
        "knowledge & skills": "K",
        "personality & behavior": "P",
        "personality & behaviour": "P",
        "simulations": "S",
    }
    codes: list[str] = []
    for token in _as_list(raw):
        t = token.strip()
        if len(t) == 1 and t.upper() in "ABCDEKPS":
            code = t.upper()
        else:
            code = label_to_code.get(t.lower())
        if code and code not in codes:
            codes.append(code)
    return tuple(codes)


def _looks_like_job_solution(d: dict[str, Any]) -> bool:
    """Best-effort filter for pre-packaged Job Solutions (out of scope)."""
    category = str(_first(d, ("category", "solution_type", "catalog"), "")).lower()
    if "job" in category and "solution" in category:
        return True
    url = str(_first(d, _URL_KEYS, "")).lower()
    return "job-solutions" in url or "prepackaged" in url


def _bool_from(v: Any) -> bool | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("yes", "true", "1", "y"):
        return True
    if s in ("no", "false", "0", "n"):
        return False
    return None


def normalize(raw_items: list[dict[str, Any]]) -> list[Assessment]:
    seen: set[str] = set()
    out: list[Assessment] = []
    for d in raw_items:
        if not isinstance(d, dict):
            continue
        if _looks_like_job_solution(d):
            continue
        name = _first(d, _NAME_KEYS)
        url = _first(d, _URL_KEYS)
        if not name or not url:
            continue  # unusable row, skip rather than emit a broken rec

        # Test-type codes: prefer an explicit code field; otherwise derive from
        # the human "keys" labels (the live SHL catalog only ships labels).
        label_source = _first(d, _KEYS_KEYS)
        types = _parse_test_types(_first(d, _TYPE_KEYS)) or _parse_test_types(label_source)
        # Human-readable keys: keep the source labels verbatim if present.
        human_keys = tuple(_as_list(label_source)) or tuple(labels_for(types))

        aid = _slug_from_url(str(url), str(name))
        if aid in seen:
            continue
        seen.add(aid)
        out.append(
            Assessment(
                id=aid,
                name=str(name).strip(),
                url=str(url).strip(),
                test_types=types,
                description=str(_first(d, _DESC_KEYS, "")).strip(),
                keys=human_keys,
                duration=str(_first(d, _DUR_KEYS, "")).strip(),
                languages=tuple(_as_list(_first(d, _LANG_KEYS))),
                job_levels=tuple(_as_list(_first(d, _LEVEL_KEYS))),
                remote_testing=_bool_from(d.get("remote")),
                adaptive=_bool_from(d.get("adaptive")),
            )
        )
    return out


def _read_json(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        logger.debug("No file at %s (expected for optional cache).", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    # Accept either a bare list or {"items": [...]} / {"products": [...]}.
    if isinstance(data, dict):
        for k in ("items", "products", "assessments", "data"):
            if isinstance(data.get(k), list):
                return data[k]
        return None
    return data if isinstance(data, list) else None


def _fetch_live(url: str, timeout: float) -> list[dict[str, Any]] | None:
    try:
        import httpx

        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # network, JSON, HTTP — all non-fatal
        logger.warning("Live catalog fetch failed (%s); falling back.", e)
        return None
    if isinstance(data, dict):
        for k in ("items", "products", "assessments", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    if isinstance(data, list):
        try:
            _CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass
        return data
    return None


def load_catalog(source_url: str | None = None, timeout: float = 15.0) -> list[Assessment]:
    """Load and normalize the catalog, trying live -> cache -> snapshot."""
    raw: list[dict[str, Any]] | None = None
    if source_url:
        raw = _fetch_live(source_url, timeout)
    if raw is None:
        raw = _read_json(_CACHE_PATH)
    if raw is None:
        raw = _read_json(_SNAPSHOT_PATH)
    if raw is None:
        raise RuntimeError("No catalog source available (live, cache, snapshot all failed).")
    items = normalize(raw)
    if not items:
        raise RuntimeError("Catalog normalized to zero items — check source schema.")
    logger.info("Loaded %d assessments.", len(items))
    return items
