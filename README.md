---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# SHL Conversational Assessment Recommender

A stateless FastAPI agent that takes a recruiter from a vague hiring intent to a
grounded shortlist of SHL **Individual Test Solutions** through conversation —
clarifying, recommending, refining, and comparing, while refusing anything
off-scope and never inventing an assessment or URL.

## Endpoints
- `GET /health` → `{"status": "ok"}` (HTTP 200)
- `POST /chat` → stateless; takes the full conversation history, returns the next
  reply plus (when appropriate) a shortlist.

```jsonc
// request
{ "messages": [ {"role":"user","content":"Hiring a mid-level Java dev"} ] }
// response
{ "reply": "...", "recommendations": [ {"name":"...","url":"https://www.shl.com/...","test_type":"K"} ], "end_of_conversation": false }
```
`recommendations` is `[]` while clarifying or refusing, and 1–10 items once the
agent commits to a shortlist.

## Run locally
```bash
pip install -r requirements.txt
# Offline mode (heuristic StubLLM, catalog snapshot) — no keys needed:
uvicorn app.main:app --reload
# With a real model (recommended): any OpenAI-compatible endpoint
export LLM_API_KEY=...            # e.g. a Groq key
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_MODEL=llama-3.3-70b-versatile
uvicorn app.main:app
```

## Tests & local eval
```bash
python tests/test_agent.py           # grounding, schema, refusal, turn-cap, resilience
python -m app.eval.replay            # Recall@10 + probes vs. local personas (if present)
```

## Configuration (all env-driven)
| Var | Default | Purpose |
|-----|---------|---------|
| `SHL_CATALOG_URL` | live JSON URL | catalog source; falls back to cache then bundled snapshot |
| `SHL_USE_SEMANTIC` | `0` | set `1` to fuse dense embeddings (needs `sentence-transformers`) |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | Groq | OpenAI-compatible LLM |
| `LLM_TIMEOUT` | `20` | per-call timeout (under the 30s cap) |
| `MAX_TURNS` | `8` | honored turn budget |

## Architecture
```
/chat → Orchestrator
        ├─ guards          blatant injection pre-filter
        ├─ Catalog.multi_search   retrieve candidate pool from FULL history
        ├─ LLM (1 call)    structured decision: action + reply + selected_ids
        ├─ GROUND          resolve ids→catalog items, drop anything unknown
        └─ ENFORCE         schema, ≤10 recs, refuse/clarify ⇒ [], never 500
```
- **Grounding boundary:** the agent may only pick from a pool we inject, and every
  selected id is re-resolved against the catalog before it can be emitted. That is
  what makes "URLs only from the catalog" a guarantee, not a hope.
- **Stateless refine:** every call rebuilds the candidate pool from the whole
  history, so "add personality tests" / "drop REST" update the shortlist naturally.
- **Graceful degradation:** live-catalog → cache → snapshot; bad/slow LLM output →
  deterministic fallback. The service stays schema-valid and 200 in every path.

## Deploy
Render (`render.yaml`), any Dockerfile host, or Fly/Railway/HF Spaces. Set
`LLM_API_KEY` in the host's secrets. `healthCheckPath: /health`.
